"""
Demo 5: Conditional Flow-Matching / Diffusion Action Chunk Policy (RTX 4090)
============================================================================
자율주행의 'DiffusionDrive' 및 최신 로봇 VLA('Pi-0', 'Diffusion Policy')의 핵심 엔진인
생성형 궤적 플래너(Generative Trajectory Planner via Flow-Matching / Diffusion) 실습입니다.

[왜 L2/L1 직접 회귀 대신 Diffusion / Flow-Matching을 쓸까?]
1. 다중 경로(Multi-modal) 해결:
   앞에 장애물이 있을 때 왼쪽 회피 궤적과 오른쪽 회피 궤적이 데이터에 반반 섞여 있으면,
   일반 회귀(Regression) 모델은 두 궤적의 평균인 '가운데(장애물 충돌)'를 출력합니다.
2. 생성형 디노이징(Denoising):
   반면 Flow-Matching / Diffusion은 순수 가우시안 노이즈 A^(0) ~ N(0, I) 에서 출발하여
   시각/상태 조건(Condition)에 맞춰 속도장(Velocity Field) v_theta(A^(tau), tau | img, state)를 따라
   10스텝 적분(tau: 0 -> 1)함으로써 하나의 일관되고 매끄러운 미래 궤적 청크 A^(1) (50 x 14)를 생성합니다!
"""
import math
import time
from pathlib import Path

import cv2
import gymnasium as gym
import gym_aloha  # noqa: F401
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from demo_2_act_pretrained import format_and_normalize_obs, load_norm_stats
from lerobot.policies.act.modeling_act import ACTPolicy


class SinusoidalTimeEmbedding(nn.Module):
    """디노이징 타임스텝 tau in [0, 1]을 고차원 벡터로 변환하는 포지셔널 임베딩"""
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=tau.device) / half)
        args = tau[:, None] * freqs[None, :] * 1000.0
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowMatchingActionPolicy(nn.Module):
    """
    조건부 Flow-Matching (Diffusion 계열) Action Chunk 생성 네트워크
    - 조건(Condition): 카메라 이미지(128x128) + 현재 14축 관절 각도(s_t)
    - 입력: 노이즈가 섞인 궤적 청크 A_tau (B, K, 14) + 디노이징 시점 tau (B,)
    - 출력: 노이즈를 정답 궤적으로 밀어주는 속도 벡터장 v_theta (B, K, 14)
    """
    def __init__(self, state_dim: int = 14, action_dim: int = 14, chunk_size: int = 50):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),  # 128 * 4 * 4 = 2048
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(2048 + state_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(128),
            nn.Linear(128, 512),
            nn.GELU(),
        )

        # 노이즈 섞인 Action Chunk (K * 14 = 700) + 조건 임베딩(512) + 시간 임베딩(512) -> 속도장 (K * 14)
        flat_chunk_dim = chunk_size * action_dim
        self.in_proj = nn.Linear(flat_chunk_dim + 512 + 512, 1024)
        self.block1 = nn.Sequential(nn.LayerNorm(1024), nn.GELU(), nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Linear(1024, 1024))
        self.block2 = nn.Sequential(nn.LayerNorm(1024), nn.GELU(), nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Linear(1024, 1024))
        self.out_proj = nn.Linear(1024, flat_chunk_dim)

    def encode_condition(self, img_128: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        vis = self.vision_encoder(img_128)
        return self.cond_proj(torch.cat([vis, state], dim=-1))

    def forward(self, noisy_chunk: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b = noisy_chunk.shape[0]
        flat_noisy = noisy_chunk.view(b, -1)
        t_emb = self.time_mlp(tau)
        h = self.in_proj(torch.cat([flat_noisy, cond, t_emb], dim=-1))
        h = h + self.block1(h)
        h = h + self.block2(h)
        vel = self.out_proj(h)
        return vel.view(b, self.chunk_size, self.action_dim)

    @torch.no_grad()
    def sample_trajectory(self, img_128: torch.Tensor, state: torch.Tensor, steps: int = 10, return_history: bool = False):
        """
        순수 가우시안 노이즈(tau=0)에서 시작해 Euler ODE 적분(steps회)으로
        최종 미래 궤적 청크(tau=1)를 깎아냅니다 (Denoising 과정).
        """
        b = img_128.shape[0]
        device = img_128.device
        cond = self.encode_condition(img_128, state)

        # tau = 0.0: 초기 가우시안 분포에서 출발 (추론 시 결정론적 시드 고정으로 흔들림 방지)
        gen = torch.Generator(device=device).manual_seed(777)
        x_tau = torch.randn((b, self.chunk_size, self.action_dim), generator=gen, device=device) * 0.5
        dt = 1.0 / steps
        history = [x_tau.clone()]

        for s in range(steps):
            tau = torch.full((b,), s * dt, device=device)
            velocity = self.forward(x_tau, tau, cond)
            x_tau = x_tau + velocity * dt
            history.append(x_tau.clone())

        if return_history:
            return x_tau, history
        return x_tau


def preprocess_img_128(rgb_numpy: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(rgb_numpy).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    return torch.nn.functional.interpolate(t, size=(128, 128), mode="bilinear", align_corners=False)


def save_denoising_visualization(history: list[torch.Tensor], target_chunk: np.ndarray, act_mean: torch.Tensor, act_std: torch.Tensor, out_path: Path):
    """노이즈(tau=0.0)가 디노이징 스텝을 거치며 매끄러운 로봇 관절 궤적(tau=1.0)으로 변하는 과정을 시각화합니다."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), dpi=160)
    fig.patch.set_facecolor("#0f172a")

    snap_indices = [0, 2, 5, 10]
    titles = [
        "Step 0 (τ = 0.0)\nPure Gaussian Noise A^(0)",
        "Step 2 (τ = 0.2)\nCoarse Direction Forming",
        "Step 5 (τ = 0.5)\nTrajectory Shape Emerging",
        "Step 10 (τ = 1.0)\nClean Action Chunk A^(1)",
    ]

    mean_np = act_mean.squeeze().cpu().numpy()
    std_np = act_std.squeeze().cpu().numpy()
    t_steps = np.arange(target_chunk.shape[0])

    for ax, idx, title in zip(axes, snap_indices, titles):
        ax.set_facecolor("#1e293b")
        ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")
        chunk_unnorm = history[idx].squeeze(0).cpu().numpy() * std_np + mean_np

        # 정답 궤적 (점선)
        ax.plot(t_steps, target_chunk[:, 0], "--", color="#facc15", lw=1.8, alpha=0.8, label="Target Left Waist")
        ax.plot(t_steps, target_chunk[:, 7], "--", color="#fb7185", lw=1.8, alpha=0.8, label="Target Right Waist")

        # 현재 디노이징 단계의 궤적 (실선)
        ax.plot(t_steps, chunk_unnorm[:, 0], color="#38bdf8", lw=2.5, label="Denoised Left Arm")
        ax.plot(t_steps, chunk_unnorm[:, 7], color="#4ade80", lw=2.5, label="Denoised Right Arm")

        ax.set_title(title, fontsize=11, fontweight="bold", color="#f8fafc")
        ax.set_xlabel("Future Horizon k (0..50 steps)", fontsize=10, color="#cbd5e1")
        ax.set_ylim(-0.6, 0.6)
        if idx == 0:
            ax.set_ylabel("Joint Angle (rad)", fontsize=10, color="#cbd5e1")
            ax.legend(loc="lower left", fontsize=8.5, facecolor="#0f172a")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_size = 50
    steps_per_ep = 400
    seeds = [1, 2, 3]

    print("=" * 74)
    print("[1/4] 시연 데이터 수집 중 (Flow-Matching 생성 모델 학습용)...")
    print("=" * 74)

    teacher_id = "lerobot/act_aloha_sim_transfer_cube_human"
    teacher = ACTPolicy.from_pretrained(teacher_id)
    teacher.config.n_action_steps = 20
    teacher.to(device).eval()
    teacher_stats = load_norm_stats(teacher_id, device)

    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", max_episode_steps=steps_per_ep)

    demo_imgs, demo_states, demo_chunks = [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        teacher.reset()
        ep_imgs, ep_states, ep_actions = [], [], []
        for _ in range(steps_per_ep):
            batch = format_and_normalize_obs(obs, teacher_stats, device)
            with torch.inference_mode():
                act_norm = teacher.select_action(batch)
                act = (act_norm * teacher_stats["action_std"] + teacher_stats["action_mean"]).squeeze(0).cpu().numpy()
            ep_imgs.append(preprocess_img_128(obs["pixels"]["top"], device).squeeze(0).cpu())
            ep_states.append(torch.from_numpy(obs["agent_pos"]).float())
            ep_actions.append(torch.from_numpy(act).float())
            obs, _, _, _, _ = env.step(act)

        ep_actions_padded = ep_actions + [ep_actions[-1]] * chunk_size
        ep_actions_t = torch.stack(ep_actions_padded)
        for t in range(steps_per_ep):
            demo_imgs.append(ep_imgs[t])
            demo_states.append(ep_states[t])
            demo_chunks.append(ep_actions_t[t : t + chunk_size])

    imgs_t = torch.stack(demo_imgs).to(device)
    states_t = torch.stack(demo_states).to(device)
    chunks_t = torch.stack(demo_chunks).to(device)

    state_mean = states_t.mean(dim=0, keepdim=True)
    state_std = states_t.std(dim=0, keepdim=True) + 1e-5
    act_mean = chunks_t.mean(dim=(0, 1), keepdim=True)
    act_std = chunks_t.std(dim=(0, 1), keepdim=True) + 1e-5

    states_norm = (states_t - state_mean) / state_std
    chunks_norm = (chunks_t - act_mean) / act_std

    print("\n" + "=" * 74)
    print("[2/4] RTX 4090에서 Flow-Matching (Diffusion ODE) 궤적 생성기 학습 시작!")
    print("      수식: x_tau = (1 - tau) * noise + tau * x_1  -->  속도장 v = x_1 - noise 예측")
    print("=" * 74)

    model = FlowMatchingActionPolicy(state_dim=14, action_dim=14, chunk_size=chunk_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 160
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    batch_size = 64
    num_samples = imgs_t.shape[0]
    t0 = time.perf_counter()

    model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_samples, device=device)
        epoch_loss = 0.0
        steps = 0
        for i in range(0, num_samples, batch_size):
            idx = perm[i : i + batch_size]
            noisy_st = states_norm[idx] + 0.02 * torch.randn_like(states_norm[idx])
            cond = model.encode_condition(imgs_t[idx], noisy_st)

            # 비전 인코더 특징(cond)을 4번 재사용해 다양한 디노이징 시점 tau in [0, 1] 동시 학습!
            x_1 = chunks_norm[idx].repeat(4, 1, 1)
            cond_rep = cond.repeat(4, 1)
            x_0 = torch.randn_like(x_1) * 0.5
            tau = torch.rand(x_1.shape[0], device=device)

            tau_view = tau[:, None, None]
            x_tau = (1.0 - tau_view) * x_0 + tau_view * x_1
            target_velocity = x_1 - x_0

            pred_velocity = model(x_tau, tau, cond_rep)

            loss = nn.functional.mse_loss(pred_velocity, target_velocity)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{epochs} | Flow Velocity MSE Loss: {epoch_loss / steps:.4f} | 경과 시간: {time.perf_counter() - t0:.1f}s")

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 디노이징 과정(tau=0.0 -> 1.0) 시각화 저장
    model.eval()
    sample_idx = 260  # 공중 전달 직전 스텝
    _, hist = model.sample_trajectory(
        imgs_t[sample_idx : sample_idx + 1],
        states_norm[sample_idx : sample_idx + 1],
        steps=10,
        return_history=True,
    )
    denoise_fig_path = out_dir / "flow_matching_denoising.png"
    save_denoising_visualization(hist, chunks_t[sample_idx].cpu().numpy(), act_mean, act_std, denoise_fig_path)
    print(f"\n[3/4] 노이즈 -> 로봇 궤적 디노이징 시각화 저장 완료: {denoise_fig_path}")

    print("[4/4] MuJoCo에서 10-step ODE 디노이징으로 양팔 큐브 전달 실행 중...")
    obs, _ = env.reset(seed=1)
    frames = []
    max_reward = 0.0
    success_step = None
    all_time_actions = np.zeros((steps_per_ep, steps_per_ep + chunk_size, 14), dtype=np.float32)

    for t in range(steps_per_ep):
        img_in = preprocess_img_128(obs["pixels"]["top"], device)
        st_in = (torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device) - state_mean) / state_std
        pred_norm = model.sample_trajectory(img_in, st_in, steps=10)
        pred_chunk = (pred_norm * act_std + act_mean).squeeze(0).cpu().numpy()

        all_time_actions[t, t : t + chunk_size] = pred_chunk
        start_idx = max(0, t - chunk_size + 1)
        actions_curr = all_time_actions[start_idx : t + 1, t]
        w = np.exp(-0.01 * np.arange(len(actions_curr))[::-1])
        w /= w.sum()
        action = (actions_curr * w[:, None]).sum(axis=0)

        obs, reward, _, _, _ = env.step(action)
        max_reward = max(max_reward, float(reward))
        if float(reward) >= 4.0 and success_step is None:
            success_step = t + 1
            print(f"  ★ Step {success_step:03d}: Flow-Matching 생성형 정책이 양팔 큐브 전달 성공! (Reward = {reward})")

        canvas = obs["pixels"]["top"].copy()
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 68), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.68, canvas, 0.32, 0, canvas)
        cv2.putText(
            canvas,
            f"Flow-Matching Policy (10-Step ODE Denoising) | Step: {t+1:03d}/400",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (56, 189, 248),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Reward: {float(reward):.1f} / 4.0 | {'SUCCESS!' if float(reward) >= 4.0 else 'Denoising Trajectory...'}",
            (14, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (74, 222, 128) if float(reward) >= 4.0 else (250, 204, 21),
            2,
            cv2.LINE_AA,
        )
        frames.append(canvas)

    env.close()
    video_path = out_dir / "aloha_flow_matching.mp4"
    imageio.mimsave(str(video_path), frames, fps=50)
    print("=" * 74)
    print(f"Flow-Matching 추론 완료! 최고 보상: {max_reward:.1f} / 4.0")
    print(f"결과 영상: {video_path}")
    print("=" * 74)


if __name__ == "__main__":
    main()
