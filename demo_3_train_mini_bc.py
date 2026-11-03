"""
Demo 3: Train Your Own Action-Chunking Imitation Policy from Scratch (RTX 4090)
===============================================================================
ACT나 Diffusion Policy 같은 최신 모방학습(Imitation Learning)의 핵심 개념인
'Action Chunking + Temporal Ensembling'이 내부적으로 어떻게 작동하는지 실습하는 코드입니다.

[핵심 실험 포인트]
단순 모방학습(Behavior Cloning)은 미세한 오차가 누적되는 Covariate Shift 문제에 취약합니다.
이를 극복하기 위해:
1) 미래 50스텝의 행동 묶음(Action Chunk)을 한 번에 예측하고,
2) 이전 스텝들에서 예측한 미래 행동들과 현재 예측한 행동을 지수 가중 평균하는
   'Temporal Ensembling (시간적 앙상블)'을 적용해 양팔 큐브 공중 전달(Reward = 4.0)을 성공시킵니다!
"""
import time
from pathlib import Path

import cv2
import gymnasium as gym
import gym_aloha  # noqa: F401
import imageio
import numpy as np
import torch
import torch.nn as nn
from demo_2_act_pretrained import format_and_normalize_obs, load_norm_stats
from lerobot.policies.act.modeling_act import ACTPolicy


class ActionChunkingNet(nn.Module):
    """
    시각 이미지(CNN 인코더, 128x128) + 현재 관절 각도(14-DoF)를 입력받아
    앞으로의 K 스텝(chunk_size=50) 관절 명령(50 x 14)을 한 번에 출력하는 네트워크
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
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),  # 256 * 4 * 4 = 4096 차원
        )

        self.policy_head = nn.Sequential(
            nn.Linear(4096 + state_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, chunk_size * action_dim),
        )

    def forward(self, img_128: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        vis_feat = self.vision_encoder(img_128)
        x = torch.cat([vis_feat, state], dim=-1)
        out = self.policy_head(x)
        return out.view(-1, self.chunk_size, self.action_dim)


def preprocess_img_128(rgb_numpy: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(rgb_numpy).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    return torch.nn.functional.interpolate(t, size=(128, 128), mode="bilinear", align_corners=False)


def label_frame(frame: np.ndarray, title: str, step: int, reward: float, color: tuple[int, int, int]) -> np.ndarray:
    canvas = frame.copy()
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 68), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
    cv2.putText(canvas, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Step: {step:03d} | Reward: {reward:.1f} / 4.0",
        (14, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return canvas


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_size = 50
    steps_per_ep = 400
    seeds = [1, 2, 3]

    print("=" * 72)
    print(f"[Step 1] 전문가(Teacher ACT) 시연 데이터 수집 중 ({len(seeds)} 에피소드)...")
    print("=" * 72)

    teacher_id = "lerobot/act_aloha_sim_transfer_cube_human"
    teacher = ACTPolicy.from_pretrained(teacher_id)
    teacher.config.n_action_steps = 20
    teacher.to(device).eval()
    teacher_stats = load_norm_stats(teacher_id, device)

    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", max_episode_steps=steps_per_ep)

    demo_imgs, demo_states, demo_action_chunks = [], [], []

    for ep_idx, seed in enumerate(seeds):
        obs, _ = env.reset(seed=seed)
        teacher.reset()
        ep_imgs, ep_states, ep_actions = [], [], []
        ep_max_reward = 0.0

        for _ in range(steps_per_ep):
            batch = format_and_normalize_obs(obs, teacher_stats, device)
            with torch.inference_mode():
                act_norm = teacher.select_action(batch)
                act = (act_norm * teacher_stats["action_std"] + teacher_stats["action_mean"]).squeeze(0).cpu().numpy()

            ep_imgs.append(preprocess_img_128(obs["pixels"]["top"], device).squeeze(0).cpu())
            ep_states.append(torch.from_numpy(obs["agent_pos"]).float())
            ep_actions.append(torch.from_numpy(act).float())

            obs, reward, _, _, _ = env.step(act)
            ep_max_reward = max(ep_max_reward, float(reward))

        # 마지막 스텝 이후 패딩을 추가해 모든 스텝 t(0..399)에 대해 chunk_size=50 타겟 생성
        ep_actions_padded = ep_actions + [ep_actions[-1]] * chunk_size
        ep_actions_tensor = torch.stack(ep_actions_padded)
        for t in range(steps_per_ep):
            demo_imgs.append(ep_imgs[t])
            demo_states.append(ep_states[t])
            demo_action_chunks.append(ep_actions_tensor[t : t + chunk_size])

        print(f"  - 데모 에피소드 {ep_idx + 1}/{len(seeds)} (seed={seed}, 최고보상={ep_max_reward:.1f}) 수집 완료 | 누적 샘플: {len(demo_imgs)}")

    imgs_t = torch.stack(demo_imgs).to(device)
    states_t = torch.stack(demo_states).to(device)
    chunks_t = torch.stack(demo_action_chunks).to(device)

    state_mean = states_t.mean(dim=0, keepdim=True)
    state_std = states_t.std(dim=0, keepdim=True) + 1e-5
    act_mean = chunks_t.mean(dim=(0, 1), keepdim=True)
    act_std = chunks_t.std(dim=(0, 1), keepdim=True) + 1e-5

    states_norm = (states_t - state_mean) / state_std
    chunks_norm = (chunks_t - act_mean) / act_std

    student = ActionChunkingNet(state_dim=14, action_dim=14, chunk_size=chunk_size).to(device)

    # 1) 학습 전(Untrained) 초기 신경망 상태에서의 움직임 기록
    print("\n[비교용] 학습 전(Untrained) 초기 신경망으로 로봇 실행 기록 중...")
    student.eval()
    obs, _ = env.reset(seed=1)
    untrained_frames = []
    for t in range(steps_per_ep):
        if t % 10 == 0:
            with torch.inference_mode():
                img_in = preprocess_img_128(obs["pixels"]["top"], device)
                st_in = (torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device) - state_mean) / state_std
                pred_norm = student(img_in, st_in)
                pred_chunk = (pred_norm * act_std + act_mean).squeeze(0).cpu().numpy()
        obs, reward, _, _, _ = env.step(pred_chunk[t % 10])
        untrained_frames.append(label_frame(obs["pixels"]["top"], "BEFORE Training (Untrained Net)", t + 1, float(reward), (255, 100, 100)))

    print("\n" + "=" * 72)
    print(f"[Step 2] RTX 4090에서 학생 모방학습 모델(ActionChunkingNet) 학습 시작!")
    print(f"         입력: 128x128 이미지 + 14축 관절 -> 출력: 미래 {chunk_size}스텝 x 14축 행동 청크")
    print("=" * 72)

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 120
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.L1Loss()

    batch_size = 64
    num_samples = imgs_t.shape[0]
    t0 = time.perf_counter()

    student.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_samples, device=device)
        epoch_loss = 0.0
        steps = 0
        for i in range(0, num_samples, batch_size):
            idx = perm[i : i + batch_size]
            # 약간의 상태 노이즈(Noise Injection)를 추가해 Covariate Shift 저항성 향상!
            noisy_state = states_norm[idx] + 0.02 * torch.randn_like(states_norm[idx])
            pred_chunk = student(imgs_t[idx], noisy_state)
            loss = loss_fn(pred_chunk, chunks_norm[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{epochs} | L1 Action Chunk Loss: {epoch_loss / steps:.4f} | 경과 시간: {time.perf_counter() - t0:.1f}s")

    print("\n" + "=" * 72)
    print("[Step 3] 직접 학습한 학생 모델 + Temporal Ensembling으로 로봇 테스트 실행 중...")
    print("=" * 72)

    student.eval()
    obs, _ = env.reset(seed=1)
    comparison_frames = []
    max_reward = 0.0
    success_step = None

    # ACT 논문의 Temporal Ensembling 버퍼: all_time_actions[t, t : t+chunk_size]
    all_time_actions = np.zeros((steps_per_ep, steps_per_ep + chunk_size, 14), dtype=np.float32)

    for t in range(steps_per_ep):
        with torch.inference_mode():
            img_in = preprocess_img_128(obs["pixels"]["top"], device)
            st_in = (torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device) - state_mean) / state_std
            pred_norm = student(img_in, st_in)
            pred_chunk = (pred_norm * act_std + act_mean).squeeze(0).cpu().numpy()

        all_time_actions[t, t : t + chunk_size] = pred_chunk

        # 현재 시점 t에 대해 과거(max(0, t-chunk_size+1) .. t)에 예측된 모든 행동들을 지수 가중 평균!
        start_idx = max(0, t - chunk_size + 1)
        actions_for_curr_step = all_time_actions[start_idx : t + 1, t]
        exp_weights = np.exp(-0.01 * np.arange(len(actions_for_curr_step))[::-1])
        exp_weights = exp_weights / exp_weights.sum()
        action = (actions_for_curr_step * exp_weights[:, None]).sum(axis=0)

        obs, reward, _, _, _ = env.step(action)
        max_reward = max(max_reward, float(reward))
        if float(reward) >= 4.0 and success_step is None:
            success_step = t + 1
            print(f"  ★ Step {success_step:03d}: 직접 학습한 학생 모델이 양팔 큐브 공중 전달(Handover) 성공! (Reward = {reward})")

        tr_frame = label_frame(obs["pixels"]["top"], "AFTER RTX 4090 Training + Temporal Ensemble", t + 1, float(reward), (80, 230, 100))
        comparison_frames.append(np.concatenate([untrained_frames[t], tr_frame], axis=1))

    env.close()

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    comp_video = out_dir / "before_vs_after_training.mp4"
    imageio.mimsave(str(comp_video), comparison_frames, fps=50)

    print("=" * 72)
    print(f"학생 모델 테스트 완료! 최고 보상(Max Reward): {max_reward:.1f} / 4.0")
    print(f"학습 전후 비교 영상: {comp_video}")
    print("=" * 72)


if __name__ == "__main__":
    main()
