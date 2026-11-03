"""
Demo 6: VLA (Vision-Language-Action) Multimodal Policy in MuJoCo (RTX 4090)
===========================================================================
자율주행 VLA(Waymo EMMA, DriveVLM) 및 로봇 VLA(Pi-0, OpenVLA)의 아키텍처 핵심인
[Vision Tokens + Language Instruction Tokens + Ego-State -> Cross-Attention -> Action Chunk Expert]
구조를 RTX 4090에서 직접 학습하고 3가지 언어 명령으로 제어해 보는 실습입니다.

[동일한 단일 VLA 신경망에 입력하는 3가지 자연어 명령(Prompt)]
  Command 0: "grasp and hold red cube with right arm only"
             -> 오른팔로 큐브를 집어 들어 올린 뒤 왼팔에 넘기지 않고 오른팔로만 계속 들고 대기 (Reward = 2.0)
  Command 1: "transfer red cube from right arm to left arm"
             -> 오른팔로 큐브를 집어 공중에서 왼팔에 끝까지 인계 (Reward = 4.0)
  Command 2: "open grippers and spread both arms to safe home"
             -> 큐브를 건드리지 않고 양팔 그리퍼를 연 채 좌우 안전 홈 자세로 넓게 벌림 (안전 정지 모드)
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

COMMANDS = [
    "grasp and hold red cube with right arm only",
    "transfer red cube from right arm to left arm",
    "open grippers and spread both arms to safe home",
]

VOCAB = {
    "<pad>": 0, "grasp": 1, "and": 2, "hold": 3, "red": 4, "cube": 5,
    "with": 6, "right": 7, "arm": 8, "only": 9, "transfer": 10, "from": 11,
    "to": 12, "left": 13, "open": 14, "grippers": 15, "spread": 16,
    "both": 17, "arms": 18, "safe": 19, "home": 20,
}


def tokenize_prompt(prompt: str, max_len: int = 10, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    words = prompt.lower().split()[:max_len]
    ids = [VOCAB.get(w, 0) for w in words]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long, device=device)


class MiniVLAPolicy(nn.Module):
    """
    Vision-Language-Action (VLA) 아키텍처:
    1. Vision Tokenizer : 128x128 이미지를 4x4 = 16개의 공간 비전 토큰(Spatial Visual Tokens)으로 변환
    2. Language Encoder : 자연어 명령어 단어들을 임베딩 + Transformer Encoder로 문맥화
    3. Multimodal Fusion: 언어+상태 쿼리 토큰들이 비전 토큰들과 Cross-Attention 수행
    4. Action Chunk Head: 융합된 표현에서 미래 50스텝 x 14축 행동 청크(Action Chunk) 출력
    """
    def __init__(self, vocab_size: int = 32, d_model: int = 256, state_dim: int = 14, action_dim: int = 14, chunk_size: int = 50):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.d_model = d_model

        # 1. Vision Backbone -> 4x4 = 16 Visual Patch Tokens of dim 256
        self.vision_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.vis_pos_emb = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)

        # 2. Language Token Embedding
        self.lang_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lang_pos_emb = nn.Parameter(torch.randn(1, 10, d_model) * 0.02)

        # 3. Proprioception (Robot Joint / Vehicle Ego State) Projector
        self.state_proj = nn.Linear(state_dim, d_model)

        # 4. Multimodal Transformer (Self + Cross Attention across Language, Vision, State)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=512, dropout=0.05, activation="gelu", batch_first=True
        )
        self.multimodal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 5. Action Chunk Expert Head (Multimodal State + Cross-Attended Lang + Direct Lang Residual)
        self.action_expert = nn.Sequential(
            nn.Linear(d_model * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, chunk_size * action_dim),
        )

    def forward(self, img_128: torch.Tensor, token_ids: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b = img_128.shape[0]
        # 비전 토큰: (B, 256, 4, 4) -> (B, 16, 256)
        vis_feat = self.vision_backbone(img_128).flatten(2).transpose(1, 2) + self.vis_pos_emb
        # 언어 토큰: (B, 10, 256)
        lang_raw = self.lang_emb(token_ids)
        lang_feat = lang_raw + self.lang_pos_emb
        # 상태 토큰: (B, 1, 256)
        state_feat = self.state_proj(state).unsqueeze(1)

        # [State_Token(1), Lang_Tokens(10), Vision_Tokens(16)] 총 27개 멀티모달 토큰 시퀀스 결합
        seq = torch.cat([state_feat, lang_feat, vis_feat], dim=1)
        fused = self.multimodal_transformer(seq)

        # 상태 토큰 출력 + 어텐션 거친 언어 표현 + 직접 언어 임베딩 잔차 연결(Residual) 결합
        pooled = torch.cat([fused[:, 0], fused[:, 1:11].mean(dim=1), lang_raw.mean(dim=1)], dim=-1)
        out = self.action_expert(pooled)
        return out.view(b, self.chunk_size, self.action_dim)


def preprocess_img_128(rgb_numpy: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(rgb_numpy).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    return torch.nn.functional.interpolate(t, size=(128, 128), mode="bilinear", align_corners=False)


def annotate_vla_frame(frame: np.ndarray, cmd_idx: int, prompt: str, step: int, reward: float) -> np.ndarray:
    canvas = frame.copy()
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 80), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

    colors = [(56, 189, 248), (74, 222, 128), (251, 191, 36)]
    c = colors[cmd_idx]

    cv2.putText(canvas, f"VLA Prompt #{cmd_idx + 1}: \"{prompt}\"", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, c, 2, cv2.LINE_AA)
    mode_labels = [
        f"Mode: Right-Arm Grasp & Hold | Step {step:03d} | Reward: {reward:.1f}",
        f"Mode: Full Bimanual Handover | Step {step:03d} | Reward: {reward:.1f}",
        f"Mode: Safe Home Spread Posture | Step {step:03d} | Reward: {reward:.1f}",
    ]
    cv2.putText(canvas, mode_labels[cmd_idx], (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 2, cv2.LINE_AA)
    return canvas


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_size = 50
    steps_per_ep = 380

    print("=" * 76)
    print("[1/3] 3가지 언어 명령(Prompt)에 대응하는 멀티모달 데모 수집 중...")
    print("=" * 76)

    teacher_id = "lerobot/act_aloha_sim_transfer_cube_human"
    teacher = ACTPolicy.from_pretrained(teacher_id)
    teacher.config.n_action_steps = 20
    teacher.to(device).eval()
    teacher_stats = load_norm_stats(teacher_id, device)

    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", max_episode_steps=steps_per_ep)

    demo_imgs, demo_tokens, demo_states, demo_chunks = [], [], [], []

    # 먼저 교사 모델의 성공 궤적(380 스텝)을 1회 추출
    obs, _ = env.reset(seed=1)
    teacher.reset()
    base_actions = []
    for _ in range(steps_per_ep):
        batch = format_and_normalize_obs(obs, teacher_stats, device)
        with torch.inference_mode():
            act_norm = teacher.select_action(batch)
            act = (act_norm * teacher_stats["action_std"] + teacher_stats["action_mean"]).squeeze(0).cpu().numpy()
        base_actions.append(act.copy())
        obs, _, _, _, _ = env.step(act)

    # 3가지 언어 명령 각각에 대해 시연 데이터 생성
    for cmd_idx, prompt in enumerate(COMMANDS):
        tok = tokenize_prompt(prompt, device=torch.device("cpu"))
        obs, _ = env.reset(seed=1)
        ep_imgs, ep_states, ep_actions = [], [], []
        home_pose = np.array([0.0, -0.95, 1.15, 0.0, -0.3, 0.0, 0.04, 0.0, -0.95, 1.15, 0.0, -0.3, 0.0, 0.04], dtype=np.float32)

        for t in range(steps_per_ep):
            if cmd_idx == 0:
                # 명령 0: 오른팔로 큐브를 공중에 번쩍 들어 올린 시점(t=205, Reward=2.0)에서 전달하지 않고 계속 공중 홀드!
                act = base_actions[min(t, 205)].copy()
                if t > 140:
                    act[:7] = base_actions[140][:7]  # 왼팔은 다가오지 않고 대기 위치 유지
            elif cmd_idx == 1:
                # 명령 1: 오른팔에서 왼팔로 공중 전달 완수!
                act = base_actions[t].copy()
            else:
                # 명령 2: 양팔 모두 그리퍼를 열고 안전 홈 자세로 부드럽게 이동
                curr_q = obs["agent_pos"].astype(np.float32)
                alpha = min(1.0, (t + 1) / 60.0)
                act = (1.0 - alpha) * curr_q + alpha * home_pose

            ep_imgs.append(preprocess_img_128(obs["pixels"]["top"], device).squeeze(0).cpu())
            ep_states.append(torch.from_numpy(obs["agent_pos"]).float())
            ep_actions.append(torch.from_numpy(act).float())
            obs, reward, _, _, _ = env.step(act)

        ep_actions_padded = ep_actions + [ep_actions[-1]] * chunk_size
        ep_actions_t = torch.stack(ep_actions_padded)
        for t in range(steps_per_ep):
            demo_imgs.append(ep_imgs[t])
            demo_tokens.append(tok)
            demo_states.append(ep_states[t])
            demo_chunks.append(ep_actions_t[t : t + chunk_size])

        print(f"  - Prompt #{cmd_idx + 1} 데이터 수집 완료: \"{prompt}\"")

    imgs_t = torch.stack(demo_imgs).to(device)
    tokens_t = torch.stack(demo_tokens).to(device)
    states_t = torch.stack(demo_states).to(device)
    chunks_t = torch.stack(demo_chunks).to(device)

    state_mean = states_t.mean(dim=0, keepdim=True)
    state_std = states_t.std(dim=0, keepdim=True) + 1e-5
    act_mean = chunks_t.mean(dim=(0, 1), keepdim=True)
    act_std = chunks_t.std(dim=(0, 1), keepdim=True) + 1e-5

    states_norm = (states_t - state_mean) / state_std
    chunks_norm = (chunks_t - act_mean) / act_std

    print("\n" + "=" * 76)
    print("[2/3] RTX 4090에서 멀티모달 VLA (Vision-Language-Action) 트랜스포머 학습 시작!")
    print("=" * 76)

    vla = MiniVLAPolicy(vocab_size=len(VOCAB) + 2, d_model=256, state_dim=14, action_dim=14, chunk_size=chunk_size).to(device)
    optimizer = torch.optim.AdamW(vla.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 110
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.L1Loss()

    batch_size = 64
    num_samples = imgs_t.shape[0]
    t0 = time.perf_counter()

    vla.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_samples, device=device)
        epoch_loss = 0.0
        steps = 0
        for i in range(0, num_samples, batch_size):
            idx = perm[i : i + batch_size]
            noisy_st = states_norm[idx] + 0.02 * torch.randn_like(states_norm[idx])
            pred = vla(imgs_t[idx], tokens_t[idx], noisy_st)
            loss = loss_fn(pred, chunks_norm[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{epochs} | VLA Multimodal L1 Loss: {epoch_loss / steps:.4f} | 경과 시간: {time.perf_counter() - t0:.1f}s")

    print("\n" + "=" * 76)
    print("[3/3] 단일 VLA 모델에 3가지 다른 언어 프롬프트를 입력해 MuJoCo 실행 비교 중...")
    print("=" * 76)

    vla.eval()
    panel_videos = []

    for cmd_idx, prompt in enumerate(COMMANDS):
        tok_in = tokenize_prompt(prompt, device=device).unsqueeze(0)
        obs, _ = env.reset(seed=1)
        frames = []
        max_r = 0.0
        all_time_actions = np.zeros((steps_per_ep, steps_per_ep + chunk_size, 14), dtype=np.float32)

        for t in range(steps_per_ep):
            with torch.inference_mode():
                img_in = preprocess_img_128(obs["pixels"]["top"], device)
                st_in = (torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device) - state_mean) / state_std
                pred_norm = vla(img_in, tok_in, st_in)
                pred_chunk = (pred_norm * act_std + act_mean).squeeze(0).cpu().numpy()

            all_time_actions[t, t : t + chunk_size] = pred_chunk
            start_idx = max(0, t - chunk_size + 1)
            actions_curr = all_time_actions[start_idx : t + 1, t]
            w = np.exp(-0.01 * np.arange(len(actions_curr))[::-1])
            w /= w.sum()
            action = (actions_curr * w[:, None]).sum(axis=0)

            obs, reward, _, _, _ = env.step(action)
            max_r = max(max_r, float(reward))
            frames.append(annotate_vla_frame(obs["pixels"]["top"], cmd_idx, prompt, t + 1, float(reward)))

        panel_videos.append(frames)
        print(f"  ★ Prompt #{cmd_idx + 1} 실행 완료 (최고 보상 = {max_r:.1f}): \"{prompt}\"")

    env.close()

    # 3개의 언어 명령 실행 영상을 가로로 나란히 결합 (3-Panel VLA Comparison)
    combined_frames = [
        np.concatenate([panel_videos[0][t], panel_videos[1][t], panel_videos[2][t]], axis=1)
        for t in range(steps_per_ep)
    ]

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    vla_mp4 = out_dir / "vla_language_commands.mp4"
    vla_snap = out_dir / "vla_3commands_snapshot.png"
    imageio.mimsave(str(vla_mp4), combined_frames, fps=50)
    imageio.imwrite(str(vla_snap), combined_frames[345])

    print("=" * 76)
    print(f"3-Panel VLA 언어 제어 비교 영상 저장 완료: {vla_mp4}")
    print(f"3-Panel VLA 스냅샷 저장 완료            : {vla_snap}")
    print("=" * 76)


if __name__ == "__main__":
    main()
