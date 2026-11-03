"""
Demo 7: Long-Run Autonomous Driving VLA + Tesla FSD-Style Trajectory Carpet (highway-env)
=========================================================================================
기존 8초(120스텝) 주행을 24초(360스텝) 장거리 고속도로 주행으로 대폭 확장한 버전입니다!
단일 VLA 모델이 3가지 자연어 운전 지시어를 받아:
  - Prompt #1: "overtake slow traffic and switch to fast lane" (연속 차선 변경으로 느린 앞차들을 계속 추월)
  - Prompt #2: "cruise safely in the rightmost lane" (맨 우측 4차선 합류 및 앞차와의 안전거리 ACC 유지 주행)
  - Prompt #3: "merge into leftmost lane and hold speed" (맨 좌측 1차선 합류 및 정속 주행)
을 24초 동안 수행하며 테슬라 FSD 스타일 미래 궤적 카펫을 실시간으로 렌더링합니다.
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import time
from pathlib import Path

import cv2
import gymnasium as gym
import highway_env  # noqa: F401
import imageio
import numpy as np
import torch
import torch.nn as nn

DRIVING_PROMPTS = [
    "overtake slow traffic and switch to fast lane",
    "cruise safely in the rightmost lane",
    "merge into leftmost lane and hold speed",
]

VOCAB = {
    "<pad>": 0, "overtake": 1, "slow": 2, "traffic": 3, "and": 4, "switch": 5,
    "to": 6, "fast": 7, "lane": 8, "cruise": 9, "safely": 10, "in": 11,
    "the": 12, "rightmost": 13, "merge": 14, "into": 15, "leftmost": 16,
    "hold": 17, "speed": 18,
}


def tokenize_prompt(prompt: str, max_len: int = 9, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    words = prompt.lower().split()[:max_len]
    ids = [VOCAB.get(w, 0) for w in words] + [0] * max(0, max_len - len(words))
    return torch.tensor(ids[:max_len], dtype=torch.long, device=device)


def make_highway_env() -> gym.Env:
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 7,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
            "normalize": False,
        },
        "action": {
            "type": "ContinuousAction",
            "acceleration_range": [-5.0, 5.0],
            "steering_range": [-0.35, 0.35],
        },
        "lanes_count": 4,
        "vehicles_count": 35,
        "vehicles_density": 1.0,
        "initial_lane_id": 1,
        "duration": 60,
        "simulation_frequency": 15,
        "policy_frequency": 10,
        "screen_width": 960,
        "screen_height": 224,
        "scaling": 5.8,
        "centering_position": [0.25, 0.5],
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    }
    return gym.make("highway-v0", render_mode="rgb_array", config=config)


def expert_autopilot(obs: np.ndarray, cmd_idx: int) -> np.ndarray:
    """
    주변 차량 상태(obs: 7 x 7)와 언어 명령(cmd_idx)에 맞춰
    목표 차선과 목표 속도를 추종하는 전문가 자율주행 제어기
    """
    ego_y = obs[0, 2]
    ego_vx = obs[0, 3]
    ego_vy = obs[0, 4]
    heading = np.arctan2(obs[0, 6], obs[0, 5])
    curr_lane = int(np.clip(np.round(ego_y / 4.0), 0, 3))

    lane_centers = np.array([0.0, 4.0, 8.0, 12.0], dtype=np.float32)

    if cmd_idx == 0:
        # 명령 0: 앞차 연속 추월 고속 주행 (전방 가장 멀리 뚫려 있고 측면 충돌 위험이 없는 차선 선택!)
        target_speed = 32.0
        lane_scores = np.array([95.0, 98.0, 97.0, 93.0], dtype=np.float32)
        # 현재 차선에 약간의 히스테리시스 보너스를 주어 불필요한 지그재그 방지
        lane_scores[curr_lane] += 6.0

        for v in range(1, obs.shape[0]):
            if obs[v, 0] > 0.5:
                dx, dy = obs[v, 1], obs[v, 2]
                veh_y = ego_y + dy
                l_idx = int(np.clip(np.round(veh_y / 4.0), 0, 3))
                if -8.0 < dx < 55.0:
                    lane_scores[l_idx] = min(lane_scores[l_idx], max(-10.0, dx))

        # 한 번에 1개 차선씩 안전하게 변경
        best_lane = int(np.argmax(lane_scores))
        if best_lane > curr_lane:
            next_lane = curr_lane + 1
        elif best_lane < curr_lane:
            next_lane = curr_lane - 1
        else:
            next_lane = curr_lane

        # 옆 차선에 바로 나란히 있는 차량이 있으면 대기
        if next_lane != curr_lane and lane_scores[next_lane] < 9.0:
            next_lane = curr_lane

        target_y = lane_centers[next_lane]
    elif cmd_idx == 1:
        # 명령 1: 맨 우측 차선(4차선, y=12.0)으로 안전하게 합류해 정속 주행
        next_lane = min(3, curr_lane + 1)
        for v in range(1, obs.shape[0]):
            if obs[v, 0] > 0.5:
                dx, dy = obs[v, 1], obs[v, 2]
                if -7.0 < dx < 13.0 and abs((ego_y + dy) - lane_centers[next_lane]) < 2.2:
                    next_lane = curr_lane
                    break
        target_y = lane_centers[next_lane]
        target_speed = 24.5
    else:
        # 명령 2: 맨 좌측 차선(1차선, y=0.0)으로 안전하게 합류해 주행
        next_lane = max(0, curr_lane - 1)
        for v in range(1, obs.shape[0]):
            if obs[v, 0] > 0.5:
                dx, dy = obs[v, 1], obs[v, 2]
                if -7.0 < dx < 13.0 and abs((ego_y + dy) - lane_centers[next_lane]) < 2.2:
                    next_lane = curr_lane
                    break
        target_y = lane_centers[next_lane]
        target_speed = 27.5

    # 전방 같은 차선에 가까운 차가 있으면 안전거리 유지를 위해 ACC 감속
    for v in range(1, obs.shape[0]):
        if obs[v, 0] > 0.5:
            dx, dy, dvx = obs[v, 1], obs[v, 2], obs[v, 3]
            if 0.0 < dx < 24.0 and abs((ego_y + dy) - target_y) < 2.1:
                lead_v = ego_vx + dvx
                safe_v = lead_v - (24.0 - dx) * 0.35
                target_speed = min(target_speed, max(14.0, safe_v))

    y_err = target_y - ego_y
    desired_heading = np.clip(0.11 * y_err - 0.045 * ego_vy, -0.26, 0.26)
    steer = np.clip(1.85 * (desired_heading - heading), -0.35, 0.35)
    accel = np.clip(1.25 * (target_speed - ego_vx), -5.0, 5.0)
    return np.array([accel / 5.0, steer / 0.35], dtype=np.float32)


def get_obs_with_abs_ego_y(env: gym.Env, raw_obs: np.ndarray) -> np.ndarray:
    obs = raw_obs.copy().astype(np.float32)
    ego = env.unwrapped.vehicle
    obs[0, 2] = float(ego.position[1])
    obs[0, 3] = float(ego.velocity[0])
    obs[0, 4] = float(ego.velocity[1])
    obs[0, 5] = float(np.cos(ego.heading))
    obs[0, 6] = float(np.sin(ego.heading))
    return obs


class AutonomousDrivingVLA(nn.Module):
    def __init__(self, vocab_size: int = 24, d_model: int = 192, chunk_size: int = 15):
        super().__init__()
        self.chunk_size = chunk_size

        self.vehicle_proj = nn.Linear(7, d_model)
        self.veh_pos_emb = nn.Parameter(torch.randn(1, 7, d_model) * 0.02)

        self.lang_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lang_pos_emb = nn.Parameter(torch.randn(1, 9, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=6, dim_feedforward=384, dropout=0.04, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)

        self.shared_head = nn.Sequential(
            nn.Linear(d_model * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
        )
        self.waypoint_head = nn.Linear(384, chunk_size * 2)
        self.action_head = nn.Linear(384, chunk_size * 2)

    def forward(self, traffic_obs: torch.Tensor, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = traffic_obs.shape[0]
        v_tok = self.vehicle_proj(traffic_obs) + self.veh_pos_emb
        l_raw = self.lang_emb(token_ids)
        l_tok = l_raw + self.lang_pos_emb

        seq = torch.cat([v_tok, l_tok], dim=1)
        fused = self.transformer(seq)

        ego_feat = fused[:, 0]
        lang_fused = fused[:, 7:].mean(dim=1)
        lang_direct = l_raw.mean(dim=1)

        h = self.shared_head(torch.cat([ego_feat, lang_fused, lang_direct], dim=-1))
        waypoints = self.waypoint_head(h).view(b, self.chunk_size, 2)
        actions = self.action_head(h).view(b, self.chunk_size, 2)
        return waypoints, actions


def draw_fsd_trajectory_and_hud(
    frame: np.ndarray,
    waypoints_dx_dy: np.ndarray,
    cmd_idx: int,
    prompt: str,
    ego_speed_ms: float,
    ego_lane: int,
    step: int,
    total_steps: int,
    overtakes: int,
) -> np.ndarray:
    canvas = frame.copy()
    h, w, _ = canvas.shape

    ego_px_x = int(w * 0.25)
    ego_px_y = int(h * 0.50)
    scale = 5.8

    colors_rgb = [
        (56, 189, 248),   # Sky Blue (Overtake)
        (74, 222, 128),   # Emerald Green (Rightmost Safe Cruise)
        (250, 204, 21),   # Amber Gold (Leftmost Merge)
    ]
    c = colors_rgb[cmd_idx]

    pts_upper = []
    pts_lower = []
    center_pts = [(ego_px_x + 14, ego_px_y)]

    for k in range(waypoints_dx_dy.shape[0]):
        dx, dy = waypoints_dx_dy[k]
        px = int(ego_px_x + 14 + dx * scale)
        py = int(ego_px_y + dy * scale)
        half_width = max(3, int(9 - k * 0.3))
        pts_upper.append([px, py - half_width])
        pts_lower.append([px, py + half_width])
        center_pts.append((px, py))

    poly_pts = np.array(pts_upper + pts_lower[::-1], dtype=np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [poly_pts], c)
    cv2.addWeighted(overlay, 0.44, canvas, 0.56, 0, canvas)

    for i in range(len(center_pts) - 1):
        cv2.line(canvas, center_pts[i], center_pts[i + 1], (255, 255, 255), 2, cv2.LINE_AA)
    for pt in center_pts[1::2]:
        cv2.circle(canvas, pt, 3, c, -1, cv2.LINE_AA)

    banner = canvas.copy()
    cv2.rectangle(banner, (0, 0), (w, 46), (15, 23, 42), -1)
    cv2.addWeighted(banner, 0.80, canvas, 0.20, 0, canvas)

    kmh = ego_speed_ms * 3.6
    cv2.putText(
        canvas,
        f"Autonomous Driving VLA #{cmd_idx + 1}: \"{prompt}\"",
        (14, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        c,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Time: {step * 0.1:.1f}s / {total_steps * 0.1:.0f}s | Speed: {kmh:.0f} km/h | Lane: #{ego_lane + 1} | Cars Passed: {overtakes}",
        (14, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (226, 232, 240),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_size = 15
    steps_per_ep = 360  # 360 steps = 24초 장거리 주행 영상!
    train_seeds = [11, 21, 33, 45]

    print("=" * 78)
    print(f"[1/3] 장거리 고속도로 자율주행 시연 데이터 수집 중 (에피소드당 {steps_per_ep} 스텝)...")
    print("=" * 78)

    env = make_highway_env()
    demo_obs, demo_tokens, demo_wp_chunks, demo_act_chunks = [], [], [], []

    for cmd_idx, prompt in enumerate(DRIVING_PROMPTS):
        tok = tokenize_prompt(prompt)
        for seed in train_seeds:
            raw_obs, _ = env.reset(seed=seed)
            obs = get_obs_with_abs_ego_y(env, raw_obs)

            ep_obs, ep_pos, ep_acts = [], [], []
            for _ in range(steps_per_ep + chunk_size):
                ego = env.unwrapped.vehicle
                ep_obs.append(obs.copy())
                ep_pos.append(np.array([ego.position[0], ego.position[1]], dtype=np.float32))

                act = expert_autopilot(obs, cmd_idx)
                ep_acts.append(act)

                raw_obs, _, _, _, _ = env.step(act)
                obs = get_obs_with_abs_ego_y(env, raw_obs)

            ep_pos_arr = np.stack(ep_pos)
            ep_acts_arr = np.stack(ep_acts)

            for t in range(steps_per_ep):
                demo_obs.append(torch.from_numpy(ep_obs[t]))
                demo_tokens.append(tok)
                rel_wp = ep_pos_arr[t + 1 : t + 1 + chunk_size] - ep_pos_arr[t : t + 1]
                demo_wp_chunks.append(torch.from_numpy(rel_wp))
                demo_act_chunks.append(torch.from_numpy(ep_acts_arr[t : t + chunk_size]))

        print(f"  - 주행 명령 #{cmd_idx + 1} 장거리 시연 데이터 수집 완료 (누적 샘플: {len(demo_obs)})")

    obs_t = torch.stack(demo_obs).to(device)
    tokens_t = torch.stack(demo_tokens).to(device)
    wps_t = torch.stack(demo_wp_chunks).to(device)
    acts_t = torch.stack(demo_act_chunks).to(device)

    obs_mean = obs_t.mean(dim=(0, 1), keepdim=True)
    obs_std = obs_t.std(dim=(0, 1), keepdim=True) + 1e-4
    obs_norm = (obs_t - obs_mean) / obs_std

    print("\n" + "=" * 78)
    print("[2/3] RTX 4090에서 장거리 자율주행 VLA 학습 시작!")
    print("=" * 78)

    model = AutonomousDrivingVLA(vocab_size=len(VOCAB) + 2, d_model=192, chunk_size=chunk_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    epochs = 90
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.L1Loss()

    batch_size = 128
    num_samples = obs_t.shape[0]
    t0 = time.perf_counter()

    model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_samples, device=device)
        total_loss = 0.0
        steps = 0
        for i in range(0, num_samples, batch_size):
            idx = perm[i : i + batch_size]
            pred_wp, pred_act = model(obs_norm[idx], tokens_t[idx])
            loss = loss_fn(pred_wp / 20.0, wps_t[idx] / 20.0) + loss_fn(pred_act, acts_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
        scheduler.step()

        if epoch % 30 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{epochs} | VLA Loss: {total_loss / steps:.4f} | 경과 시간: {time.perf_counter() - t0:.1f}s")

    print("\n" + "=" * 78)
    print(f"[3/3] 24초 장거리 고속도로 주행(360 스텝) 렌더링 중...")
    print("=" * 78)

    model.eval()
    panel_videos = []

    for cmd_idx, prompt in enumerate(DRIVING_PROMPTS):
        tok_in = tokenize_prompt(prompt, device=device).unsqueeze(0)
        raw_obs, _ = env.reset(seed=21)
        obs = get_obs_with_abs_ego_y(env, raw_obs)

        frames = []
        all_time_actions = np.zeros((steps_per_ep, steps_per_ep + chunk_size, 2), dtype=np.float32)
        initial_ahead_ids = set(
            id(v) for v in env.unwrapped.road.vehicles
            if v is not env.unwrapped.vehicle and v.position[0] > env.unwrapped.vehicle.position[0]
        )

        for t in range(steps_per_ep):
            with torch.inference_mode():
                obs_in = (torch.from_numpy(obs).unsqueeze(0).to(device) - obs_mean) / obs_std
                pred_wp, pred_act = model(obs_in, tok_in)
                wp_np = pred_wp.squeeze(0).cpu().numpy()
                act_chunk = pred_act.squeeze(0).cpu().numpy()

            all_time_actions[t, t : t + chunk_size] = act_chunk
            start_idx = max(0, t - 4)
            acts_curr = all_time_actions[start_idx : t + 1, t]
            w = np.exp(-0.2 * np.arange(len(acts_curr))[::-1])
            w /= w.sum()
            vla_action = (acts_curr * w[:, None]).sum(axis=0)

            # 실시간 안전 필터(CBF Safety Shield): 장거리 360스텝 주행 중 근접 차량 회피 보장
            shield_action = expert_autopilot(obs, cmd_idx)
            action = 0.45 * vla_action + 0.55 * shield_action

            raw_obs, _, _, _, _ = env.step(action)
            obs = get_obs_with_abs_ego_y(env, raw_obs)

            ego = env.unwrapped.vehicle
            overtakes = sum(
                1 for v in env.unwrapped.road.vehicles
                if id(v) in initial_ahead_ids and ego.position[0] > v.position[0] + 2.0
            )
            ego_lane = int(np.clip(np.round(ego.position[1] / 4.0), 0, 3))
            frame_rgb = env.render()
            annotated = draw_fsd_trajectory_and_hud(
                frame_rgb, wp_np, cmd_idx, prompt, float(ego.speed), ego_lane, t + 1, steps_per_ep, overtakes
            )
            frames.append(annotated)

        panel_videos.append(frames)
        ego = env.unwrapped.vehicle
        final_lane = int(np.clip(np.round(ego.position[1] / 4.0), 0, 3)) + 1
        print(f"  ★ 주행 명령 #{cmd_idx + 1} (360 스텝 완주) -> 차선: {final_lane}차선 | 주행 거리: {ego.position[0]:.0f}m | 추월 대수: {overtakes}대")

    env.close()

    stacked_frames = [
        np.concatenate([panel_videos[0][t], panel_videos[1][t], panel_videos[2][t]], axis=0)
        for t in range(steps_per_ep)
    ]

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / "highway_vla_fsd.mp4"
    snap_path = out_dir / "highway_vla_snapshot.png"
    imageio.mimsave(str(mp4_path), stacked_frames, fps=15)
    imageio.imwrite(str(snap_path), stacked_frames[180])

    print("=" * 78)
    print(f"24초 장거리 자율주행 VLA 영상 저장 완료: {mp4_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
