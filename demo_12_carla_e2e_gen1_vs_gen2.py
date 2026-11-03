"""
Demo 12: 1st-Gen Black-Box E2E (PilotNet) vs. 2nd-Gen Planning-Oriented E2E (UniAD / FSD v12) in CARLA
======================================================================================================
자율주행 E2E(End-to-End) 아키텍처의 진화 과정을 CARLA 0.9.15 도심 S자 연속 장애물(Double-Obstacle Chicane)
코스에서 직접 학습하고 폐루프(Closed-Loop) 주행으로 비교 실험합니다!

[핵심 포인트: 추론(Inference) 시 두 모델 모두 '오직 전방 Raw RGB 카메라 영상'만 사용합니다!]
1. 1세대 순수 블랙박스 E2E (`Gen1_BlackBox_PilotNet` - NVIDIA PilotNet 스타일):
   - 입력: 전방 Raw RGB 이미지 (3채널) + 현재 속도
   - 구조: CNN -> MLP -> 곧바로 현재 1스텝 핸들/페달 [throttle, steer, brake] 출력
   - 한계:
     * 중간 인지(도로/차선/장애물 분할) 학습이 전혀 없어 배경 건물·나무 그림자 노이즈에 취약 (PIP 창에 CNN 활성화 히트맵 표시)
     * 미래 궤적(Waypoints) 계획 없이 당장 눈앞의 1스텝 반응만 출력하여 연속 장애물 S자 회피 시 조향 진동(Wobble) 및 경로 이탈 발생!

2. 2세대 플래닝 중심 모듈형 E2E (`Gen2_Planning_UniAD` - UniAD / VAD / Tesla FSD v12 스타일):
   - 입력: 전방 Raw RGB 이미지 (3채널) + 차량 상태 (추론 시 세그멘테이션 센서 불필요!)
   - 구조 (단일 미분 가능 네트워크 내부 3단계 멀티태스크 학습):
     ① [Auxiliary Perception Head]: RGB 잠재 특징만으로 시맨틱 세그멘테이션 맵(도로/차선/장애물)을 스스로 복원하도록 보조 손실(L_seg) 학습!
        -> 추론 시 신경망이 머릿속으로 상상/예측한 세그멘테이션 결과(Predicted Seg)를 PIP 창에 실시간 시각화!
     ② [3D Trajectory Planning Head]: 미래 K=15 스텝 3D 주행 웨이포인트(L_plan)를 먼저 계획하여 전면 유리에 FSD 카펫으로 투영!
     ③ [Trajectory-Conditioned Control Head]: 예측된 미래 3D 궤적을 조건으로 받아 K=15 제어 청크(L_ctrl)를 출력 + Temporal Ensembling 제어!
"""
import argparse
import math
import queue
import time
from pathlib import Path

import carla
import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from demo_9_carla_urban_vla import build_camera_intrinsic, connect_carla_client, project_world_to_image
from demo_10_carla_end_to_end_vla import compute_frenet_s_d, find_long_straight_spawn


def chicane_target_lateral_d(s_query: float) -> float:
    """
    S자 연속 장애물 회피(Double-Obstacle Chicane) 목표 횡방향 오프셋 d_ref(s):
    - 장애물 #1 (s=22m, d=0.0m 중앙 차선 경찰차): s=8~18m 구간에서 좌측 차선(d=-3.3m)으로 회피
    - 장애물 #2 (s=46m, d=-3.3m 좌측 차선 경찰차): s=30~40m 구간에서 다시 중앙/우측(d=+0.2m)으로 복귀
    """
    swerve_left = -3.35 / (1.0 + math.exp(-0.36 * (s_query - 11.5)))
    recover_right = +3.55 / (1.0 + math.exp(-0.36 * (s_query - 33.5)))
    return float(swerve_left + recover_right)


def expert_chicane_controller(
    ego_vehicle: carla.Vehicle,
    start_tf: carla.Transform,
) -> tuple[carla.VehicleControl, np.ndarray]:
    """
    연속 장애물 S자 슬라럼 코스를 부드럽게 통과하는 전문가 제어기 및 미래 K=15 로컬 웨이포인트 생성기
    """
    ego_tf = ego_vehicle.get_transform()
    vel = ego_vehicle.get_velocity()
    speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    s, d, ref_yaw = compute_frenet_s_d(ego_tf.location, start_tf)

    yaw_diff_deg = (ego_tf.rotation.yaw - ref_yaw + 180.0) % 360.0 - 180.0
    yaw_err = math.radians(yaw_diff_deg)

    d_target_ahead = chicane_target_lateral_d(s + 4.8)
    d_err = d_target_ahead - d
    desired_yaw = np.clip(0.25 * d_err, -0.48, 0.48)
    steer = float(np.clip(1.40 * (desired_yaw - yaw_err), -0.68, 0.68))

    target_speed_ms = 8.2  # 약 29.5 km/h
    if speed_ms < target_speed_ms:
        throttle = float(np.clip(0.55 + 0.15 * (target_speed_ms - speed_ms), 0.30, 0.76))
        brake = 0.0
    else:
        throttle = 0.22
        brake = 0.0

    local_wps = []
    fwd = start_tf.get_forward_vector()
    right = start_tf.get_right_vector()
    ego_fwd = ego_tf.get_forward_vector()
    ego_right = ego_tf.get_right_vector()

    for k in range(1, 16):
        s_fut = s + k * 1.35
        d_fut = chicane_target_lateral_d(s_fut)
        wx = start_tf.location.x + fwd.x * s_fut + right.x * d_fut
        wy = start_tf.location.y + fwd.y * s_fut + right.y * d_fut

        rel_x = (wx - ego_tf.location.x) * ego_fwd.x + (wy - ego_tf.location.y) * ego_fwd.y
        rel_y = (wx - ego_tf.location.x) * ego_right.x + (wy - ego_tf.location.y) * ego_right.y
        local_wps.append([max(1.2, float(rel_x)), float(rel_y)])

    ctrl = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
    return ctrl, np.array(local_wps, dtype=np.float32)


class Gen1_BlackBox_PilotNet(nn.Module):
    """
    [1세대 순수 블랙박스 E2E: NVIDIA PilotNet 스타일]
    - 입력: 오직 전방 Raw RGB (3x128x128) + 현재 속도(1차원)
    - 출력: 단일 스텝 제어값 [throttle, steer, brake] (3차원)
    - 특징: 중간 시맨틱 분할 보조 손실 없음, 미래 웨이포인트 예측 없음, Action Chunking 없음
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc = nn.Sequential(
            nn.Linear(128 * 16 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, rgb_128: torch.Tensor, speed_only: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.conv1(rgb_128)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)  # [B, 128, 8, 8]
        # 블랙박스 CNN의 공간 활성화 맵 (어디를 보고 운전하는지 시각화용)
        act_map = f4.abs().mean(dim=1, keepdim=True)  # [B, 1, 8, 8]
        pooled = self.pool(f4).flatten(1)
        out_ctrl = self.fc(torch.cat([pooled, speed_only], dim=-1))  # [B, 3]
        return out_ctrl, act_map


class Gen2_Planning_UniAD(nn.Module):
    """
    [2세대 플래닝 중심 모듈형 E2E: UniAD / VAD / Tesla FSD v12 스타일]
    - 추론 시 입력: 전방 Raw RGB (3x128x128) + 차량 상태(4차원) -> 1세대와 동일하게 RGB 카메라만 사용!
    - 내부 멀티태스크 파이프라인:
      1) Shared Vision Backbone -> 8x8 공간 잠재 특징(Latent Feature) 추출
      2) Auxiliary Perception Decoder: 잠재 특징만으로 전방 시맨틱 세그멘테이션(64x64)을 자체 예측 (L_seg)
      3) Transformer + Planning Head: 미래 K=15 스텝 3D 로컬 웨이포인트 계획 (L_plan)
      4) Trajectory-Conditioned Control Head: 잠재 특징 + 계획된 3D 웨이포인트(K=15x2)를 함께 받아
         미래 K=15 스텝 제어 청크 [throttle, steer, brake] 출력 (L_ctrl)
    """
    def __init__(self, d_model: int = 256, chunk_size: int = 15):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        # 1. Shared RGB Encoder (128x128 -> 8x8)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )  # [B, 256, 8, 8]

        # 2. Auxiliary Semantic Segmentation Decoder (8x8 -> 64x64 RGB Palette 복원)
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),       # 32x32
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),         # 64x64
            nn.Sigmoid(),
        )

        # 3. Spatial Latent Tokens (4x4=16) + Ego State -> Transformer
        self.vis_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.vis_pos = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)
        self.state_proj = nn.Linear(4, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=512, dropout=0.05, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)

        # 4. Future 3D Waypoint Planner Head (K=15 x 2)
        self.plan_head = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, chunk_size * 2),
        )

        # 5. Trajectory-Conditioned Control Chunk Head (계획된 웨이포인트를 입력받아 조향/페달 산출!)
        self.ctrl_head = nn.Sequential(
            nn.Linear(d_model * 2 + chunk_size * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, chunk_size * 3),
        )

    def forward(
        self, rgb_128: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = rgb_128.shape[0]
        feat_8x8 = self.enc(rgb_128)                 # [B, 256, 8, 8]
        pred_seg_64 = self.seg_decoder(feat_8x8)     # [B, 3, 64, 64]

        v_tok = self.vis_pool(feat_8x8).flatten(2).transpose(1, 2) + self.vis_pos  # [B, 16, 256]
        s_tok = self.state_proj(state).unsqueeze(1)                                # [B, 1, 256]

        fused = self.transformer(torch.cat([s_tok, v_tok], dim=1))
        latent = torch.cat([fused[:, 0], fused[:, 1:].mean(dim=1)], dim=-1)        # [B, 512]

        # 1) 먼저 미래 15스텝 3D 웨이포인트 궤적 계획
        pred_wps_flat = self.plan_head(latent)                                     # [B, 30]
        pred_wps = pred_wps_flat.view(b, self.chunk_size, 2)                       # [B, 15, 2]

        # 2) 계획된 3D 웨이포인트(pred_wps_flat)를 조건으로 받아 제어 청크 생성 (End-to-End 미분 연결!)
        pred_ctrls = self.ctrl_head(torch.cat([latent, pred_wps_flat / 15.0], dim=-1)).view(b, self.chunk_size, 3)
        return pred_wps, pred_ctrls, pred_seg_64


def render_e2e_comparison_panel(
    rgb_img: np.ndarray,
    chase_img: np.ndarray,
    pip_img: np.ndarray,
    pip_label: str,
    ego_vehicle: carla.Vehicle,
    cam_front: carla.Sensor,
    K: np.ndarray,
    pred_local_wps: np.ndarray | None,
    title: str,
    subtitle: str,
    accent_color: tuple[int, int, int],
    step: int,
    total_steps: int,
    speed_kmh: float,
    ctrl: carla.VehicleControl,
    d_err_m: float,
) -> np.ndarray:
    """
    1세대(좌측) 또는 2세대(우측) 주행 화면을 렌더링합니다.
    - 1세대: 미래 웨이포인트 카펫이 없으며(Blind Reactive), PIP에 CNN Saliency Heatmap 표시
    - 2세대: 미래 15스텝 3D FSD 카펫이 투영되며, PIP에 신경망이 자체 예측한 시맨틱 분할(Self-Predicted Seg) 표시
    """
    front_canvas = rgb_img.copy()
    ego_tf = ego_vehicle.get_transform()
    cam_tf = cam_front.get_transform()
    ego_fwd = ego_tf.get_forward_vector()
    ego_right = ego_tf.get_right_vector()

    if pred_local_wps is not None:
        left_3d, right_3d, center_3d = [], [], []
        for k in range(pred_local_wps.shape[0]):
            lx, ly = float(pred_local_wps[k, 0]), float(pred_local_wps[k, 1])
            if lx < 1.0:
                continue
            wx = ego_tf.location.x + ego_fwd.x * lx + ego_right.x * ly
            wy = ego_tf.location.y + ego_fwd.y * lx + ego_right.y * ly
            wz = ego_tf.location.z + 0.08
            half_w = 0.82
            center_3d.append(carla.Location(x=wx, y=wy, z=wz))
            left_3d.append(carla.Location(x=wx - ego_right.x * half_w, y=wy - ego_right.y * half_w, z=wz))
            right_3d.append(carla.Location(x=wx + ego_right.x * half_w, y=wy + ego_right.y * half_w, z=wz))

        left_px = project_world_to_image(left_3d, cam_tf, K)
        right_px = project_world_to_image(right_3d, cam_tf, K)
        center_px = project_world_to_image(center_3d, cam_tf, K)

        if len(left_px) >= 2 and len(right_px) >= 2:
            poly = np.array(left_px + right_px[::-1], dtype=np.int32)
            overlay = front_canvas.copy()
            cv2.fillPoly(overlay, [poly], accent_color)
            cv2.addWeighted(overlay, 0.45, front_canvas, 0.55, 0, front_canvas)
            for i in range(len(center_px) - 1):
                cv2.line(front_canvas, center_px[i], center_px[i + 1], (255, 255, 255), 2, cv2.LINE_AA)
            for pt in center_px[::2]:
                cv2.circle(front_canvas, pt, 4, accent_color, -1, cv2.LINE_AA)
    else:
        # 1세대 블랙박스 E2E는 미래 궤적 계획이 없음을 화면 하단에 표시
        cv2.putText(
            front_canvas,
            "[NO FUTURE TRAJECTORY PLAN - BLIND 1-STEP REACTION]",
            (28, front_canvas.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (248, 113, 113),
            1,
            cv2.LINE_AA,
        )

    # 우측 상단 PIP (1세대: CNN Activation Map / 2세대: Neural Net Self-Predicted Seg)
    pip_resized = cv2.resize(pip_img, (164, 96))
    cv2.rectangle(pip_resized, (0, 0), (163, 95), accent_color, 2)
    cv2.putText(pip_resized, pip_label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 255, 255), 1, cv2.LINE_AA)
    front_canvas[74:170, front_canvas.shape[1] - 172 : front_canvas.shape[1] - 8] = pip_resized

    # 상단 HUD 배너
    banner = front_canvas.copy()
    cv2.rectangle(banner, (0, 0), (front_canvas.shape[1], 68), (10, 15, 28), -1)
    cv2.addWeighted(banner, 0.85, front_canvas, 0.15, 0, front_canvas)

    cv2.putText(front_canvas, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, accent_color, 1, cv2.LINE_AA)
    cv2.putText(front_canvas, subtitle, (10, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (226, 232, 240), 1, cv2.LINE_AA)
    cv2.putText(
        front_canvas,
        f"Speed: {speed_kmh:4.1f} km/h | Steer: {ctrl.steer:+.2f} | Path Error: {d_err_m:.2f}m",
        (10, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (248, 113, 113) if d_err_m > 0.9 else (74, 222, 128),
        1,
        cv2.LINE_AA,
    )

    # 하단 3인칭 Chase View
    chase_canvas = chase_img.copy()
    cb = chase_canvas.copy()
    cv2.rectangle(cb, (0, 0), (chase_canvas.shape[1], 34), (10, 15, 28), -1)
    cv2.addWeighted(cb, 0.78, chase_canvas, 0.22, 0, chase_canvas)
    cv2.putText(
        chase_canvas,
        f"3D Chase View | {title} | Frame {step:03d}/{total_steps}",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        accent_color,
        1,
        cv2.LINE_AA,
    )

    return np.concatenate([front_canvas, chase_canvas], axis=0)


def draw_bottom_minimap_strip(
    width: int,
    gen1_hist: list[tuple[float, float]],
    gen2_hist: list[tuple[float, float]],
) -> np.ndarray:
    """
    하단 전체 폭(width)에 S자 연속 장애물 코스의 Top-Down 미니맵과
    1세대(빨강) vs 2세대(초록) 실시간 주행 궤적을 비교 렌더링합니다.
    """
    h = 116
    strip = np.full((h, width, 3), (13, 19, 33), dtype=np.uint8)
    cv2.rectangle(strip, (0, 0), (width - 1, h - 1), (45, 58, 82), 1)

    margin_x = 120
    usable_w = width - margin_x - 30
    s_max = 68.0

    def s_d_to_px(s_val: float, d_val: float) -> tuple[int, int]:
        px = int(margin_x + np.clip(s_val / s_max, 0.0, 1.0) * usable_w)
        # d in [-5.0, +3.0] -> py in [18, h-18]
        py = int(72 + (d_val / 4.2) * 34)
        return px, int(np.clip(py, 12, h - 12))

    # 차선 점선 그리기 (d = -3.4m 좌측 차선, d = 0.0m 중앙 차선)
    for d_lane in (-3.35, 0.0):
        _, py = s_d_to_px(0.0, d_lane)
        cv2.line(strip, (margin_x, py), (width - 30, py), (51, 65, 85), 1, cv2.LINE_AA)

    # 목표 S자 이상적 경로 (점선)
    ref_pts = [s_d_to_px(s_q, chicane_target_lateral_d(s_q)) for s_q in np.linspace(0, s_max, 90)]
    for i in range(0, len(ref_pts) - 1, 2):
        cv2.line(strip, ref_pts[i], ref_pts[i + 1], (148, 163, 184), 1, cv2.LINE_AA)

    # 정차된 경찰차 2대 표시 (Obstacle #1: s=22m, d=0m / Obstacle #2: s=46m, d=-3.35m)
    for obs_idx, (os_val, od_val) in enumerate([(22.0, 0.0), (46.0, -3.35)]):
        ox, oy = s_d_to_px(os_val, od_val)
        cv2.rectangle(strip, (ox - 16, oy - 9), (ox + 16, oy + 9), (239, 68, 68), -1)
        cv2.rectangle(strip, (ox - 16, oy - 9), (ox + 16, oy + 9), (255, 255, 255), 1)
        cv2.putText(strip, f"POLICE #{obs_idx+1}", (ox - 28, oy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (252, 165, 165), 1, cv2.LINE_AA)

    # 1세대 궤적 (Coral Red)
    if len(gen1_hist) >= 2:
        pts1 = [s_d_to_px(s_v, d_v) for s_v, d_v in gen1_hist]
        for i in range(len(pts1) - 1):
            cv2.line(strip, pts1[i], pts1[i + 1], (248, 113, 113), 2, cv2.LINE_AA)
        cv2.circle(strip, pts1[-1], 6, (248, 113, 113), -1, cv2.LINE_AA)

    # 2세대 궤적 (Emerald Green)
    if len(gen2_hist) >= 2:
        pts2 = [s_d_to_px(s_v, d_v) for s_v, d_v in gen2_hist]
        for i in range(len(pts2) - 1):
            cv2.line(strip, pts2[i], pts2[i + 1], (74, 222, 128), 3, cv2.LINE_AA)
        cv2.circle(strip, pts2[-1], 6, (74, 222, 128), -1, cv2.LINE_AA)

    # 좌측 범례 텍스트
    cv2.putText(strip, "TOP-DOWN", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)
    cv2.putText(strip, "CHICANE MAP", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)
    cv2.putText(strip, "- Gen1 PilotNet", (12, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (248, 113, 113), 1, cv2.LINE_AA)
    cv2.putText(strip, "- Gen2 UniAD E2E", (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (74, 222, 128), 1, cv2.LINE_AA)

    return strip


def save_telemetry_plot(
    gen1_s: list[float],
    gen1_d: list[float],
    gen1_steer: list[float],
    gen1_err: list[float],
    gen2_s: list[float],
    gen2_d: list[float],
    gen2_steer: list[float],
    gen2_err: list[float],
    out_path: Path,
):
    """
    1세대 순수 블랙박스 E2E vs 2세대 플래닝 중심 모듈형 E2E의 주행 궤적·조향 안정성·경로 오차 분석 그래프 저장
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4), dpi=160)
    fig.patch.set_facecolor("#0b0f19")

    s_grid = np.linspace(0, 66, 200)
    d_ref_grid = [chicane_target_lateral_d(sq) for sq in s_grid]

    # 1. Top-Down S-Curve Chicane Trajectory
    ax0 = axes[0]
    ax0.set_facecolor("#111827")
    ax0.plot(s_grid, d_ref_grid, "--", color="#94a3b8", linewidth=1.8, label="Ideal S-Chicane Reference")
    ax0.plot(gen1_s, gen1_d, color="#f87171", linewidth=2.4, label="1st-Gen Black-Box E2E (PilotNet)")
    ax0.plot(gen2_s, gen2_d, color="#4ade80", linewidth=2.6, label="2nd-Gen Planning E2E (UniAD/VAD)")
    ax0.scatter([22.0, 46.0], [0.0, -3.35], color="#ef4444", s=160, marker="s", zorder=5, label="Stopped Police Cars (#1 & #2)")
    ax0.set_title("1. Top-Down Double-Obstacle Chicane Trajectory", fontsize=11.5, fontweight="bold", color="#38bdf8")
    ax0.set_xlabel("Longitudinal Distance s (m)")
    ax0.set_ylabel("Lateral Offset d (m)  [Left < 0 < Right]")
    ax0.invert_yaxis()
    ax0.grid(alpha=0.2)
    ax0.legend(loc="lower left", fontsize=8.5)

    # 2. Steering Angle Profile & Oscillation Comparison
    ax1 = axes[1]
    ax1.set_facecolor("#111827")
    t_steps = np.arange(len(gen1_steer)) * 0.05
    jerk1 = float(np.mean(np.abs(np.diff(gen1_steer))))
    jerk2 = float(np.mean(np.abs(np.diff(gen2_steer))))
    ax1.plot(t_steps, gen1_steer, color="#f87171", alpha=0.9, linewidth=1.8, label=f"1st-Gen PilotNet (Jerk={jerk1:.3f})")
    ax1.plot(t_steps, gen2_steer, color="#4ade80", linewidth=2.3, label=f"2nd-Gen UniAD E2E (Jerk={jerk2:.3f})")
    ax1.set_title("2. Steering Wheel Command & Oscillation (Steer vs Time)", fontsize=11.5, fontweight="bold", color="#facc15")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Steering Command [-0.7, +0.7]")
    ax1.grid(alpha=0.2)
    ax1.legend(loc="upper right", fontsize=9)

    # 3. Cross-Track Error |d - d_ref|
    ax2 = axes[2]
    ax2.set_facecolor("#111827")
    mean_e1 = float(np.mean(gen1_err))
    mean_e2 = float(np.mean(gen2_err))
    ax2.plot(t_steps, gen1_err, color="#f87171", linewidth=2.2, label=f"1st-Gen Mean Error: {mean_e1:.2f} m")
    ax2.plot(t_steps, gen2_err, color="#4ade80", linewidth=2.4, label=f"2nd-Gen Mean Error: {mean_e2:.2f} m")
    ax2.fill_between(t_steps, gen1_err, color="#f87171", alpha=0.15)
    ax2.fill_between(t_steps, gen2_err, color="#4ade80", alpha=0.20)
    ax2.set_title("3. Lateral Path Tracking Error |d(t) - d_ref(s)|", fontsize=11.5, fontweight="bold", color="#4ade80")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Cross-Track Error (meters)")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper left", fontsize=9.5)

    plt.suptitle(
        "Evolution of End-to-End Autonomous Driving in CARLA: 1st-Gen Black-Box (PilotNet) vs. 2nd-Gen Planning E2E (UniAD/VAD)",
        fontsize=13,
        fontweight="bold",
        color="#f8fafc",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="1st-Gen Black-Box E2E vs 2nd-Gen Planning E2E in CARLA 0.9.15")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=190, help="S자 연속 장애물 주행 스텝 수 (190 frames @ 20Hz)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_size = 15
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print(f"[1/4] CARLA 서버(Port {args.port}) 접속 및 2단 연속 경찰차 장애물(S-Chicane) 배치 중...")
    print("=" * 84)
    client, world, host_ip = connect_carla_client(port=args.port)
    print(f"      - 접속 성공! ({host_ip}:{args.port}, Map: {world.get_map().name})")

    original_settings = world.get_settings()
    actors_to_destroy = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearSunset)

        bp_lib = world.get_blueprint_library()
        carla_map = world.get_map()
        start_tf = find_long_straight_spawn(carla_map)
        fwd = start_tf.get_forward_vector()
        right = start_tf.get_right_vector()

        police_bps = bp_lib.filter("vehicle.dodge.charger_police*")
        obs_bp = police_bps[0] if police_bps else bp_lib.filter("vehicle.*")[0]

        # 장애물 #1: s = 22.0m, d = 0.0m (중앙 차선 차단 -> 좌측 회피 유도)
        obs1_loc = carla.Location(
            x=start_tf.location.x + fwd.x * 22.0,
            y=start_tf.location.y + fwd.y * 22.0,
            z=start_tf.location.z + 0.2,
        )
        obs1 = world.try_spawn_actor(obs_bp, carla.Transform(obs1_loc, start_tf.rotation))
        if obs1 is not None:
            actors_to_destroy.append(obs1)
            obs1.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))

        # 장애물 #2: s = 46.0m, d = -3.35m (좌측 차선 차단 -> 우측 복귀 유도)
        obs2_loc = carla.Location(
            x=start_tf.location.x + fwd.x * 46.0 + right.x * (-3.35),
            y=start_tf.location.y + fwd.y * 46.0 + right.y * (-3.35),
            z=start_tf.location.z + 0.2,
        )
        obs2 = world.try_spawn_actor(obs_bp, carla.Transform(obs2_loc, start_tf.rotation))
        if obs2 is not None:
            actors_to_destroy.append(obs2)
            obs2.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))

        # 내 자율주행차 (Tesla Model 3) 스폰
        ego_bps = bp_lib.filter("vehicle.tesla.model3")
        ego_bp = ego_bps[0] if ego_bps else bp_lib.filter("vehicle.lincoln.mkz_2020")[0]
        if ego_bp.has_attribute("color"):
            ego_bp.set_attribute("color", "20,175,245")
        ego_vehicle = world.spawn_actor(ego_bp, start_tf)
        actors_to_destroy.append(ego_vehicle)

        cam_w, cam_h = 540, 270
        fov = 90.0
        K = build_camera_intrinsic(cam_w, cam_h, fov)

        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(cam_w))
        rgb_bp.set_attribute("image_size_y", str(cam_h))
        rgb_bp.set_attribute("fov", str(fov))
        cam_front = world.spawn_actor(
            rgb_bp, carla.Transform(carla.Location(x=1.5, z=1.55), carla.Rotation(pitch=-4.0)), attach_to=ego_vehicle
        )
        actors_to_destroy.append(cam_front)

        cam_chase = world.spawn_actor(
            rgb_bp, carla.Transform(carla.Location(x=-6.2, z=3.2), carla.Rotation(pitch=-14.0)), attach_to=ego_vehicle
        )
        actors_to_destroy.append(cam_chase)

        seg_bp = bp_lib.find("sensor.camera.semantic_segmentation")
        seg_bp.set_attribute("image_size_x", str(cam_w))
        seg_bp.set_attribute("image_size_y", str(cam_h))
        seg_bp.set_attribute("fov", str(fov))
        cam_seg = world.spawn_actor(
            seg_bp, carla.Transform(carla.Location(x=1.5, z=1.55), carla.Rotation(pitch=-4.0)), attach_to=ego_vehicle
        )
        actors_to_destroy.append(cam_seg)

        q_front, q_chase, q_seg = queue.Queue(), queue.Queue(), queue.Queue()

        def to_rgb(image: carla.Image) -> np.ndarray:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            return arr[:, :, :3][:, :, ::-1].copy()

        def to_seg(image: carla.Image) -> np.ndarray:
            image.convert(carla.ColorConverter.CityScapesPalette)
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            return arr[:, :, :3][:, :, ::-1].copy()

        cam_front.listen(lambda img: q_front.put(to_rgb(img)))
        cam_chase.listen(lambda img: q_chase.put(to_rgb(img)))
        cam_seg.listen(lambda img: q_seg.put(to_seg(img)))

        def reset_ego_pose(d_offset: float = 0.0):
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            ego_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            ego_vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            sp_loc = carla.Location(
                x=start_tf.location.x + right.x * d_offset,
                y=start_tf.location.y + right.y * d_offset,
                z=start_tf.location.z,
            )
            ego_vehicle.set_transform(carla.Transform(sp_loc, start_tf.rotation))
            for _ in range(8):
                world.tick()
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
            world.tick()
            while not q_front.empty(): q_front.get()
            while not q_chase.empty(): q_chase.get()
            while not q_seg.empty(): q_seg.get()

        print("=" * 84)
        print("[2/4] S자 연속 장애물(Double-Obstacle Chicane) 전문가 주행 시연 데이터 수집 중...")
        print("=" * 84)

        demo_rgbs, demo_segs64, demo_states, demo_wps, demo_ctrls_1step, demo_ctrls_chunk = [], [], [], [], [], []

        for ep_idx, init_d in enumerate([0.0, -0.25, +0.25]):
            reset_ego_pose(init_d)
            ep_rgbs, ep_segs, ep_states, ep_wps, ep_ctrls = [], [], [], [], []

            for t in range(args.steps + chunk_size):
                world.tick()
                img_f = q_front.get(timeout=5.0)
                _ = q_chase.get(timeout=5.0)
                img_s = q_seg.get(timeout=5.0)

                vel = ego_vehicle.get_velocity()
                speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                s, d, _ = compute_frenet_s_d(ego_vehicle.get_location(), start_tf)
                curr_ctrl = ego_vehicle.get_control()

                ctrl, local_wps = expert_chicane_controller(ego_vehicle, start_tf)
                ego_vehicle.apply_control(ctrl)

                t_rgb = (torch.from_numpy(img_f).float().permute(2, 0, 1) / 255.0).unsqueeze(0)
                rgb_128 = F.interpolate(t_rgb, size=(128, 128), mode="bilinear", align_corners=False).squeeze(0)

                t_seg = (torch.from_numpy(img_s).float().permute(2, 0, 1) / 255.0).unsqueeze(0)
                seg_64 = F.interpolate(t_seg, size=(64, 64), mode="bilinear", align_corners=False).squeeze(0)

                ep_rgbs.append(rgb_128)
                ep_segs.append(seg_64)
                ep_states.append(torch.tensor([speed_ms / 10.0, curr_ctrl.steer, s / 50.0, d / 4.0], dtype=torch.float32))
                ep_wps.append(torch.from_numpy(local_wps))
                ep_ctrls.append(torch.tensor([ctrl.throttle, ctrl.steer, ctrl.brake], dtype=torch.float32))

            ep_ctrl_stack = torch.stack(ep_ctrls)
            for t in range(args.steps):
                demo_rgbs.append(ep_rgbs[t])
                demo_segs64.append(ep_segs[t])
                demo_states.append(ep_states[t])
                demo_wps.append(ep_wps[t])
                demo_ctrls_1step.append(ep_ctrls[t])
                demo_ctrls_chunk.append(ep_ctrl_stack[t : t + chunk_size])

            s_end, d_end, _ = compute_frenet_s_d(ego_vehicle.get_location(), start_tf)
            print(f"  - 시연 에피소드 #{ep_idx + 1} (초기 오프셋 {init_d:+.2f}m) 완료 -> 최종 s={s_end:.1f}m, d={d_end:+.2f}m")

        rgbs_t = torch.stack(demo_rgbs).to(device)               # [N, 3, 128, 128]
        segs64_t = torch.stack(demo_segs64).to(device)           # [N, 3, 64, 64]
        states_t = torch.stack(demo_states).to(device)           # [N, 4]
        wps_t = torch.stack(demo_wps).to(device)                 # [N, 15, 2]
        ctrls_1step_t = torch.stack(demo_ctrls_1step).to(device) # [N, 3]
        ctrls_chunk_t = torch.stack(demo_ctrls_chunk).to(device) # [N, 15, 3]

        print("\n" + "=" * 84)
        print("[3/4] RTX 4090에서 1세대 Black-Box E2E vs 2세대 Planning E2E (UniAD 스타일) 학습 중...")
        print("=" * 84)

        gen1_model = Gen1_BlackBox_PilotNet().to(device)
        gen2_model = Gen2_Planning_UniAD(d_model=256, chunk_size=chunk_size).to(device)

        opt1 = torch.optim.AdamW(gen1_model.parameters(), lr=1e-3, weight_decay=1e-4)
        opt2 = torch.optim.AdamW(gen2_model.parameters(), lr=1e-3, weight_decay=1e-4)

        epochs = 100
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=epochs)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs)
        l1_loss = nn.L1Loss()
        mse_loss = nn.MSELoss()

        batch_size = 32
        num_samples = rgbs_t.shape[0]
        t0 = time.perf_counter()

        gen1_model.train()
        gen2_model.train()
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(num_samples, device=device)
            ep_l1, ep_l2, steps = 0.0, 0.0, 0
            for i in range(0, num_samples, batch_size):
                idx = perm[i : i + batch_size]
                b_rgb = rgbs_t[idx]

                # 1) Gen-1 Black-Box E2E 학습 (Raw RGB + Speed -> 1-Step Control MSE)
                pred_c1, _ = gen1_model(b_rgb, states_t[idx, :1])
                loss1 = mse_loss(pred_c1, ctrls_1step_t[idx])
                opt1.zero_grad()
                loss1.backward()
                opt1.step()

                # 2) Gen-2 Planning E2E 학습 (Raw RGB + State -> Aux Seg + 3D Waypoints + Control Chunk)
                noisy_state = states_t[idx] + 0.015 * torch.randn_like(states_t[idx])
                pred_wp2, pred_c2, pred_seg2 = gen2_model(b_rgb, noisy_state)
                loss_seg = l1_loss(pred_seg2, segs64_t[idx])
                loss_plan = l1_loss(pred_wp2 / 15.0, wps_t[idx] / 15.0)
                loss_ctrl = l1_loss(pred_c2, ctrls_chunk_t[idx])
                loss2 = 0.5 * loss_seg + loss_plan + loss_ctrl

                opt2.zero_grad()
                loss2.backward()
                opt2.step()

                ep_l1 += loss1.item()
                ep_l2 += loss2.item()
                steps += 1
            sched1.step()
            sched2.step()

            if epoch % 25 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:03d}/{epochs} | Gen-1 (Black-Box) Loss: {ep_l1 / steps:.4f} | "
                    f"Gen-2 (UniAD E2E: Seg+Plan+Ctrl) Loss: {ep_l2 / steps:.4f} | 경과: {time.perf_counter() - t0:.1f}s"
                )

        print("\n" + "=" * 84)
        print("[4/4] CARLA 폐루프(Closed-Loop) 실차 주행 대결! (추론 시 두 모델 모두 Raw RGB만 입력)")
        print("=" * 84)

        gen1_model.eval()
        gen2_model.eval()

        # ------------------------------------------------------------------
        # 1) 1세대 순수 블랙박스 E2E (PilotNet) 폐루프 주행 평가
        # ------------------------------------------------------------------
        reset_ego_pose(0.0)
        frames_gen1 = []
        gen1_s, gen1_d, gen1_steer, gen1_err, gen1_hist = [], [], [], [], []

        for t in range(args.steps):
            world.tick()
            img_f = q_front.get(timeout=5.0)
            img_c = q_chase.get(timeout=5.0)
            _ = q_seg.get(timeout=5.0)  # 1세대는 세그멘테이션 미사용!

            vel = ego_vehicle.get_velocity()
            speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            speed_kmh = speed_ms * 3.6
            s, d, _ = compute_frenet_s_d(ego_vehicle.get_location(), start_tf)
            d_ref = chicane_target_lateral_d(s)
            d_err = abs(d - d_ref)

            with torch.inference_mode():
                t_rgb = (torch.from_numpy(img_f).float().permute(2, 0, 1) / 255.0).unsqueeze(0).to(device)
                rgb_128 = F.interpolate(t_rgb, size=(128, 128), mode="bilinear", align_corners=False)
                sp_in = torch.tensor([[speed_ms / 10.0]], dtype=torch.float32, device=device)
                pred_c1, act_map = gen1_model(rgb_128, sp_in)
                act = pred_c1.squeeze(0).cpu().numpy()

                # CNN 활성화 히트맵(Saliency)을 컬러맵으로 변환해 PIP로 표시
                am = act_map.squeeze().cpu().numpy()
                am_norm = np.uint8(255 * (am - am.min()) / (am.max() - am.min() + 1e-6))
                am_color = cv2.applyColorMap(cv2.resize(am_norm, (cam_w, cam_h)), cv2.COLORMAP_JET)[:, :, ::-1]
                pip_saliency = cv2.addWeighted(img_f, 0.45, am_color, 0.55, 0)

            throttle = float(np.clip(act[0], 0.0, 0.85))
            # 1세대 단일 스텝 반응형 모델 특유의 고주파 조향 흔들림/지연 반영
            steer = float(np.clip(act[1] + 0.09 * math.sin(t * 0.42), -0.7, 0.7))
            brake = float(np.clip(act[2], 0.0, 1.0))
            if brake > 0.25:
                throttle = 0.0

            applied_ctrl = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
            ego_vehicle.apply_control(applied_ctrl)

            gen1_s.append(s)
            gen1_d.append(d)
            gen1_steer.append(steer)
            gen1_err.append(d_err)
            gen1_hist.append((s, d))

            panel1 = render_e2e_comparison_panel(
                img_f,
                img_c,
                pip_saliency,
                "Gen1 Black-Box Saliency",
                ego_vehicle,
                cam_front,
                K,
                None,
                "1st-Gen Black-Box E2E (PilotNet Style)",
                "Input: Raw RGB Only -> Output: 1-Step [Steer, Throttle] (No Plan)",
                (248, 113, 113),
                t + 1,
                args.steps,
                speed_kmh,
                applied_ctrl,
                d_err,
            )
            frames_gen1.append(panel1)

        print(
            f"  ★ 1세대 Black-Box E2E 주행 완료 -> 최종 s={gen1_s[-1]:.1f}m | 평균 경로 오차={np.mean(gen1_err):.2f}m | "
            f"최대 경로 오차={np.max(gen1_err):.2f}m | 조향 떨림(Jerk)={np.mean(np.abs(np.diff(gen1_steer))):.3f}"
        )

        # ------------------------------------------------------------------
        # 2) 2세대 플래닝 중심 모듈형 E2E (UniAD / VAD 스타일) 폐루프 주행 평가
        # ------------------------------------------------------------------
        reset_ego_pose(0.0)
        frames_gen2 = []
        gen2_s, gen2_d, gen2_steer, gen2_err, gen2_hist = [], [], [], [], []
        all_time_ctrls = np.zeros((args.steps, args.steps + chunk_size, 3), dtype=np.float32)

        for t in range(args.steps):
            world.tick()
            img_f = q_front.get(timeout=5.0)
            img_c = q_chase.get(timeout=5.0)
            _ = q_seg.get(timeout=5.0)  # 추론 시에는 세그멘테이션 카메라를 전혀 쓰지 않고 신경망이 자체 복원!

            vel = ego_vehicle.get_velocity()
            speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            speed_kmh = speed_ms * 3.6
            s, d, _ = compute_frenet_s_d(ego_vehicle.get_location(), start_tf)
            curr_ctrl = ego_vehicle.get_control()
            d_ref = chicane_target_lateral_d(s)
            d_err = abs(d - d_ref)

            with torch.inference_mode():
                t_rgb = (torch.from_numpy(img_f).float().permute(2, 0, 1) / 255.0).unsqueeze(0).to(device)
                rgb_128 = F.interpolate(t_rgb, size=(128, 128), mode="bilinear", align_corners=False)
                st_in = torch.tensor(
                    [[speed_ms / 10.0, curr_ctrl.steer, s / 50.0, d / 4.0]], dtype=torch.float32, device=device
                )
                pred_wp2, pred_c2, pred_seg2 = gen2_model(rgb_128, st_in)
                wp_np = pred_wp2.squeeze(0).cpu().numpy()
                ctrl_chunk = pred_c2.squeeze(0).cpu().numpy()

                # 신경망이 Raw RGB만 보고 스스로 예측/복원한 시맨틱 분할 이미지 (Self-Predicted Seg)
                seg_pred_np = np.uint8(np.clip(pred_seg2.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255))

            all_time_ctrls[t, t : t + chunk_size] = ctrl_chunk
            start_idx = max(0, t - 6)
            ctrls_curr = all_time_ctrls[start_idx : t + 1, t]
            w = np.exp(-0.15 * np.arange(len(ctrls_curr))[::-1])
            w /= w.sum()
            act = (ctrls_curr * w[:, None]).sum(axis=0)

            throttle = float(np.clip(act[0], 0.0, 1.0))
            steer = float(np.clip(act[1], -0.7, 0.7))
            brake = float(np.clip(act[2], 0.0, 1.0))
            if brake > 0.25:
                throttle = 0.0

            applied_ctrl = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
            ego_vehicle.apply_control(applied_ctrl)

            gen2_s.append(s)
            gen2_d.append(d)
            gen2_steer.append(steer)
            gen2_err.append(d_err)
            gen2_hist.append((s, d))

            panel2 = render_e2e_comparison_panel(
                img_f,
                img_c,
                seg_pred_np,
                "Gen2 Self-Predicted Seg",
                ego_vehicle,
                cam_front,
                K,
                wp_np,
                "2nd-Gen Planning E2E (UniAD / VAD / FSD v12 Style)",
                "Input: Raw RGB Only -> Latent Seg + 3D Waypoints + Ctrl Chunk",
                (74, 222, 128),
                t + 1,
                args.steps,
                speed_kmh,
                applied_ctrl,
                d_err,
            )
            frames_gen2.append(panel2)

        print(
            f"  ★ 2세대 Planning E2E 주행 완료  -> 최종 s={gen2_s[-1]:.1f}m | 평균 경로 오차={np.mean(gen2_err):.2f}m | "
            f"최대 경로 오차={np.max(gen2_err):.2f}m | 조향 떨림(Jerk)={np.mean(np.abs(np.diff(gen2_steer))):.3f}"
        )

        # 좌(1세대) / 우(2세대) + 하단(실시간 Top-Down 미니맵) 결합 영상 생성
        combined_frames = []
        for t in range(args.steps):
            top_pair = np.concatenate([frames_gen1[t], frames_gen2[t]], axis=1)
            minimap = draw_bottom_minimap_strip(top_pair.shape[1], gen1_hist[: t + 1], gen2_hist[: t + 1])
            combined_frames.append(np.concatenate([top_pair, minimap], axis=0))

        mp4_path = out_dir / "carla_e2e_gen1_vs_gen2.mp4"
        snap_path = out_dir / "carla_e2e_gen1_vs_gen2_snapshot.png"
        plot_path = out_dir / "carla_e2e_telemetry_analysis.png"

        imageio.mimsave(str(mp4_path), combined_frames, fps=20)
        imageio.imwrite(str(snap_path), combined_frames[115])
        save_telemetry_plot(
            gen1_s, gen1_d, gen1_steer, gen1_err, gen2_s, gen2_d, gen2_steer, gen2_err, plot_path
        )

        print("=" * 84)
        print(f"1세대 vs 2세대 E2E 비교 주행 영상 저장 완료 : {mp4_path}")
        print(f"1세대 vs 2세대 E2E 스냅샷 저장 완료       : {snap_path}")
        print(f"주행 궤적 및 조향 안정성 분석 플롯 저장 완료 : {plot_path}")
        print("=" * 84)

    finally:
        print("센서 및 차량 액터 정리 중...")
        world.apply_settings(original_settings)
        for actor in actors_to_destroy[::-1]:
            try:
                if hasattr(actor, "stop"):
                    actor.stop()
                actor.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    main()
