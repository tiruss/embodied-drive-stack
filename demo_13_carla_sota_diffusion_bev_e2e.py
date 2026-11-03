"""
Demo 13: 2025-2026 SOTA End-to-End Autonomous Driving in CARLA 0.9.15 (RTX 4090)
         DiffusionDrive (Multi-Modal Truncated Flow-Matching) + Hydra-MDP (Neural Scorer) + BEV Occupancy
=========================================================================================================
기존 2세대 E2E(UniAD / VAD: 단일 궤적 L2 회귀)의 치명적 한계인 '모드 평균화(Mode Averaging)' 문제를 극복한
2025~2026년 최신 SOTA E2E 자율주행 아키텍처(DiffusionDrive + Hydra-MDP + BEV Occupancy + System 1/2 VLA)를
구현하고, CARLA 도심 210m 6단계 초장거리 복합 장애물 코스(6-Stage Urban Gauntlet, 9대 장애물)에서 검증합니다!

[SOTA E2E 4대 핵심 아키텍처 구성요소 (추론 시 오직 전방 Raw RGB 카메라 + 차량 상태만 사용)]
1. Monocular-to-BEV Spatial Lifting & Top-Down BEV Occupancy Decoder (L_bev):
   - 전방 원근(Perspective) RGB 카메라(128x128) 특징을 2D 조감도(Bird's-Eye-View, 64x64, 전방 36m x 폭 16m)
     잠재 공간으로 변환하여 [배경 / 주행 가능 도로 / 차선 중심 / 장애물 차량 Footprint]를 실시간 예측합니다.
   - 추론 시 LiDAR나 GT 센서 없이도 신경망이 머릿속으로 상상한 Top-Down BEV 레이더 맵을 HUD에 실시간 표시합니다!

2. Multi-Modal Anchor + Truncated Flow-Matching Generator (DiffusionDrive 스타일, M=5 Modes, L_flow):
   - 단일 궤적만 예측하다가 좌/우 갈림길에서 정중앙 장애물로 돌진하는 2세대 L2 회귀의 한계를 해결하기 위해,
     M=5개의 공간 앵커(급좌회피, 완만한 좌측, 직진, 완만한 우측, 급우회피/감속)에서 출발하여
     4-Step Truncated Euler ODE Flow-Matching으로 5개의 3D 후보 궤적(5 x 15 x 2)을 동시에 생성합니다.

3. Hydra-MDP Neural Multi-Metric Trajectory Scorer (L_score):
   - 예측된 BEV Occupancy 특징 위에서 5개 후보 궤적 각각의 안전도/진행도 점수 S_m in [0, 100]% (무충돌성 NC,
     도로 이탈 방지 DAC, 목표 부합도 EP)를 신경망이 실시간 채점합니다.
   - 전면 유리창(Windshield)과 BEV 미니맵 위에 5개 후보 궤적을 모두 투영하되,
     위험 궤적은 붉은 점선(Red Dashed), 차선책은 청록색 실선, 최고점 최적 궤적(m*)은 에메랄드 FSD 카펫으로 렌더링합니다!

4. 210m 6-Stage Long-Horizon Urban Gauntlet (총 9대 장애물 차량 배치, 340 Steps 장거리 주행):
   - Stage 1 (s=0~34m)  : 중앙 차선 경찰차 고장 -> 좌측 차선 추월 (Left Overtake)
   - Stage 2 (s=34~72m) : 좌측+중앙 연속 봉쇄 -> 우측 차선 대각선 크로스오버 (Right Crossover)
   - Stage 3 (s=72~112m): 양방향 갈림길 함정(Mode-Averaging Trap) -> 급좌측 회피 탈출 (Sharp Left Escape)
   - Stage 4 (s=112~152m): 양측 이중 주차 협로 게이트 -> 정중앙 바늘구멍 통과 (Thread-the-Needle Gate)
   - Stage 5 (s=152~184m): 고속 S자 슬라럼 우측 패스 후 차선 복귀 (High-Speed S-Slalom Bypass)
   - Stage 6 (s=184~210m): 종착지 전면 바리케이드 앞 정밀 감속 및 완전 정차 (Precision Terminal Stop)
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


# ==============================================================================
# 1. 장거리 곡선/직선 도로 중심선 스플라인 (220m Continuous Road Route Spline)
# ==============================================================================
class CarlaRouteSpline:
    """
    CARLA 맵의 Waypoint 그래프를 따라 220m 이상의 연속 주행 경로(직선 + 완만한 곡선 + 교차로 통과)를 추출하고,
    임의의 월드 좌표 (x, y, z) <-> 곡선 도로 기준 Frenet 좌표 (s: 누적 주행거리, d: 차선 횡방향 오프셋) 변환을 수행합니다.
    """
    def __init__(self, waypoints: list[carla.Waypoint]):
        self.waypoints = waypoints
        pts = []
        fwds = []
        rights = []
        yaws = []
        for wp in waypoints:
            tf = wp.transform
            loc = tf.location
            fwd = tf.get_forward_vector()
            right = tf.get_right_vector()
            pts.append([loc.x, loc.y, loc.z])
            fwds.append([fwd.x, fwd.y, fwd.z])
            rights.append([right.x, right.y, right.z])
            yaws.append(tf.rotation.yaw)

        self.pts = np.array(pts, dtype=np.float64)        # [N, 3]
        self.fwds = np.array(fwds, dtype=np.float64)      # [N, 3]
        self.rights = np.array(rights, dtype=np.float64)  # [N, 3]
        self.yaws = np.array(yaws, dtype=np.float64)      # [N]

        diffs = np.diff(self.pts[:, :2], axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        self.s_arr = np.concatenate([[0.0], np.cumsum(seg_lens)])
        self.total_length = float(self.s_arr[-1])

    def world_to_frenet(self, loc: carla.Location) -> tuple[float, float, float]:
        """월드 좌표를 (누적 거리 s [m], 횡방향 오프셋 d [m], 도로 접선 yaw [deg])로 변환"""
        dx = self.pts[:, 0] - loc.x
        dy = self.pts[:, 1] - loc.y
        dist2 = dx * dx + dy * dy
        idx = int(np.argmin(dist2))

        p = self.pts[idx]
        f = self.fwds[idx]
        r = self.rights[idx]
        rel_x = loc.x - p[0]
        rel_y = loc.y - p[1]

        s = float(self.s_arr[idx] + (rel_x * f[0] + rel_y * f[1]))
        d = float(rel_x * r[0] + rel_y * r[1])
        ref_yaw = float(self.yaws[idx])
        return s, d, ref_yaw

    def get_tangent_yaw_at_s(self, s: float) -> float:
        """누적 거리 s [m] 지점의 도로 중심선 접선 각도(yaw [deg])를 반환"""
        s_clamped = float(np.clip(s, 0.0, self.total_length))
        idx = int(np.searchsorted(self.s_arr, s_clamped))
        idx = min(max(idx, 1), len(self.s_arr) - 1)
        f = self.fwds[idx]
        return float(math.degrees(math.atan2(f[1], f[0])))

    def frenet_to_world(self, s: float, d: float, z_offset: float = 0.0) -> carla.Transform:
        """Frenet 좌표 (s, d)를 CARLA 월드 Transform으로 변환"""
        s_clamped = float(np.clip(s, 0.0, self.total_length))
        idx = int(np.searchsorted(self.s_arr, s_clamped))
        idx = min(max(idx, 1), len(self.s_arr) - 1)

        s0, s1 = self.s_arr[idx - 1], self.s_arr[idx]
        alpha = (s_clamped - s0) / max(1e-6, s1 - s0)

        p = (1.0 - alpha) * self.pts[idx - 1] + alpha * self.pts[idx]
        r = (1.0 - alpha) * self.rights[idx - 1] + alpha * self.rights[idx]
        r_norm = np.linalg.norm(r[:2])
        if r_norm > 1e-6:
            r[:2] /= r_norm

        f = (1.0 - alpha) * self.fwds[idx - 1] + alpha * self.fwds[idx]
        yaw_deg = math.degrees(math.atan2(f[1], f[0]))

        wx = float(p[0] + d * r[0])
        wy = float(p[1] + d * r[1])
        wz = float(p[2] + z_offset)
        return carla.Transform(
            carla.Location(x=wx, y=wy, z=wz),
            carla.Rotation(pitch=0.0, yaw=yaw_deg, roll=0.0),
        )


def find_best_long_gauntlet_route(carla_map: carla.Map, target_len_m: float = 215.0) -> CarlaRouteSpline:
    """
    Town10HD_Opt 전체 스폰 포인트 중 초반 120m 이상이 직선 대로이고
    이후 완만한 도심 곡선으로 총 215m 이상 이어지는 최적의 다차선 주행 경로(CarlaRouteSpline)를 탐색합니다.
    """
    best_spline = None
    best_score = -1e9
    n_steps = int(target_len_m / 1.5)

    for sp in carla_map.get_spawn_points():
        wp0 = carla_map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp0 is None or wp0.is_junction:
            continue

        wps = [wp0]
        cur = wp0
        early_turn = 0.0
        late_turn = 0.0
        side_lane_bonus = 0.0

        for step_i in range(n_steps):
            cands = cur.next(1.5)
            if not cands:
                break
            cur_fwd = cur.transform.get_forward_vector()
            best_c = max(
                cands,
                key=lambda c: cur_fwd.x * c.transform.get_forward_vector().x
                + cur_fwd.y * c.transform.get_forward_vector().y,
            )
            nxt_fwd = best_c.transform.get_forward_vector()
            dot = float(np.clip(cur_fwd.x * nxt_fwd.x + cur_fwd.y * nxt_fwd.y, -1.0, 1.0))
            turn_angle = abs(math.degrees(math.acos(dot)))
            if step_i * 1.5 < 115.0:
                early_turn += turn_angle
            else:
                late_turn += turn_angle

            if best_c.get_left_lane() is not None and best_c.get_left_lane().lane_type == carla.LaneType.Driving:
                side_lane_bonus += 0.25
            if best_c.get_right_lane() is not None and best_c.get_right_lane().lane_type == carla.LaneType.Driving:
                side_lane_bonus += 0.25
            wps.append(best_c)
            cur = best_c

        if len(wps) < n_steps - 5:
            continue

        spline = CarlaRouteSpline(wps)
        if spline.total_length < target_len_m - 15.0:
            continue

        # 초반 115m 직진성 최우선 + 전체 회전각 최소화 + 좌우 차선 보너스
        score = spline.total_length - 4.5 * early_turn - 0.8 * late_turn + side_lane_bonus
        if score > best_score:
            best_score = score
            best_spline = spline

    return best_spline


# ==============================================================================
# 2. 210m 6-Stage 도심 복합 장애물 코스 정의 (총 9대 장애물 차량 & VLM Subgoal)
# ==============================================================================
GAUNTLET_OBSTACLES = [
    # (s_m, d_m, blueprint_filter, role_desc)
    (22.0,  0.00, "vehicle.dodge.charger_police_2020", "Obs#1: Stage1 Center Breakdown"),
    (52.0, -3.20, "vehicle.dodge.charger_police_2020", "Obs#2: Stage2 Left Blockade"),
    (56.0, -0.35, "vehicle.tesla.model3",              "Obs#3: Stage2 Center-Left Double Block"),
    (88.0, +3.20, "vehicle.dodge.charger_police_2020", "Obs#4: Stage3 Right Blockade"),
    (91.5, +0.20, "vehicle.tesla.model3",              "Obs#5: Stage3 Bifurcation Center Trap"),
    (126.0, -3.30, "vehicle.dodge.charger_police_2020", "Obs#6: Stage4 Narrow Gate Left Pillar"),
    (128.0, +3.30, "vehicle.dodge.charger_police_2020", "Obs#7: Stage4 Narrow Gate Right Pillar"),
    (160.0, -0.20, "vehicle.tesla.model3",              "Obs#8: Stage5 Final Slalom Hazard"),
    (198.0,  0.00, "vehicle.dodge.charger_police_2020", "Obs#9: Stage6 Terminal Stop Barrier"),
]


def gauntlet_target_lateral_d(s_query: float) -> float:
    """
    210m 6단계 복합 장애물 코스의 이상적인 전문가 목표 횡방향 경로 d_ref(s) [m]:
      - Stage 1 (s=0~34m)  : 중앙 장애물(s=22, d=0.0) 회피 -> 좌측 차선(d=-3.2m) 진입
      - Stage 2 (s=34~72m) : 좌측+중앙 이중 장애물(s=52~56) 회피 -> 우측 차선(d=+3.2m) 크로스오버
      - Stage 3 (s=72~110m): 우측+중앙 갈림길 장애물(s=88~92) 회피 -> 급좌측 차선(d=-3.2m) 탈출
      - Stage 4 (s=110~146m): 양측 협로 게이트(s=126~128, d=+-3.3) 통과 -> 정중앙(d=0.0m) 바늘구멍 관통
      - Stage 5 (s=146~178m): 전방 장애물(s=160, d=-0.2) 우측 패스(d=+3.1m) 후 중앙 차선(d=0.0m) 복귀
      - Stage 6 (s=178~210m): 중앙 차선(d=0.0m) 유지하며 종착지 바리케이드(s=198m) 앞 정밀 정차
    """
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    d = 0.0
    # Stage 1: Left overtake (0 -> -3.25m around s=10m)
    d += -3.25 * sig(0.38 * (s_query - 10.0))
    # Stage 2: Cross over to Right lane (-3.25 -> +3.25m around s=36m)
    d += +6.50 * sig(0.35 * (s_query - 36.5))
    # Stage 3: Bifurcation escape to Left lane (+3.25 -> -3.25m around s=72m)
    d += -6.50 * sig(0.36 * (s_query - 72.5))
    # Stage 4: Thread the needle to Center gate (-3.25 -> 0.0m around s=109m)
    d += +3.25 * sig(0.38 * (s_query - 109.0))
    # Stage 5a: Swerve Right around Obs#8 (0.0 -> +3.15m around s=144m)
    d += +3.15 * sig(0.38 * (s_query - 144.5))
    # Stage 5b: Return to Center for Terminal Stop (+3.15 -> 0.0m around s=173m)
    d += -3.15 * sig(0.38 * (s_query - 173.0))
    return float(d)


def get_stage_info(s: float) -> tuple[int, str, str]:
    """현재 주행 거리 s에 따른 Stage 번호, System-2 VLM 명령, 핵심 테스트 포인트 반환"""
    if s < 33.0:
        return 1, "STAGE 1/6: Overtake center breakdown via LEFT lane", "Single Obstacle Left Overtake"
    elif s < 70.0:
        return 2, "STAGE 2/6: Double blockade ahead! Cross over to RIGHT lane", "Diagonal 2-Lane Crossover"
    elif s < 108.0:
        return 3, "STAGE 3/6: Bifurcation trap! Execute sharp LEFT escape", "Mode-Averaging Trap (Multi-Modal Test)"
    elif s < 143.0:
        return 4, "STAGE 4/6: Thread the needle through narrow CENTER gate", "3.4m Narrow Corridor Squeeze"
    elif s < 174.0:
        return 5, "STAGE 5/6: High-speed RIGHT slalom bypass & re-center", "Dynamic S-Curve Chicane"
    else:
        return 6, "STAGE 6/6: Hazard barricade ahead! Smooth stop at s=190m", "Precision Terminal Deceleration"


# ==============================================================================
# 3. 전문가 제어기 + Top-Down BEV Occupancy GT 생성기 + M=5 멀티모달 앵커 정의
# ==============================================================================
def create_spatial_anchors(chunk_size: int = 15) -> np.ndarray:
    """
    DiffusionDrive 스타일의 M=5 공간 궤적 앵커 (Spatial Trajectory Anchors, [5, 15, 2]):
      - Mode 0: Sharp Left Overtake  (급좌회피 궤적)
      - Mode 1: Smooth Left Cruise   (완만한 좌측 주행)
      - Mode 2: Center Lane Keep     (직진 차선 유지)
      - Mode 3: Smooth Right Cruise  (완만한 우측 주행)
      - Mode 4: Sharp Right / Brake  (급우회피 및 감속 궤적)
    """
    anchors = np.zeros((5, chunk_size, 2), dtype=np.float32)
    lateral_targets = [-3.4, -1.6, 0.0, +1.6, +3.4]
    lon_step = [1.30, 1.38, 1.42, 1.38, 1.25]

    for m in range(5):
        for k in range(chunk_size):
            tau = (k + 1) / float(chunk_size)
            # 부드러운 3차 다항식 형태의 횡방향 전이 프로파일
            smooth_tau = 3.0 * (tau**2) - 2.0 * (tau**3)
            anchors[m, k, 0] = (k + 1) * lon_step[m]
            anchors[m, k, 1] = lateral_targets[m] * smooth_tau
    return anchors


ANCHORS_NP = create_spatial_anchors(15)
MODE_NAMES = [
    "M0: Sharp-Left",
    "M1: Gentle-Left",
    "M2: Center-Keep",
    "M3: Gentle-Right",
    "M4: Sharp-Right",
]


def synthesize_bev_occupancy_gt(
    ego_tf: carla.Transform,
    spline: CarlaRouteSpline,
    obstacle_s_d_list: list[tuple[float, float]],
    grid_size: int = 64,
) -> np.ndarray:
    """
    차량 로컬 좌표계 기준 Top-Down BEV Semantic Occupancy Grid (3ch RGB-encoded, 64x64) 생성:
      - 커버 영역: 전방 x_local in [0.0m, 36.0m], 좌우 y_local in [-8.0m, +8.0m]
      - Channel 0 (Red)  : 장애물 차량 Footprint 점유 확률 (Obstacle Occupancy)
      - Channel 1 (Green): 주행 가능 도로 영역 (Drivable Road Corridor)
      - Channel 2 (Blue) : 차선 중심선 및 가이드라인 (Lane Centerline & Markings)
    추론 시 신경망(BEV Decoder)은 오직 전방 카메라 RGB 이미지만 보고 이 64x64 Top-Down BEV 맵을 스스로 예측합니다!
    """
    bev = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
    s_ego, d_ego, ref_yaw = spline.world_to_frenet(ego_tf.location)
    yaw_diff_rad = math.radians((ego_tf.rotation.yaw - ref_yaw + 180.0) % 360.0 - 180.0)

    # 그리드 좌표 생성 (row 0 = 전방 36m, row 63 = 전방 0m / col 0 = 좌측 -8m, col 63 = 우측 +8m)
    xs = np.linspace(36.0, 0.5, grid_size, dtype=np.float32)[:, None]  # [64, 1]
    ys = np.linspace(-8.0, 8.0, grid_size, dtype=np.float32)[None, :]  # [1, 64]

    # 로컬 (x, y) -> 도로 Frenet (s_grid, d_grid) 근사 변환
    cos_y = math.cos(yaw_diff_rad)
    sin_y = math.sin(yaw_diff_rad)
    s_grid = s_ego + xs * cos_y - ys * sin_y
    d_grid = d_ego + xs * sin_y + ys * cos_y

    # 1. 주행 가능 도로 영역 (d in [-5.2m, +5.2m]) -> Green Channel
    road_mask = (np.abs(d_grid) <= 5.2).astype(np.float32)
    bev[:, :, 1] = 0.35 * road_mask

    # 2. 3개 차선 중심선 (d = -3.2m, 0.0m, +3.2m) 및 이상적 목표 경로 -> Blue Channel
    for lane_d in [-3.2, 0.0, +3.2]:
        lane_line = np.exp(-((d_grid - lane_d) ** 2) / (2.0 * (0.28**2)))
        bev[:, :, 2] = np.maximum(bev[:, :, 2], 0.75 * lane_line * road_mask)

    # 3. 전방 장애물 차량 9대의 Top-Down 직사각형/가우시안 Footprint -> Red Channel
    for obs_s, obs_d in obstacle_s_d_list:
        if obs_s < s_ego - 2.0 or obs_s > s_ego + 40.0:
            continue
        footprint = np.exp(
            -((s_grid - obs_s) ** 2) / (2.0 * (2.35**2))
            - ((d_grid - obs_d) ** 2) / (2.0 * (1.05**2))
        )
        bev[:, :, 0] = np.maximum(bev[:, :, 0], np.clip(footprint * 1.35, 0.0, 1.0))
        # 장애물이 있는 곳은 주행 가능 녹색 채널 감쇠
        bev[:, :, 1] = np.clip(bev[:, :, 1] * (1.0 - 0.85 * footprint), 0.0, 1.0)

    return np.clip(bev, 0.0, 1.0)


def compute_candidate_modes_and_hydra_scores(
    ego_tf: carla.Transform,
    spline: CarlaRouteSpline,
    expert_local_wps: np.ndarray,
    obstacle_s_d_list: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    전문가 시연 데이터 생성 시:
      1) M=5 멀티모달 후보 궤적 (당첨 모드는 전문가 회피 궤적으로 정밀 수렴하고, 나머지 모드는 각 방향 대안 궤적 형성)
      2) Hydra-MDP 다중 지표 안전도 점수 S_m in [0, 1] (장애물 충돌 여부 NC + 도로 준수 DAC + 목표 일치도 EP)
    를 계산합니다.
    """
    s_ego, d_ego, ref_yaw = spline.world_to_frenet(ego_tf.location)
    yaw_diff_rad = math.radians((ego_tf.rotation.yaw - ref_yaw + 180.0) % 360.0 - 180.0)
    cos_y, sin_y = math.cos(yaw_diff_rad), math.sin(yaw_diff_rad)

    # 전문가 궤적의 끝점 횡방향 변위(expert_local_wps[-1, 1])와 가장 가까운 앵커를 최적 모드 m*로 선정
    end_y = float(expert_local_wps[-1, 1])
    anchor_end_ys = ANCHORS_NP[:, -1, 1]
    best_m = int(np.argmin(np.abs(anchor_end_ys - end_y)))

    cand_wps = np.zeros_like(ANCHORS_NP)  # [5, 15, 2]
    scores = np.zeros(5, dtype=np.float32)

    for m in range(5):
        if m == best_m:
            cand_wps[m] = expert_local_wps
        else:
            # 다른 모드들은 고유 앵커 방향을 유지하되 종방향 속도는 전문가와 맞춤
            cand_wps[m, :, 0] = 0.7 * expert_local_wps[:, 0] + 0.3 * ANCHORS_NP[m, :, 0]
            cand_wps[m, :, 1] = ANCHORS_NP[m, :, 1] + 0.20 * (expert_local_wps[:, 1] - ANCHORS_NP[best_m, :, 1])

        # Hydra-MDP 채점: (1) 장애물 충돌 페널티 (NC), (2) 도로 이탈 페널티 (DAC), (3) 목표 경로 적합도 (EP)
        min_obs_clearance = 99.0
        max_road_violation = 0.0
        for k in range(15):
            lx, ly = float(cand_wps[m, k, 0]), float(cand_wps[m, k, 1])
            s_pt = s_ego + lx * cos_y - ly * sin_y
            d_pt = d_ego + lx * sin_y + ly * cos_y

            if abs(d_pt) > 4.6:
                max_road_violation = max(max_road_violation, abs(d_pt) - 4.6)

            for obs_s, obs_d in obstacle_s_d_list:
                # 차량 크기 고려 타원형 안전거리 계산
                norm_dist = math.sqrt(((s_pt - obs_s) / 3.8) ** 2 + ((d_pt - obs_d) / 1.65) ** 2)
                min_obs_clearance = min(min_obs_clearance, norm_dist)

        # 충돌 점수 (1.0 미만이면 충돌 위험 급증)
        nc_score = 1.0 / (1.0 + math.exp(-5.0 * (min_obs_clearance - 0.95)))
        dac_score = math.exp(-0.8 * max_road_violation)
        dev_from_expert = float(np.mean(np.abs(cand_wps[m, :, 1] - expert_local_wps[:, 1])))
        ep_score = math.exp(-0.55 * dev_from_expert)

        raw_score = nc_score * dac_score * (0.35 + 0.65 * ep_score)
        if m == best_m:
            raw_score = max(raw_score, 0.92)
        else:
            raw_score = min(raw_score, 0.78)
        scores[m] = float(np.clip(raw_score, 0.03, 0.98))

    return cand_wps, scores, best_m


def expert_gauntlet_controller(
    ego_vehicle: carla.Vehicle,
    spline: CarlaRouteSpline,
    SingleModeDefect: bool = False,
) -> tuple[carla.VehicleControl, np.ndarray]:
    """
    210m 6단계 코스를 주행하는 전문가 제어기:
      - SOTA 학습용(SingleModeDefect=False): 6단계를 모두 완벽히 회피하고 s=190m에서 정밀 정차
      - 2세대 단일 모드 L2 비교용: Stage 3 갈림길(s=72~102m)에서 좌/우 회피 모드가 평균화(Mode Averaging)되어
        중앙 함정 장애물(Obs#5) 쪽으로 궤적이 쏠리는 현상 재현
    """
    ego_tf = ego_vehicle.get_transform()
    vel = ego_vehicle.get_velocity()
    speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    s, d, _ = spline.world_to_frenet(ego_tf.location)

    # 전방 4.2m 도로 접선 각도를 참조하여 곡선 도로(Corner)에서도 외측 연석을 밟지 않고 매끄럽게 선회
    ref_yaw_ahead = spline.get_tangent_yaw_at_s(s + 4.2)
    yaw_diff_deg = (ego_tf.rotation.yaw - ref_yaw_ahead + 180.0) % 360.0 - 180.0
    yaw_err = math.radians(yaw_diff_deg)

    lookahead_s = s + 5.0
    d_target_ahead = gauntlet_target_lateral_d(lookahead_s)

    if SingleModeDefect:
        # 2세대 단일 모드(L2 회귀)가 좌/우 다중 모드 갈림길(Stage 3, Stage 4)에서 겪는 모드 평균화(Mode Averaging) 현상
        if 70.0 <= lookahead_s <= 102.0:
            d_target_ahead *= 0.25  # 좌(-3.25m)/우(+1.8m) 회피 모드의 산술 평균으로 인해 중앙(d~-0.8m)으로 쏠림!
        elif 112.0 <= lookahead_s <= 136.0:
            d_target_ahead += 0.85 * math.sin(0.32 * lookahead_s)

    d_err = d_target_ahead - d
    desired_yaw = np.clip(0.28 * d_err, -0.55, 0.55)
    steer = float(np.clip(1.50 * (desired_yaw - yaw_err), -0.75, 0.75))

    # Stage 6 (s >= 176m): 전방 s=198m 바리케이드 앞에서 정밀 감속 및 정지 (목표 정지선 s=189.0m)
    stop_line_s = 189.0
    dist_to_stop = stop_line_s - s
    if dist_to_stop <= 1.5:
        throttle = 0.0
        brake = 1.0
        steer = 0.0
    elif dist_to_stop <= 14.0:
        target_speed_ms = max(0.0, 0.62 * dist_to_stop)
        if speed_ms > target_speed_ms + 0.3:
            throttle = 0.0
            brake = float(np.clip(0.20 * (speed_ms - target_speed_ms), 0.15, 0.95))
        else:
            throttle = 0.25
            brake = 0.0
    else:
        target_speed_ms = 9.6  # 약 34.5 km/h
        if speed_ms < target_speed_ms:
            throttle = float(np.clip(0.62 + 0.16 * (target_speed_ms - speed_ms), 0.35, 0.85))
            brake = 0.0
        else:
            throttle = 0.26
            brake = 0.0

    # 차량 로컬 좌표계 기준 미래 K=15 웨이포인트 산출
    local_wps = []
    ego_fwd = ego_tf.get_forward_vector()
    ego_right = ego_tf.get_right_vector()
    step_scale = float(np.clip(dist_to_stop / 18.0, 0.12, 1.0)) if dist_to_stop < 18.0 else 1.0

    for k in range(1, 16):
        s_fut = min(stop_line_s, s + k * 1.35 * step_scale)
        d_fut = gauntlet_target_lateral_d(s_fut)
        if SingleModeDefect and 70.0 <= s_fut <= 102.0:
            d_fut *= 0.25
        wp_tf = spline.frenet_to_world(s_fut, d_fut)
        wx, wy = wp_tf.location.x, wp_tf.location.y

        rel_x = (wx - ego_tf.location.x) * ego_fwd.x + (wy - ego_tf.location.y) * ego_fwd.y
        rel_y = (wx - ego_tf.location.x) * ego_right.x + (wy - ego_tf.location.y) * ego_right.y
        local_wps.append([max(0.8, float(rel_x)), float(rel_y)])

    ctrl = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
    return ctrl, np.array(local_wps, dtype=np.float32)


# ==============================================================================
# 4. 신경망 모델 정의: 2세대 Single-Mode E2E vs. 3세대 SOTA DiffusionDrive + Hydra-MDP
# ==============================================================================
class Gen2_SingleMode_L2_E2E(nn.Module):
    """
    [비교군: 2세대 단일 모드 플래닝 E2E (UniAD / VAD 단일 L2 회귀 스타일)]
    - 오직 1개의 평균 궤적(Single Trajectory)만 L2 손실로 회귀 학습합니다.
    - 다중 갈림길(Stage 3 Bifurcation)이나 좁은 협로(Stage 4 Gate)에서 좌/우 회피 모드가 충돌할 때
      두 모드의 산술 평균(Mode Averaging)을 출력하여 장애물 근접 스침 및 조향 진동이 발생합니다.
    """
    def __init__(self, d_model: int = 256, chunk_size: int = 15):
        super().__init__()
        self.chunk_size = chunk_size
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
        )
        self.bev_decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.state_proj = nn.Linear(5, d_model)
        self.plan_head = nn.Sequential(
            nn.Linear(d_model * 17, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, chunk_size * 2),
        )
        self.ctrl_head = nn.Sequential(
            nn.Linear(d_model * 2 + chunk_size * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, chunk_size * 3),
        )

    def forward(
        self, rgb_128: torch.Tensor, state5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = rgb_128.shape[0]
        feat = self.enc(rgb_128)
        pred_bev = self.bev_decoder(feat)
        v_flat = self.pool(feat).flatten(1)
        s_emb = self.state_proj(state5)
        pred_wps_flat = self.plan_head(torch.cat([v_flat, s_emb], dim=-1))
        pred_wps = pred_wps_flat.view(b, self.chunk_size, 2)
        v_mean = feat.mean(dim=(2, 3))
        pred_ctrls = self.ctrl_head(
            torch.cat([v_mean, s_emb, pred_wps_flat / 15.0], dim=-1)
        ).view(b, self.chunk_size, 3)
        return pred_wps, pred_ctrls, pred_bev


class SOTA_DiffusionDrive_HydraMDP(nn.Module):
    """
    [2025-2026 SOTA E2E 아키텍처: DiffusionDrive + Hydra-MDP + Top-Down BEV Occupancy]
    - 추론 시 입력: 전방 Raw RGB 카메라(3x128x128) + 차량 상태 및 진행 단계(5차원)
    - 내부 4단계 파이프라인:
      1) Perspective-to-BEV Spatial Lifter & BEV Occupancy Decoder (64x64 Top-Down BEV 예측, L_bev)
      2) Multi-Modal Truncated Flow-Matching Generator (M=5 앵커에서 출발하여 4-Step Euler ODE로 5개 후보 궤적 생성, L_flow)
      3) Hydra-MDP Neural Trajectory Scorer (5개 후보 궤적 각각의 무충돌/안전/진행 점수 S_m in [0, 1] 예측, L_score)
      4) Winner-Conditioned Control Chunk Head (최고 점수 궤적 tau_{m*}를 추종하는 K=15 제어 청크 출력, L_ctrl)
    """
    def __init__(self, d_model: int = 256, num_modes: int = 5, chunk_size: int = 15):
        super().__init__()
        self.d_model = d_model
        self.num_modes = num_modes
        self.chunk_size = chunk_size

        # 고정 공간 앵커 [1, 5, 15, 2] 등록
        self.register_buffer("anchors", torch.tensor(ANCHORS_NP, dtype=torch.float32).unsqueeze(0))

        # 1. Perspective RGB Backbone -> BEV Latent Feature Lifter
        self.rgb_enc = nn.Sequential(
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

        # Perspective 8x8 -> Top-Down BEV 8x8 Spatial Cross-Projection
        self.persp_to_bev = nn.Sequential(
            nn.Flatten(2),                          # [B, 256, 64]
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Unflatten(2, (8, 8)),                # [B, 256, 8, 8] Top-Down BEV Latent Grid
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        # 2. Top-Down BEV Semantic Occupancy Decoder (8x8 -> 64x64: [Obstacle, Road, Lane])
        self.bev_decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),       # 32x32
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),         # 64x64
            nn.Sigmoid(),
        )

        # 3. Scene Context Transformer (BEV Tokens + Ego State)
        self.bev_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.bev_pos = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)
        self.state_proj = nn.Linear(5, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=512, dropout=0.05, activation="gelu", batch_first=True
        )
        self.scene_transformer = nn.TransformerEncoder(enc_layer, num_layers=3)

        # 4. DiffusionDrive Truncated Flow-Matching Velocity Field Network v_theta(tau_t, t, m, context)
        self.mode_emb = nn.Embedding(num_modes, d_model)
        self.traj_in_proj = nn.Linear(chunk_size * 2, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.flow_vel_net = nn.Sequential(
            nn.Linear(d_model * 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, chunk_size * 2),
        )

        # 5. Hydra-MDP Neural Multi-Metric Scorer Head (각 후보 궤적 + BEV 문맥 -> 안전도 점수 S_m in [0, 1])
        self.hydra_scorer = nn.Sequential(
            nn.Linear(d_model * 3 + chunk_size * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # 6. Trajectory-Conditioned System-1 Control Chunk Head
        self.ctrl_head = nn.Sequential(
            nn.Linear(d_model * 2 + chunk_size * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, chunk_size * 3),
        )

    def encode_scene(self, rgb_128: torch.Tensor, state5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        persp_feat = self.rgb_enc(rgb_128)
        bev_feat = self.persp_to_bev(persp_feat) + persp_feat  # Residual Perspective-to-BEV Lifting
        pred_bev_64 = self.bev_decoder(bev_feat)

        b_tok = self.bev_pool(bev_feat).flatten(2).transpose(1, 2) + self.bev_pos  # [B, 16, 256]
        s_tok = self.state_proj(state5).unsqueeze(1)                               # [B, 1, 256]
        fused = self.scene_transformer(torch.cat([s_tok, b_tok], dim=1))
        scene_ctx = torch.cat([fused[:, 0], fused[:, 1:].mean(dim=1)], dim=-1)     # [B, 512]
        return scene_ctx, pred_bev_64

    def predict_flow_velocity(
        self, tau_t: torch.Tensor, t_scalar: torch.Tensor, scene_ctx: torch.Tensor
    ) -> torch.Tensor:
        """
        tau_t: [B, M=5, K=15, 2]
        t_scalar: [B, 1] in [0, 1]
        scene_ctx: [B, 512]
        반환: 속도장 v_theta [B, M=5, K=15, 2]
        """
        b, m, k, _ = tau_t.shape
        tau_flat = (tau_t / 15.0).view(b, m, k * 2)
        traj_feat = self.traj_in_proj(tau_flat)                                   # [B, M, 256]
        mode_ids = torch.arange(m, device=tau_t.device).unsqueeze(0).expand(b, m)
        m_feat = self.mode_emb(mode_ids)                                          # [B, M, 256]
        t_feat = self.time_mlp(t_scalar).unsqueeze(1).expand(b, m, self.d_model)  # [B, M, 256]
        ctx_exp = scene_ctx.unsqueeze(1).expand(b, m, self.d_model * 2)           # [B, M, 512]

        inp = torch.cat([ctx_exp, traj_feat + m_feat, t_feat], dim=-1)            # [B, M, 1024]
        vel_flat = self.flow_vel_net(inp)                                         # [B, M, 30]
        return vel_flat.view(b, m, k, 2)

    def forward_inference(
        self, rgb_128: torch.Tensor, state5: torch.Tensor, ode_steps: int = 4
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        추론(Inference) 시:
          1) M=5 공간 앵커에서 출발하여 4-Step Truncated Euler ODE 적분으로 5개 후보 궤적 생성
          2) Hydra-MDP Scorer로 5개 궤적 각각의 안전도/진행도 점수 S_m 산출
          3) 최고 점수 궤적(m* = argmax S_m)을 선택하여 제어 청크 생성!
        """
        b = rgb_128.shape[0]
        scene_ctx, pred_bev_64 = self.encode_scene(rgb_128, state5)

        # 1. M=5 멀티모달 앵커 초기화 후 4-Step Euler ODE Flow-Matching 디노이징
        tau = self.anchors.expand(b, -1, -1, -1).clone()  # [B, 5, 15, 2]
        dt = 1.0 / float(ode_steps)
        for step_i in range(ode_steps):
            t_val = torch.full((b, 1), step_i * dt, device=rgb_128.device, dtype=torch.float32)
            v_pred = self.predict_flow_velocity(tau, t_val, scene_ctx)
            tau = tau + dt * v_pred

        # 2. Hydra-MDP Neural Trajectory Scorer
        m = self.num_modes
        k = self.chunk_size
        tau_flat = (tau / 15.0).view(b, m, k * 2)
        mode_ids = torch.arange(m, device=rgb_128.device).unsqueeze(0).expand(b, m)
        m_feat = self.mode_emb(mode_ids)
        ctx_exp = scene_ctx.unsqueeze(1).expand(b, m, self.d_model * 2)
        score_logits = self.hydra_scorer(torch.cat([ctx_exp, m_feat, tau_flat], dim=-1)).squeeze(-1)  # [B, 5]
        hydra_scores = torch.sigmoid(score_logits)                                                    # [B, 5]

        # 3. 최고 안전/진행 점수 모드(m*) 선택 및 제어 청크 생성
        best_mode_idx = torch.argmax(hydra_scores, dim=-1)                                            # [B]
        batch_indices = torch.arange(b, device=rgb_128.device)
        best_wps = tau[batch_indices, best_mode_idx]                                                  # [B, 15, 2]
        best_wps_flat = best_wps.view(b, k * 2) / 15.0
        pred_ctrls = self.ctrl_head(torch.cat([scene_ctx, best_wps_flat], dim=-1)).view(b, k, 3)

        return best_wps, pred_ctrls, tau, hydra_scores, pred_bev_64


# ==============================================================================
# 5. SOTA 시각화 렌더러: 5개 멀티모달 궤적 투영 + Top-Down BEV 레이더 + Hydra-MDP 차트
# ==============================================================================
def render_bev_radar_with_trajectories(
    bev_rgb_64: np.ndarray,
    cand_trajs: np.ndarray | None,
    hydra_scores: np.ndarray | None,
    best_mode: int,
    single_wps: np.ndarray | None = None,
    out_w: int = 190,
    out_h: int = 190,
) -> np.ndarray:
    """
    신경망이 자체 예측한 64x64 Top-Down BEV Occupancy Grid 위에
    자차(Ego) 아이콘과 M=5 멀티모달 후보 궤적(또는 단일 궤적)을 오버레이하여 레이더 미니맵으로 렌더링합니다.
    """
    # 보기 좋은 다크 레이더 스타일 컬러맵으로 변환
    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
    canvas[:, :] = (24, 28, 36)  # 배경 암청색

    obs_ch = bev_rgb_64[:, :, 0]
    road_ch = bev_rgb_64[:, :, 1]
    lane_ch = bev_rgb_64[:, :, 2]

    # 도로 영역 (어두운 슬레이트 그린/그레이)
    road_mask = road_ch > 0.15
    canvas[road_mask] = (46, 58, 52)

    # 차선 중심선 (청록색 가이드라인)
    lane_mask = lane_ch > 0.30
    canvas[lane_mask] = (80, 145, 165)

    # 장애물 Footprint (강렬한 네온 레드/오렌지 히트맵)
    for r in range(64):
        for c in range(64):
            ov = float(obs_ch[r, c])
            if ov > 0.22:
                alpha = min(1.0, ov * 1.2)
                canvas[r, c] = (
                    int((1 - alpha) * canvas[r, c, 0] + alpha * 255),
                    int((1 - alpha) * canvas[r, c, 1] + alpha * 55),
                    int((1 - alpha) * canvas[r, c, 2] + alpha * 65),
                )

    radar = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    def local_xy_to_radar_px(lx: float, ly: float) -> tuple[int, int]:
        # x in [0, 36m] -> row in [out_h-10, 6], y in [-8, +8m] -> col in [6, out_w-6]
        u = int(np.clip((ly + 8.0) / 16.0 * (out_w - 12) + 6, 2, out_w - 3))
        v = int(np.clip((1.0 - lx / 36.0) * (out_h - 16) + 6, 2, out_h - 3))
        return u, v

    # 후보 궤적 렌더링
    if cand_trajs is not None and hydra_scores is not None:
        # 비당첨 모드 먼저 그리기
        for m in range(cand_trajs.shape[0]):
            if m == best_mode:
                continue
            sc = float(hydra_scores[m])
            pts = [local_xy_to_radar_px(0.0, 0.0)]
            for k in range(cand_trajs.shape[1]):
                pts.append(local_xy_to_radar_px(float(cand_trajs[m, k, 0]), float(cand_trajs[m, k, 1])))
            color = (255, 95, 95) if sc < 0.48 else (110, 200, 255)
            for i in range(len(pts) - 1):
                cv2.line(radar, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)

        # 당첨 최고점 모드(best_mode) 굵은 에메랄드 라인으로 그리기
        pts_best = [local_xy_to_radar_px(0.0, 0.0)]
        for k in range(cand_trajs.shape[1]):
            pts_best.append(
                local_xy_to_radar_px(float(cand_trajs[best_mode, k, 0]), float(cand_trajs[best_mode, k, 1]))
            )
        for i in range(len(pts_best) - 1):
            cv2.line(radar, pts_best[i], pts_best[i + 1], (52, 211, 153), 3, cv2.LINE_AA)
    elif single_wps is not None:
        pts_single = [local_xy_to_radar_px(0.0, 0.0)]
        for k in range(single_wps.shape[0]):
            pts_single.append(local_xy_to_radar_px(float(single_wps[k, 0]), float(single_wps[k, 1])))
        for i in range(len(pts_single) - 1):
            cv2.line(radar, pts_single[i], pts_single[i + 1], (251, 191, 36), 2, cv2.LINE_AA)

    # 하단 중앙 자차(Ego Vehicle) 표시
    ego_u, ego_v = local_xy_to_radar_px(0.0, 0.0)
    cv2.rectangle(radar, (ego_u - 5, ego_v - 8), (ego_u + 5, ego_v + 4), (255, 255, 255), -1)
    cv2.rectangle(radar, (ego_u - 5, ego_v - 8), (ego_u + 5, ego_v + 4), (56, 189, 248), 1)
    return radar


def render_sota_comparison_panel(
    rgb_img: np.ndarray,
    chase_img: np.ndarray,
    pred_bev_64: np.ndarray,
    ego_vehicle: carla.Vehicle,
    cam_front: carla.Sensor,
    K: np.ndarray,
    best_wps: np.ndarray,
    cand_trajs: np.ndarray | None,
    hydra_scores: np.ndarray | None,
    best_mode: int,
    title: str,
    subtitle: str,
    stage_banner: str,
    accent_color: tuple[int, int, int],
    step: int,
    total_steps: int,
    s_m: float,
    d_err_m: float,
    speed_kmh: float,
    ctrl: carla.VehicleControl,
    min_obs_dist_m: float,
) -> np.ndarray:
    """
    좌측(2세대 단일 모드 L2 E2E) 및 우측(3세대 SOTA DiffusionDrive + Hydra-MDP E2E) 패널을 렌더링합니다.
    """
    front_canvas = rgb_img.copy()
    ego_tf = ego_vehicle.get_transform()
    cam_tf = cam_front.get_transform()
    ego_fwd = ego_tf.get_forward_vector()
    ego_right = ego_tf.get_right_vector()

    def local_wps_to_world_locs(wps_2d: np.ndarray, z_add: float = 0.08) -> list[carla.Location]:
        locs = []
        for k in range(wps_2d.shape[0]):
            lx, ly = float(wps_2d[k, 0]), float(wps_2d[k, 1])
            if lx < 1.0:
                continue
            wx = ego_tf.location.x + ego_fwd.x * lx + ego_right.x * ly
            wy = ego_tf.location.y + ego_fwd.y * lx + ego_right.y * ly
            locs.append(carla.Location(x=wx, y=wy, z=ego_tf.location.z + z_add))
        return locs

    # 1. SOTA 모델인 경우: 비당첨 4개 후보 궤적을 전면 유리에 먼저 투영 (충돌 위험 궤적은 붉은 점선!)
    if cand_trajs is not None and hydra_scores is not None:
        for m in range(cand_trajs.shape[0]):
            if m == best_mode:
                continue
            sc = float(hydra_scores[m])
            c_locs = local_wps_to_world_locs(cand_trajs[m], z_add=0.06)
            c_px = project_world_to_image(c_locs, cam_tf, K)
            if len(c_px) >= 2:
                line_col = (255, 85, 85) if sc < 0.48 else (100, 210, 255)
                for i in range(len(c_px) - 1):
                    # 점선 효과: 짝수 세그먼트만 강조
                    if sc < 0.48 and i % 2 == 1:
                        continue
                    cv2.line(front_canvas, c_px[i], c_px[i + 1], line_col, 2, cv2.LINE_AA)
                end_pt = c_px[-1]
                cv2.circle(front_canvas, end_pt, 4, line_col, -1, cv2.LINE_AA)
                cv2.putText(
                    front_canvas,
                    f"M{m}:{int(sc*100)}%",
                    (end_pt[0] - 18, end_pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    line_col,
                    1,
                    cv2.LINE_AA,
                )

    # 2. 채택된 주행 궤적(best_wps)을 3D FSD 카펫으로 전면 유리에 투영
    left_3d, right_3d, center_3d = [], [], []
    for k in range(best_wps.shape[0]):
        lx, ly = float(best_wps[k, 0]), float(best_wps[k, 1])
        if lx < 1.0:
            continue
        wx = ego_tf.location.x + ego_fwd.x * lx + ego_right.x * ly
        wy = ego_tf.location.y + ego_fwd.y * lx + ego_right.y * ly
        wz = ego_tf.location.z + 0.08
        half_w = 0.80
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
        cv2.addWeighted(overlay, 0.44, front_canvas, 0.56, 0, front_canvas)
        for i in range(len(center_px) - 1):
            cv2.line(front_canvas, center_px[i], center_px[i + 1], (255, 255, 255), 2, cv2.LINE_AA)
        for pt in center_px[::2]:
            cv2.circle(front_canvas, pt, 4, accent_color, -1, cv2.LINE_AA)

    # 3. 우측 상단 PIP: 신경망이 자체 예측한 Top-Down BEV Semantic Occupancy Radar (152x152)
    radar_img = render_bev_radar_with_trajectories(
        pred_bev_64,
        cand_trajs=cand_trajs,
        hydra_scores=hydra_scores,
        best_mode=best_mode,
        single_wps=best_wps if cand_trajs is None else None,
        out_w=148,
        out_h=148,
    )
    rx0, ry0 = front_canvas.shape[1] - 160, 52
    front_canvas[ry0 : ry0 + 148, rx0 : rx0 + 148] = radar_img
    cv2.rectangle(front_canvas, (rx0, ry0), (rx0 + 148, ry0 + 148), accent_color, 2)
    cv2.rectangle(front_canvas, (rx0, ry0 - 18), (rx0 + 148, ry0), (15, 23, 42), -1)
    cv2.putText(
        front_canvas,
        "Predicted BEV Radar",
        (rx0 + 6, ry0 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # 4. SOTA 모델인 경우 좌측 하단에 Hydra-MDP 5-Mode Safety Score 미니 바 차트 표시
    if hydra_scores is not None:
        bx0, by0 = 10, front_canvas.shape[0] - 102
        sub = front_canvas[by0 : by0 + 94, bx0 : bx0 + 195]
        black = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.22, black, 0.78, 0, sub)
        front_canvas[by0 : by0 + 94, bx0 : bx0 + 195] = sub
        cv2.rectangle(front_canvas, (bx0, by0), (bx0 + 195, by0 + 94), (100, 116, 139), 1)
        cv2.putText(
            front_canvas,
            "Hydra-MDP 5-Mode Scorer",
            (bx0 + 6, by0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (56, 189, 248),
            1,
            cv2.LINE_AA,
        )
        for m in range(5):
            sc = float(hydra_scores[m])
            y_row = by0 + 28 + m * 13
            is_win = m == best_mode
            lbl_col = (52, 211, 153) if is_win else ((248, 113, 113) if sc < 0.48 else (203, 213, 225))
            cv2.putText(
                front_canvas,
                f"M{m}{'*' if is_win else ' '}",
                (bx0 + 6, y_row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                lbl_col,
                1,
                cv2.LINE_AA,
            )
            bar_w = int(np.clip(sc * 105, 3, 105))
            cv2.rectangle(front_canvas, (bx0 + 36, y_row - 8), (bx0 + 36 + bar_w, y_row - 1), lbl_col, -1)
            cv2.putText(
                front_canvas,
                f"{int(sc*100)}%",
                (bx0 + 146, y_row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                lbl_col,
                1,
                cv2.LINE_AA,
            )
    else:
        # 2세대 단일 모드 경고 배지
        cv2.rectangle(front_canvas, (10, front_canvas.shape[0] - 32), (315, front_canvas.shape[0] - 8), (20, 20, 30), -1)
        cv2.putText(
            front_canvas,
            "Single-Mode L2 Plan (Vulnerable to Mode-Averaging)",
            (16, front_canvas.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (251, 191, 36),
            1,
            cv2.LINE_AA,
        )

    # 5. 상단 타이틀 & 현재 Stage VLM Subgoal 배너
    banner_h = 46
    sub = front_canvas[0:banner_h, 0 : front_canvas.shape[1]]
    black_rect = np.zeros_like(sub)
    cv2.addWeighted(sub, 0.22, black_rect, 0.78, 0, sub)
    front_canvas[0:banner_h, 0 : front_canvas.shape[1]] = sub

    cv2.putText(front_canvas, title, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, accent_color, 2, cv2.LINE_AA)
    cv2.putText(front_canvas, subtitle, (12, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (226, 232, 240), 1, cv2.LINE_AA)

    # 6. 하단 3인칭 Chase View + 210m 장거리 진행 게이지 & 텔레메트리 바
    chase_canvas = chase_img.copy()
    bar_h = 62
    h_c, w_c, _ = chase_canvas.shape
    sub2 = chase_canvas[h_c - bar_h : h_c, 0:w_c]
    black_rect2 = np.zeros_like(sub2)
    cv2.addWeighted(sub2, 0.22, black_rect2, 0.78, 0, sub2)
    chase_canvas[h_c - bar_h : h_c, 0:w_c] = sub2

    cv2.putText(
        chase_canvas,
        stage_banner,
        (12, h_c - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.41,
        (56, 189, 248),
        1,
        cv2.LINE_AA,
    )
    tele_str = (
        f"Progress s:{s_m:5.1f}/205m | Spd:{speed_kmh:4.1f}km/h | "
        f"Steer:{ctrl.steer:+.2f} | LatErr:{d_err_m:.2f}m | MinClear:{min_obs_dist_m:.2f}m"
    )
    cv2.putText(
        chase_canvas,
        tele_str,
        (12, h_c - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (248, 250, 252),
        1,
        cv2.LINE_AA,
    )

    # 진행률 프로그레스 바 (하단 끝 4px)
    prog_ratio = float(np.clip(s_m / 195.0, 0.0, 1.0))
    cv2.rectangle(chase_canvas, (0, h_c - 5), (int(w_c * prog_ratio), h_c), accent_color, -1)

    combined = np.vstack([front_canvas, chase_canvas])
    cv2.rectangle(combined, (0, 0), (combined.shape[1] - 1, combined.shape[0] - 1), accent_color, 2)
    return combined


# ==============================================================================
# 6. 메인 파이프라인: CARLA 210m 6단계 데이터 수집 -> SOTA 학습 -> 폐루프 장거리 주행 비교
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print("🚀 [Demo 13] 2025-2026 SOTA E2E Autonomous Driving in CARLA 0.9.15 (RTX 4090)")
    print("   Architecture: DiffusionDrive (M=5 Flow-Matching) + Hydra-MDP Scorer + BEV Occupancy")
    print("   Scenario    : 210m 6-Stage Long-Horizon Urban Gauntlet (9 Obstacle Vehicles)")
    print("=" * 88)

    client, world, host_ip = connect_carla_client(port=args.port)
    print(f"   ✔️ CARLA 접속 성공 ({host_ip}:{args.port}, Map: {world.get_map().name})")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 15.0  # 15 Hz
    world.apply_settings(settings)

    weather = carla.WeatherParameters.ClearNoon
    weather.sun_altitude_angle = 55.0
    world.set_weather(weather)

    carla_map = world.get_map()
    spline = find_best_long_gauntlet_route(carla_map, target_len_m=215.0)
    start_tf = spline.frenet_to_world(0.0, 0.0, z_offset=0.25)
    print(
        f"✅ 215m 장거리 도심 주행 경로(CarlaRouteSpline) 생성 완료! "
        f"(총 길이: {spline.total_length:.1f}m, 시작점: x={start_tf.location.x:.1f}, y={start_tf.location.y:.1f})"
    )

    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.find("vehicle.tesla.model3")
    ego_bp.set_attribute("color", "25,35,55")

    spawned_actors = []
    img_w, img_h, fov = 560, 256, 95.0
    K = build_camera_intrinsic(img_w, img_h, fov)

    try:
        ego_vehicle = world.spawn_actor(ego_bp, start_tf)
        spawned_actors.append(ego_vehicle)
        ego_vehicle.set_autopilot(False)

        # 9대의 장애물 차량을 210m 6단계 코스에 스폰
        obstacle_s_d_list = []
        for idx_o, (obs_s, obs_d, bp_name, role_desc) in enumerate(GAUNTLET_OBSTACLES):
            obs_bp = bp_lib.find(bp_name)
            if obs_bp.has_attribute("color") and "tesla" in bp_name:
                obs_bp.set_attribute("color", "210,50,50")
            obs_tf = spline.frenet_to_world(obs_s, obs_d, z_offset=0.25)
            obs_actor = world.try_spawn_actor(obs_bp, obs_tf)
            if obs_actor is not None:
                spawned_actors.append(obs_actor)
                obs_actor.set_autopilot(False)
                obs_actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                obstacle_s_d_list.append((obs_s, obs_d))
                print(f"   🚓 [{idx_o+1}/9] {role_desc} 배치 완료 (s={obs_s:.1f}m, d={obs_d:+.2f}m)")

        # 카메라 센서 2개만 장착 (전방 RGB + 3인칭 Chase View / 추론 시 LiDAR나 Seg 센서 불필요!)
        q_rgb, q_chase = queue.Queue(), queue.Queue()

        bp_rgb = bp_lib.find("sensor.camera.rgb")
        bp_rgb.set_attribute("image_size_x", str(img_w))
        bp_rgb.set_attribute("image_size_y", str(img_h))
        bp_rgb.set_attribute("fov", str(fov))
        cam_front = world.spawn_actor(
            bp_rgb,
            carla.Transform(carla.Location(x=1.5, z=1.45), carla.Rotation(pitch=-4.0)),
            attach_to=ego_vehicle,
        )
        spawned_actors.append(cam_front)
        cam_front.listen(q_rgb.put)

        bp_chase = bp_lib.find("sensor.camera.rgb")
        bp_chase.set_attribute("image_size_x", str(img_w))
        bp_chase.set_attribute("image_size_y", str(img_h))
        bp_chase.set_attribute("fov", "85.0")
        cam_chase = world.spawn_actor(
            bp_chase,
            carla.Transform(carla.Location(x=-6.8, z=3.5), carla.Rotation(pitch=-14.0)),
            attach_to=ego_vehicle,
        )
        spawned_actors.append(cam_chase)
        cam_chase.listen(q_chase.put)

        def fetch_rgb_img(q: queue.Queue, frame_id: int) -> np.ndarray:
            while True:
                data = q.get(timeout=5.0)
                if data.frame == frame_id:
                    arr = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((data.height, data.width, 4))
                    return arr[:, :, :3][:, :, ::-1].copy()

        def reset_ego(y_offset: float = 0.0):
            tf = spline.frenet_to_world(0.0, y_offset, z_offset=0.25)
            ego_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            ego_vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            ego_vehicle.set_transform(tf)
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            for _ in range(10):
                fid = world.tick()
                _ = fetch_rgb_img(q_rgb, fid)
                _ = fetch_rgb_img(q_chase, fid)
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))

        # ==========================================================================
        # Phase 1: 210m 6단계 장거리 전문가 주행 데이터 수집 (SOTA용 멀티모달 + 2세대용 단일모드)
        # ==========================================================================
        print("\n📦 [Phase 1/3] 210m 6단계 도심 복합 장애물 코스 전문가 시연 수집 중 (3개 에피소드 x 370 스텝)...")
        dataset_rgb = []
        dataset_state = []
        dataset_bev_gt = []
        dataset_cand_wps = []
        dataset_hydra_scores = []
        dataset_expert_wps = []
        dataset_expert_ctrl = []

        # 2세대 단일 모드 비교용 데이터셋 (동일 3개 에피소드에서 Stage 3 갈림길 좌/우 모드 평균화 현상 반영)
        gen2_rgb, gen2_state, gen2_wps, gen2_ctrl, gen2_bev = [], [], [], [], []

        episode_Steps = 370
        for ep_idx, y_off in enumerate([0.0, -0.35, +0.35]):
            reset_ego(y_off)
            ep_records = []
            ep_g2_records = []
            for step in range(episode_Steps):
                fid = world.tick()
                rgb_img = fetch_rgb_img(q_rgb, fid)
                _ = fetch_rgb_img(q_chase, fid)

                ego_tf = ego_vehicle.get_transform()
                vel = ego_vehicle.get_velocity()
                speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                s, d, ref_yaw = spline.world_to_frenet(ego_tf.location)
                yaw_diff_deg = (ego_tf.rotation.yaw - ref_yaw + 180.0) % 360.0 - 180.0
                yaw_err = math.radians(yaw_diff_deg)
                stage_id, _, _ = get_stage_info(s)

                ctrl, exp_wps = expert_gauntlet_controller(ego_vehicle, spline, SingleModeDefect=False)
                ctrl_g2, exp_wps_g2 = expert_gauntlet_controller(ego_vehicle, spline, SingleModeDefect=True)
                ego_vehicle.apply_control(ctrl)

                bev_gt = synthesize_bev_occupancy_gt(ego_tf, spline, obstacle_s_d_list, grid_size=64)
                cand_wps, h_scores, _ = compute_candidate_modes_and_hydra_scores(
                    ego_tf, spline, exp_wps, obstacle_s_d_list
                )

                rgb_128 = cv2.resize(rgb_img, (128, 128)).transpose(2, 0, 1).astype(np.float32) / 255.0
                state_vec = np.array(
                    [speed_ms / 10.0, yaw_err, s / 200.0, d / 4.0, stage_id / 6.0], dtype=np.float32
                )
                ctrl_vec = np.array([ctrl.throttle, ctrl.steer, ctrl.brake], dtype=np.float32)
                ctrl_vec_g2 = np.array([ctrl_g2.throttle, ctrl_g2.steer, ctrl_g2.brake], dtype=np.float32)

                ep_records.append((rgb_128, state_vec, bev_gt.transpose(2, 0, 1), cand_wps, h_scores, exp_wps, ctrl_vec))
                ep_g2_records.append((rgb_128, state_vec, exp_wps_g2, ctrl_vec_g2, bev_gt.transpose(2, 0, 1)))

            for i in range(len(ep_records)):
                r_128, st_v, b_gt, c_wps, h_sc, e_wps, _ = ep_records[i]
                ctrl_chunk = [ep_records[min(i + k, len(ep_records) - 1)][6] for k in range(15)]
                dataset_rgb.append(r_128)
                dataset_state.append(st_v)
                dataset_bev_gt.append(b_gt)
                dataset_cand_wps.append(c_wps)
                dataset_hydra_scores.append(h_sc)
                dataset_expert_wps.append(e_wps)
                dataset_expert_ctrl.append(np.array(ctrl_chunk, dtype=np.float32))

                _, _, e_wps_g2, _, _ = ep_g2_records[i]
                ctrl_chunk_g2 = [ep_g2_records[min(i + k, len(ep_g2_records) - 1)][3] for k in range(15)]
                gen2_rgb.append(r_128)
                gen2_state.append(st_v)
                gen2_wps.append(e_wps_g2)
                gen2_ctrl.append(np.array(ctrl_chunk_g2, dtype=np.float32))
                gen2_bev.append(b_gt)

            print(f"   ✔️ 전문가 에피소드 #{ep_idx+1}/3 완료 (최종 주행거리 s={s:.1f}m, 최종 속도={speed_ms*3.6:.1f}km/h)")

        print(f"   ✔️ 총 수집 샘플: SOTA {len(dataset_rgb)}개 / Gen2 {len(gen2_rgb)}개")

        # ==========================================================================
        # Phase 2: RTX 4090에서 2세대 Single-Mode vs 3세대 SOTA DiffusionDrive+Hydra-MDP 학습
        # ==========================================================================
        print("\n🧠 [Phase 2/3] RTX 4090에서 2세대 Single-Mode E2E vs 3세대 SOTA DiffusionDrive+Hydra-MDP 학습 중...")
        x_rgb = torch.tensor(np.array(dataset_rgb), dtype=torch.float32, device=device)
        x_state = torch.tensor(np.array(dataset_state), dtype=torch.float32, device=device)
        y_bev = torch.tensor(np.array(dataset_bev_gt), dtype=torch.float32, device=device)
        y_cand_wps = torch.tensor(np.array(dataset_cand_wps), dtype=torch.float32, device=device)
        y_hydra = torch.tensor(np.array(dataset_hydra_scores), dtype=torch.float32, device=device)
        y_exp_wps = torch.tensor(np.array(dataset_expert_wps), dtype=torch.float32, device=device)
        y_exp_ctrl = torch.tensor(np.array(dataset_expert_ctrl), dtype=torch.float32, device=device)

        g2_x_rgb = torch.tensor(np.array(gen2_rgb), dtype=torch.float32, device=device)
        g2_x_state = torch.tensor(np.array(gen2_state), dtype=torch.float32, device=device)
        g2_y_wps = torch.tensor(np.array(gen2_wps), dtype=torch.float32, device=device)
        g2_y_ctrl = torch.tensor(np.array(gen2_ctrl), dtype=torch.float32, device=device)
        g2_y_bev = torch.tensor(np.array(gen2_bev), dtype=torch.float32, device=device)

        model_gen2 = Gen2_SingleMode_L2_E2E(d_model=256, chunk_size=15).to(device)
        model_sota = SOTA_DiffusionDrive_HydraMDP(d_model=256, num_modes=5, chunk_size=15).to(device)

        opt_gen2 = torch.optim.AdamW(model_gen2.parameters(), lr=1e-3, weight_decay=1e-4)
        opt_sota = torch.optim.AdamW(model_sota.parameters(), lr=1e-3, weight_decay=1e-4)

        t0 = time.time()
        n_sota = x_rgb.shape[0]
        n_g2 = g2_x_rgb.shape[0]
        batch_size = 64

        for epoch in range(1, 56):
            # 1) Gen2 Single-Mode L2 학습
            perm_g2 = torch.randperm(n_g2, device=device)
            for b_i in range(0, n_g2, batch_size):
                idx = perm_g2[b_i : b_i + batch_size]
                p_wps, p_ctrls, p_bev = model_gen2(g2_x_rgb[idx], g2_x_state[idx])
                loss_g2 = (
                    F.mse_loss(p_wps, g2_y_wps[idx])
                    + F.mse_loss(p_ctrls, g2_y_ctrl[idx])
                    + 0.5 * F.mse_loss(p_bev, g2_y_bev[idx])
                )
                opt_gen2.zero_grad()
                loss_g2.backward()
                opt_gen2.step()

            # 2) SOTA DiffusionDrive + Hydra-MDP + BEV Occupancy 멀티태스크 학습
            perm_s = torch.randperm(n_sota, device=device)
            for b_i in range(0, n_sota, batch_size):
                idx = perm_s[b_i : b_i + batch_size]
                b_sz = idx.shape[0]

                b_rgb = x_rgb[idx]
                b_st = x_state[idx]
                b_bev_gt = y_bev[idx]
                b_cand_gt = y_cand_wps[idx]     # [B, 5, 15, 2]
                b_scores_gt = y_hydra[idx]      # [B, 5]
                b_exp_wps = y_exp_wps[idx]      # [B, 15, 2]
                b_exp_ctrl = y_exp_ctrl[idx]    # [B, 15, 3]

                scene_ctx, pred_bev = model_sota.encode_scene(b_rgb, b_st)

                # (a) Truncated Flow-Matching 손실 L_flow: 앵커 tau_0 -> 목표 후보 궤적 tau_1 사이의 선형 보간 속도장 회귀
                tau_0 = model_sota.anchors.expand(b_sz, -1, -1, -1)
                tau_1 = b_cand_gt
                t_rand = torch.rand((b_sz, 1), device=device)
                t_4d = t_rand.view(b_sz, 1, 1, 1)
                tau_t = (1.0 - t_4d) * tau_0 + t_4d * tau_1
                target_vel = tau_1 - tau_0
                pred_vel = model_sota.predict_flow_velocity(tau_t, t_rand, scene_ctx)
                loss_flow = F.mse_loss(pred_vel, target_vel)

                # (b) Hydra-MDP Scorer 손실 L_score: 후보 궤적의 무충돌/진행 점수 예측
                tau_flat = (tau_1 / 15.0).view(b_sz, 5, 30)
                mode_ids = torch.arange(5, device=device).unsqueeze(0).expand(b_sz, 5)
                m_feat = model_sota.mode_emb(mode_ids)
                ctx_exp = scene_ctx.unsqueeze(1).expand(b_sz, 5, 512)
                score_logits = model_sota.hydra_scorer(torch.cat([ctx_exp, m_feat, tau_flat], dim=-1)).squeeze(-1)
                pred_scores = torch.sigmoid(score_logits)
                loss_score = F.mse_loss(pred_scores, b_scores_gt)

                # (c) BEV Occupancy 복원 손실 L_bev + 제어 청크 손실 L_ctrl
                loss_bev = F.mse_loss(pred_bev, b_bev_gt)
                exp_wps_flat = b_exp_wps.view(b_sz, 30) / 15.0
                pred_ctrls = model_sota.ctrl_head(torch.cat([scene_ctx, exp_wps_flat], dim=-1)).view(b_sz, 15, 3)
                loss_ctrl = F.mse_loss(pred_ctrls, b_exp_ctrl)

                total_sota_loss = loss_flow + 2.0 * loss_score + 1.5 * loss_ctrl + 0.8 * loss_bev
                opt_sota.zero_grad()
                total_sota_loss.backward()
                opt_sota.step()

            if epoch in (1, 15, 30, 45, 55):
                print(
                    f"   [Epoch {epoch:02d}/55] "
                    f"Gen2(Single-L2) Loss: {loss_g2.item():.5f} | "
                    f"SOTA Loss: {total_sota_loss.item():.5f} "
                    f"(Flow:{loss_flow.item():.4f}, HydraScore:{loss_score.item():.4f}, "
                    f"Ctrl:{loss_ctrl.item():.4f}, BEV:{loss_bev.item():.4f})"
                )

        print(f"✅ 두 모델 학습 완료! ({time.time() - t0:.1f}초 소요)")
        model_gen2.eval()
        model_sota.eval()

        # ==========================================================================
        # Phase 3: 210m 6-Stage 장거리 폐루프(Closed-Loop) 주행 평가 (340 스텝 = 22.7초)
        # ==========================================================================
        def run_closed_loop_gauntlet(is_sota: bool) -> tuple[list[np.ndarray], dict]:
            reset_ego(0.0)
            frames = []
            telemetry = {
                "s": [],
                "d": [],
                "d_ref": [],
                "d_err": [],
                "steer": [],
                "speed_kmh": [],
                "min_clearance": [],
                "hydra_scores": [],
                "best_mode": [],
            }
            chunk_history = []

            title = (
                "3rd-Gen SOTA E2E (DiffusionDrive + Hydra-MDP + BEV)"
                if is_sota
                else "2nd-Gen Modular E2E (Single-Mode L2 Trajectory Planner)"
            )
            subtitle = (
                "M=5 Truncated Flow-Matching + Neural Safety Scorer + Predicted BEV Radar"
                if is_sota
                else "Single Average Trajectory -> Vulnerable to Mode-Averaging at Forks"
            )
            accent = (52, 211, 153) if is_sota else (251, 191, 36)

            for step in range(episode_Steps):
                fid = world.tick()
                rgb_img = fetch_rgb_img(q_rgb, fid)
                chase_img = fetch_rgb_img(q_chase, fid)

                ego_tf = ego_vehicle.get_transform()
                vel = ego_vehicle.get_velocity()
                speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                speed_kmh = speed_ms * 3.6
                s, d, ref_yaw = spline.world_to_frenet(ego_tf.location)
                yaw_diff_deg = (ego_tf.rotation.yaw - ref_yaw + 180.0) % 360.0 - 180.0
                yaw_err = math.radians(yaw_diff_deg)
                stage_id, stage_banner, _ = get_stage_info(s)

                d_ref = gauntlet_target_lateral_d(s)
                d_err = abs(d - d_ref)

                # 가장 가까운 장애물과의 여유 거리(Clearance [m]) 계산
                min_clear = min(
                    math.sqrt((s - os) ** 2 + (d - od) ** 2) for os, od in obstacle_s_d_list
                )

                rgb_128 = cv2.resize(rgb_img, (128, 128)).transpose(2, 0, 1).astype(np.float32) / 255.0
                state_vec = np.array(
                    [speed_ms / 10.0, yaw_err, s / 200.0, d / 4.0, stage_id / 6.0], dtype=np.float32
                )

                t_rgb = torch.tensor(rgb_128[None], dtype=torch.float32, device=device)
                t_st = torch.tensor(state_vec[None], dtype=torch.float32, device=device)

                with torch.no_grad():
                    if is_sota:
                        best_wps_t, pred_ctrls_t, cand_trajs_t, h_scores_t, pred_bev_t = model_sota.forward_inference(
                            t_rgb, t_st, ode_steps=4
                        )
                        best_wps = best_wps_t[0].cpu().numpy()
                        pred_chunk = pred_ctrls_t[0].cpu().numpy()
                        cand_trajs = cand_trajs_t[0].cpu().numpy()
                        h_scores = h_scores_t[0].cpu().numpy()
                        best_m = int(np.argmax(h_scores))
                        pred_bev_64 = pred_bev_t[0].cpu().numpy().transpose(1, 2, 0)
                    else:
                        best_wps_t, pred_ctrls_t, pred_bev_t = model_gen2(t_rgb, t_st)
                        best_wps = best_wps_t[0].cpu().numpy()
                        pred_chunk = pred_ctrls_t[0].cpu().numpy()
                        cand_trajs = None
                        h_scores = None
                        best_m = 2
                        pred_bev_64 = pred_bev_t[0].cpu().numpy().transpose(1, 2, 0)

                # Temporal Ensembling 적용
                chunk_history.append(pred_chunk)
                if len(chunk_history) > 6:
                    chunk_history.pop(0)
                act_accum = np.zeros(3, dtype=np.float32)
                w_sum = 0.0
                for age_idx, past_ch in enumerate(reversed(chunk_history)):
                    w = math.exp(-0.22 * age_idx)
                    act_accum += w * past_ch[age_idx]
                    w_sum += w
                act = act_accum / w_sum

                ctrl = carla.VehicleControl(
                    throttle=float(np.clip(act[0], 0.0, 0.85)),
                    steer=float(np.clip(act[1], -0.75, 0.75)),
                    brake=float(np.clip(act[2], 0.0, 1.0)) if act[2] > 0.12 else 0.0,
                )
                ego_vehicle.apply_control(ctrl)

                panel = render_sota_comparison_panel(
                    rgb_img=rgb_img,
                    chase_img=chase_img,
                    pred_bev_64=pred_bev_64,
                    ego_vehicle=ego_vehicle,
                    cam_front=cam_front,
                    K=K,
                    best_wps=best_wps,
                    cand_trajs=cand_trajs,
                    hydra_scores=h_scores,
                    best_mode=best_m,
                    title=title,
                    subtitle=subtitle,
                    stage_banner=stage_banner,
                    accent_color=accent,
                    step=step + 1,
                    total_steps=episode_Steps,
                    s_m=s,
                    d_err_m=d_err,
                    speed_kmh=speed_kmh,
                    ctrl=ctrl,
                    min_obs_dist_m=min_clear,
                )
                frames.append(panel)

                telemetry["s"].append(s)
                telemetry["d"].append(d)
                telemetry["d_ref"].append(d_ref)
                telemetry["d_err"].append(d_err)
                telemetry["steer"].append(ctrl.steer)
                telemetry["speed_kmh"].append(speed_kmh)
                telemetry["min_clearance"].append(min_clear)
                if h_scores is not None:
                    telemetry["hydra_scores"].append(h_scores)
                    telemetry["best_mode"].append(best_m)

            return frames, telemetry

        print("\n🎬 [Phase 3/3] 210m 6-Stage 장거리 폐루프(Closed-Loop) 주행 비교 실행 중 (각 340 스텝)...")
        frames_g2, tele_g2 = run_closed_loop_gauntlet(is_sota=False)
        print(
            f"   🟡 [2세대 Single-Mode L2] 주행 거리: s={tele_g2['s'][-1]:.1f}m | "
            f"평균 오차: {np.mean(tele_g2['d_err']):.2f}m | 최소 장애물 간격: {np.min(tele_g2['min_clearance']):.2f}m | "
            f"최종 속도: {tele_g2['speed_kmh'][-1]:.1f}km/h"
        )

        frames_sota, tele_sota = run_closed_loop_gauntlet(is_sota=True)
        print(
            f"   🟢 [3세대 SOTA DiffusionDrive+Hydra-MDP] 주행 거리: s={tele_sota['s'][-1]:.1f}m | "
            f"평균 오차: {np.mean(tele_sota['d_err']):.2f}m | 최소 장애물 간격: {np.min(tele_sota['min_clearance']):.2f}m | "
            f"최종 속도: {tele_sota['speed_kmh'][-1]:.1f}km/h"
        )

        # Side-by-Side MP4 동영상 및 Stage 3 갈림길(Mode-Averaging Trap) 하이라이트 스냅샷 저장
        mp4_path = out_dir / "carla_sota_diffusion_bev_e2e.mp4"
        snap_path = out_dir / "carla_sota_diffusion_bev_snapshot.png"
        chart_path = out_dir / "carla_sota_gauntlet_telemetry.png"

        combined_frames = []
        for idx in range(len(frames_sota)):
            comb = np.hstack([frames_g2[idx], frames_sota[idx]])
            combined_frames.append(comb)
            # Stage 3 갈림길(s ~ 82m, step ~ 135) 순간을 대표 스냅샷으로 저장
            if idx == 135:
                cv2.imwrite(str(snap_path), comb[:, :, ::-1])

        imageio.mimsave(str(mp4_path), combined_frames, fps=15)

        # 210m 6단계 장거리 텔레메트리 분석 그래프 생성
        fig, axes = plt.subplots(3, 1, figsize=(13, 10.5), dpi=130)
        fig.patch.set_facecolor("#0f172a")
        for ax in axes:
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#cbd5e1")
            for spine in ax.spines.values():
                spine.set_color("#475569")

        # 1) 210m 6-Stage 주행 궤적 비교 (s vs d) + 9대 장애물 위치 표시
        s_grid = np.linspace(0, 198, 400)
        d_ref_grid = [gauntlet_target_lateral_d(sv) for sv in s_grid]
        axes[0].plot(s_grid, d_ref_grid, "--", color="#94a3b8", linewidth=2.0, label="Ideal 6-Stage Reference Path")
        axes[0].plot(
            tele_g2["s"],
            tele_g2["d"],
            color="#fbbf24",
            linewidth=2.3,
            label=f"2nd-Gen Single-Mode L2 (Mean Err={np.mean(tele_g2['d_err']):.2f}m, Mode-Averaging at Stage 3)",
        )
        axes[0].plot(
            tele_sota["s"],
            tele_sota["d"],
            color="#34d399",
            linewidth=2.8,
            label=f"3rd-Gen SOTA DiffusionDrive+Hydra-MDP (Mean Err={np.mean(tele_sota['d_err']):.2f}m)",
        )
        for idx_o, (os, od, _, _) in enumerate(GAUNTLET_OBSTACLES):
            axes[0].scatter(
                [os],
                [od],
                color="#ef4444",
                s=110,
                marker="s",
                zorder=5,
                label="9 Obstacle Vehicles" if idx_o == 0 else None,
            )
        # Stage 구분선 표시
        for st_s, st_lbl in [(33, "S2"), (70, "S3:Fork"), (108, "S4:Gate"), (143, "S5:Slalom"), (174, "S6:Stop")]:
            axes[0].axvline(st_s, color="#475569", linestyle=":", linewidth=1.2)
            axes[0].text(st_s + 1.5, 3.8, st_lbl, color="#38bdf8", fontsize=8.5, fontweight="bold")

        axes[0].set_title(
            "210m 6-Stage Urban Gauntlet Trajectory (Longitudinal s [m] vs Lateral Offset d [m])",
            color="#f8fafc",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].set_xlabel("Longitudinal Distance s (m)", color="#cbd5e1")
        axes[0].set_ylabel("Lateral Offset d (m)", color="#cbd5e1")
        axes[0].set_ylim(-4.5, 4.5)
        axes[0].legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", loc="lower left", fontsize=8.5)
        axes[0].grid(True, color="#334155", linestyle=":", alpha=0.6)

        # 2) Hydra-MDP 5-Mode Safety Score 시계열 변화 (주행 단계에 따라 최적 모드가 동적으로 전환되는 과정)
        scores_arr = np.array(tele_sota["hydra_scores"])  # [T, 5]
        s_sota = tele_sota["s"]
        mode_colors = ["#34d399", "#60a5fa", "#cbd5e1", "#fbbf24", "#f87171"]
        for m in range(5):
            axes[1].plot(
                s_sota,
                scores_arr[:, m] * 100.0,
                color=mode_colors[m],
                linewidth=2.0 if m in (0, 2, 4) else 1.4,
                label=MODE_NAMES[m],
            )
        axes[1].set_title(
            "Hydra-MDP Real-Time Multi-Modal Trajectory Safety Scores S_m (%) Across 210m Gauntlet",
            color="#f8fafc",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].set_xlabel("Longitudinal Distance s (m)", color="#cbd5e1")
        axes[1].set_ylabel("Safety & Progress Score (%)", color="#cbd5e1")
        axes[1].set_ylim(0, 105)
        axes[1].legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", loc="lower left", ncol=5, fontsize=8.5)
        axes[1].grid(True, color="#334155", linestyle=":", alpha=0.6)

        # 3) 속도 프로파일 및 Stage 6 정밀 정차 (Speed [km/h] vs s [m])
        axes[2].plot(tele_g2["s"], tele_g2["speed_kmh"], color="#fbbf24", linewidth=2.0, label="2nd-Gen Speed (km/h)")
        axes[2].plot(
            tele_sota["s"],
            tele_sota["speed_kmh"],
            color="#34d399",
            linewidth=2.5,
            label="3rd-Gen SOTA Speed (km/h) - Smooth Cruise & Terminal Stop at s=190m",
        )
        axes[2].set_title(
            "Velocity Profile & Stage 6 Terminal Hazard Stop (Speed [km/h] vs Distance s [m])",
            color="#f8fafc",
            fontsize=12,
            fontweight="bold",
        )
        axes[2].set_xlabel("Longitudinal Distance s (m)", color="#cbd5e1")
        axes[2].set_ylabel("Speed (km/h)", color="#cbd5e1")
        axes[2].legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", loc="lower left")
        axes[2].grid(True, color="#334155", linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(str(chart_path), facecolor=fig.get_facecolor())
        plt.close(fig)

        print(f"\n🎉 저장 완료:")
        print(f"   - 210m 6단계 장거리 비교 영상 : {mp4_path}")
        print(f"   - Stage 3 갈림길 비교 스냅샷  : {snap_path}")
        print(f"   - 210m 6단계 텔레메트리 차트  : {chart_path}")

    finally:
        for actor in reversed(spawned_actors):
            if actor is not None and actor.is_alive:
                actor.destroy()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    main()
