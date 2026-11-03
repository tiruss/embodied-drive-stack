"""
Demo 10: True End-to-End VLA (Vision-Language-Action) in CARLA 0.9.15 (RTX 4090)
================================================================================
CARLA 내장 오토파일럿을 완전히 끄고(set_autopilot(False)),
RTX 4090에서 학습한 단일 멀티모달 VLA 신경망(CarlaEndToEndVLA)이
전방 실사 RGB 영상 + 시맨틱 세그멘테이션 + 차량 상태 + 자연어 운전 명령(Prompt)을 입력받아:
  1) 미래 K=15 스텝 3D 주행 웨이포인트 (카메라 영상 위 3D FSD 카펫으로 실시간 투영)
  2) 미래 K=15 스텝 차량 제어 청크 [throttle, steer, brake] (Temporal Ensembling 적용 직접 제어)
를 출력하여 Tesla Model 3를 직접 운전하는 실습입니다!

[동일한 단일 VLA 모델에 입력하는 3가지 자연어 명령]
  - Prompt #1: "change to left lane and overtake the stopped car"
               -> 좌측 차선으로 변경하며 정차된 경찰차를 추월 주행!
  - Prompt #2: "change to right lane and pass the obstacle safely"
               -> 우측 차선으로 변경하며 정차된 경찰차를 회피 주행!
  - Prompt #3: "brake smoothly and stop behind the police car"
               -> 차선을 유지한 채 감속하여 전방 경찰차 뒤에 안전하게 완전 정지(0 km/h)!
"""
import argparse
import math
import queue
import time
from pathlib import Path

import carla
import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
from demo_9_carla_urban_vla import build_camera_intrinsic, connect_carla_client, project_world_to_image

CARLA_PROMPTS = [
    "change to left lane and overtake the stopped car",
    "change to right lane and pass the obstacle safely",
    "brake smoothly and stop behind the police car",
]

VOCAB = {
    "<pad>": 0, "change": 1, "to": 2, "left": 3, "lane": 4, "and": 5,
    "overtake": 6, "the": 7, "stopped": 8, "car": 9, "right": 10,
    "pass": 11, "obstacle": 12, "safely": 13, "brake": 14, "smoothly": 15,
    "stop": 16, "behind": 17, "police": 18,
}


def tokenize_prompt(prompt: str, max_len: int = 10, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    words = prompt.lower().split()[:max_len]
    ids = [VOCAB.get(w, 0) for w in words] + [0] * max(0, max_len - len(words))
    return torch.tensor(ids[:max_len], dtype=torch.long, device=device)


def find_long_straight_spawn(carla_map: carla.Map) -> carla.Transform:
    """
    전방 75m 동안 직선이고 좌/우로 회피 공간이 충분한 도심 대로 스폰 포인트를 자동으로 탐색합니다.
    """
    best_sp = None
    best_score = -1.0

    for sp in carla_map.get_spawn_points():
        wp = carla_map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue

        # 전방 70m 웨이포인트가 모두 직선(곡률 최소)인지 확인
        nxt = wp.next(70.0)
        if not nxt:
            continue
        wp_far = nxt[0]
        fwd = sp.get_forward_vector()
        dx = wp_far.transform.location.x - sp.location.x
        dy = wp_far.transform.location.y - sp.location.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 65.0:
            continue

        alignment = (dx * fwd.x + dy * fwd.y) / dist
        # 좌/우 인접 차선 존재 여부 가산점
        side_bonus = 0.0
        if wp.get_left_lane() is not None:
            side_bonus += 0.2
        if wp.get_right_lane() is not None:
            side_bonus += 0.2

        score = alignment + side_bonus
        if score > best_score:
            best_score = score
            best_sp = sp

    return best_sp if best_sp is not None else carla_map.get_spawn_points()[0]


class CarlaEndToEndVLA(nn.Module):
    """
    CARLA End-to-End VLA 신경망:
    - 입력 1 (Vision): 전방 RGB 이미지(3ch) + 시맨틱 세그멘테이션 이미지(3ch) = 6채널 (128x128)
    - 입력 2 (Language): 자연어 운전 명령 토큰 시퀀스 (10 토큰)
    - 입력 3 (Ego State): 차량 속도, 조향각, 출발점 대비 전진 거리 s, 횡방향 오프셋 d (4차원)
    - 출력 1 (3D Waypoints): 차량 로컬 좌표계 기준 미래 K=15 스텝 웨이포인트 (15 x 2) [local_x, local_y]
    - 출력 2 (Vehicle Control): 미래 K=15 스텝 제어 명령 청크 (15 x 3) [throttle, steer, brake]
    """
    def __init__(self, vocab_size: int = 24, d_model: int = 256, chunk_size: int = 15):
        super().__init__()
        self.chunk_size = chunk_size

        # 6채널(RGB + Seg) 비전 백본 -> 4x4 = 16 패치 토큰
        self.vis_backbone = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=5, stride=2, padding=2),
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
        self.vis_pos = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)

        self.lang_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lang_pos = nn.Parameter(torch.randn(1, 10, d_model) * 0.02)

        self.state_proj = nn.Linear(4, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=512, dropout=0.05, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)

        self.head = nn.Sequential(
            nn.Linear(d_model * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )
        self.wp_out = nn.Linear(512, chunk_size * 2)
        self.ctrl_out = nn.Linear(512, chunk_size * 3)

    def forward(self, img6_128: torch.Tensor, tokens: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = img6_128.shape[0]
        v_tok = self.vis_backbone(img6_128).flatten(2).transpose(1, 2) + self.vis_pos
        l_raw = self.lang_emb(tokens)
        l_tok = l_raw + self.lang_pos
        s_tok = self.state_proj(state).unsqueeze(1)

        seq = torch.cat([s_tok, l_tok, v_tok], dim=1)
        fused = self.transformer(seq)

        pooled = torch.cat([fused[:, 0], fused[:, 1:11].mean(dim=1), l_raw.mean(dim=1)], dim=-1)
        h = self.head(pooled)
        wps = self.wp_out(h).view(b, self.chunk_size, 2)
        ctrls = self.ctrl_out(h).view(b, self.chunk_size, 3)
        return wps, ctrls


def compute_frenet_s_d(ego_loc: carla.Location, start_tf: carla.Transform) -> tuple[float, float, float]:
    """출발점 기준 전진 거리 s(m), 우측 횡방향 오프셋 d(m)를 계산합니다."""
    dx = ego_loc.x - start_tf.location.x
    dy = ego_loc.y - start_tf.location.y
    fwd = start_tf.get_forward_vector()
    right = start_tf.get_right_vector()
    s = dx * fwd.x + dy * fwd.y
    d = dx * right.x + dy * right.y
    return float(s), float(d), float(start_tf.rotation.yaw)


def expert_urban_controller(
    ego_vehicle: carla.Vehicle,
    start_tf: carla.Transform,
    cmd_idx: int,
) -> tuple[carla.VehicleControl, np.ndarray]:
    """
    전방 s=28m 지점에 정차된 경찰차를 기준으로 3가지 언어 명령에 맞는
    전문가 제어 입력(throttle, steer, brake)과 미래 15스텝 로컬 웨이포인트(15 x 2)를 생성합니다.
    """
    ego_tf = ego_vehicle.get_transform()
    vel = ego_vehicle.get_velocity()
    speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    s, d, ref_yaw = compute_frenet_s_d(ego_tf.location, start_tf)

    # 현재 차량의 기준 도로 대비 헤딩 오차 (rad)
    yaw_diff_deg = (ego_tf.rotation.yaw - ref_yaw + 180.0) % 360.0 - 180.0
    yaw_err = math.radians(yaw_diff_deg)

    def target_lateral_d(s_query: float) -> float:
        if cmd_idx == 0:
            # 명령 0: 좌측 차선(d = -3.4m)으로 부드럽게 변경해 추월!
            return -3.4 / (1.0 + math.exp(-0.32 * (s_query - 13.0)))
        elif cmd_idx == 1:
            # 명령 1: 우측 차선(d = +3.4m)으로 부드럽게 변경해 추월!
            return +3.4 / (1.0 + math.exp(-0.32 * (s_query - 13.0)))
        else:
            # 명령 2: 차선 유지 (d = 0.0m)
            return 0.0

    # 전방 4.5m 지점의 목표 횡방향 위치를 추종하는 Stanley/PD 조향 제어
    d_target_ahead = target_lateral_d(s + 4.5)
    d_err = d_target_ahead - d
    desired_yaw = np.clip(0.24 * d_err, -0.45, 0.45)
    steer = float(np.clip(1.35 * (desired_yaw - yaw_err), -0.65, 0.65))

    if cmd_idx in (0, 1):
        target_speed_ms = 8.5  # 약 31 km/h로 부드럽게 추월 주행
        if speed_ms < target_speed_ms:
            throttle = float(np.clip(0.55 + 0.15 * (target_speed_ms - speed_ms), 0.3, 0.78))
            brake = 0.0
        else:
            throttle = 0.2
            brake = 0.0
    else:
        # 명령 2: s = 20m 지점(전방 경찰차 s=28m의 8m 뒤)에 부드럽게 완전 정지!
        stop_s = 19.5
        dist_to_stop = stop_s - s
        if dist_to_stop > 6.5:
            target_v = 6.5
            throttle = float(np.clip(0.50 + 0.1 * (target_v - speed_ms), 0.2, 0.65))
            brake = 0.0
        elif dist_to_stop > 0.5:
            # 감속 구간
            throttle = 0.0
            brake = float(np.clip(0.35 + (6.5 - dist_to_stop) * 0.12, 0.25, 0.85))
        else:
            # 완전 정지 유지
            throttle = 0.0
            brake = 1.0
            steer = 0.0

    # 차량 로컬 좌표계 기준 미래 15스텝 웨이포인트 (local_x: 전방, local_y: 우측) 생성
    local_wps = []
    step_ds = 0.0 if (cmd_idx == 2 and s >= 18.5) else 1.4
    for k in range(1, 16):
        s_fut = min(20.5, s + k * 0.8) if cmd_idx == 2 else (s + k * step_ds)
        d_fut = target_lateral_d(s_fut)
        # 월드 좌표로 복원 후 현재 차량 로컬 좌표로 변환
        fwd = start_tf.get_forward_vector()
        right = start_tf.get_right_vector()
        wx = start_tf.location.x + fwd.x * s_fut + right.x * d_fut
        wy = start_tf.location.y + fwd.y * s_fut + right.y * d_fut

        ego_fwd = ego_tf.get_forward_vector()
        ego_right = ego_tf.get_right_vector()
        rel_x = (wx - ego_tf.location.x) * ego_fwd.x + (wy - ego_tf.location.y) * ego_fwd.y
        rel_y = (wx - ego_tf.location.x) * ego_right.x + (wy - ego_tf.location.y) * ego_right.y
        local_wps.append([max(1.2, float(rel_x)), float(rel_y)])

    ctrl = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
    return ctrl, np.array(local_wps, dtype=np.float32)


def draw_vla_predicted_carpet(
    rgb_img: np.ndarray,
    chase_img: np.ndarray,
    seg_img: np.ndarray,
    ego_vehicle: carla.Vehicle,
    cam_front: carla.Sensor,
    K: np.ndarray,
    pred_local_wps: np.ndarray,
    cmd_idx: int,
    prompt: str,
    step: int,
    total_steps: int,
    speed_kmh: float,
    ctrl: carla.VehicleControl,
) -> np.ndarray:
    """
    VLA 신경망이 예측한 미래 15스텝 로컬 웨이포인트(pred_local_wps)를
    카메라 2D 픽셀 공간으로 투영하여 전방 블랙박스 화면 위에 FSD 카펫으로 그리고,
    하단에 3인칭 Chase 뷰와 시맨틱 세그멘테이션 PIP를 결합한 패널을 반환합니다.
    """
    front_canvas = rgb_img.copy()
    ego_tf = ego_vehicle.get_transform()
    cam_tf = cam_front.get_transform()
    ego_fwd = ego_tf.get_forward_vector()
    ego_right = ego_tf.get_right_vector()

    colors = [
        (56, 189, 248),   # Prompt 1: Sky Blue (Left Lane Overtake)
        (74, 222, 128),   # Prompt 2: Emerald Green (Right Lane Overtake)
        (251, 146, 60),   # Prompt 3: Orange-Red (Smooth Brake & Stop)
    ]
    c = colors[cmd_idx]

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
        cv2.fillPoly(overlay, [poly], c)
        cv2.addWeighted(overlay, 0.45, front_canvas, 0.55, 0, front_canvas)
        for i in range(len(center_px) - 1):
            cv2.line(front_canvas, center_px[i], center_px[i + 1], (255, 255, 255), 2, cv2.LINE_AA)
        for pt in center_px[::2]:
            cv2.circle(front_canvas, pt, 4, c, -1, cv2.LINE_AA)

    # 우측 상단에 Semantic Segmentation PIP 합성
    seg_pip = cv2.resize(seg_img, (156, 96))
    cv2.rectangle(seg_pip, (0, 0), (155, 95), c, 2)
    cv2.putText(seg_pip, "Vision Seg Input", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    front_canvas[68:164, front_canvas.shape[1] - 166 : front_canvas.shape[1] - 10] = seg_pip

    # 상단 HUD 배너 (프롬프트 + 실시간 VLA 제어값)
    banner = front_canvas.copy()
    cv2.rectangle(banner, (0, 0), (front_canvas.shape[1], 60), (12, 18, 30), -1)
    cv2.addWeighted(banner, 0.82, front_canvas, 0.18, 0, front_canvas)

    cv2.putText(
        front_canvas,
        f"CARLA VLA #{cmd_idx + 1}: \"{prompt}\"",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        c,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        front_canvas,
        f"Speed: {speed_kmh:4.1f} km/h | Steer: {ctrl.steer:+.2f} | Throt: {ctrl.throttle:.2f} | Brake: {ctrl.brake:.2f}",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (226, 232, 240),
        1,
        cv2.LINE_AA,
    )

    # 하단에 3인칭 Unreal Engine Chase View 배치
    chase_canvas = chase_img.copy()
    cb = chase_canvas.copy()
    cv2.rectangle(cb, (0, 0), (chase_canvas.shape[1], 34), (12, 18, 30), -1)
    cv2.addWeighted(cb, 0.76, chase_canvas, 0.24, 0, chase_canvas)
    mode_names = ["LEFT OVERTAKE (VLA Controlled)", "RIGHT OVERTAKE (VLA Controlled)", "SAFE FULL STOP (VLA Controlled)"]
    cv2.putText(
        chase_canvas,
        f"3D Chase View: {mode_names[cmd_idx]} | Frame {step:03d}/{total_steps}",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        c,
        1,
        cv2.LINE_AA,
    )

    return np.concatenate([front_canvas, chase_canvas], axis=0)


def preprocess_rgb_seg_128(rgb: np.ndarray, seg: np.ndarray, device: torch.device) -> torch.Tensor:
    t_rgb = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    t_seg = torch.from_numpy(seg).float().permute(2, 0, 1) / 255.0
    cat6 = torch.cat([t_rgb, t_seg], dim=0).unsqueeze(0).to(device)
    return torch.nn.functional.interpolate(cat6, size=(128, 128), mode="bilinear", align_corners=False)


def main():
    parser = argparse.ArgumentParser(description="Train and Run End-to-End VLA in CARLA 0.9.15")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=160, help="에피소드당 주행 프레임 수 (160 frames @ 20Hz)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_size = 15

    print("=" * 78)
    print(f"[1/4] CARLA 서버(Port {args.port}) 접속 및 도심 장애물 시나리오 구성 중...")
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

        # 전방 28m 지점에 정차된 경찰차(Obstacle Vehicle) 스폰
        fwd = start_tf.get_forward_vector()
        obs_loc = carla.Location(
            x=start_tf.location.x + fwd.x * 28.0,
            y=start_tf.location.y + fwd.y * 28.0,
            z=start_tf.location.z + 0.2,
        )
        obs_tf = carla.Transform(obs_loc, start_tf.rotation)
        police_bps = bp_lib.filter("vehicle.dodge.charger_police*")
        obs_bp = police_bps[0] if police_bps else bp_lib.filter("vehicle.*")[0]
        obstacle_car = world.try_spawn_actor(obs_bp, obs_tf)
        if obstacle_car is not None:
            actors_to_destroy.append(obstacle_car)
            obstacle_car.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))

        # 내 자율주행차 (Tesla Model 3) 스폰
        ego_bps = bp_lib.filter("vehicle.tesla.model3")
        ego_bp = ego_bps[0] if ego_bps else bp_lib.filter("vehicle.lincoln.mkz_2020")[0]
        if ego_bp.has_attribute("color"):
            ego_bp.set_attribute("color", "20,175,245")
        ego_vehicle = world.spawn_actor(ego_bp, start_tf)
        actors_to_destroy.append(ego_vehicle)

        cam_w, cam_h = 448, 264
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
            rgb_bp, carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-14.0)), attach_to=ego_vehicle
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

        def reset_ego_pose():
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            ego_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            ego_vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            ego_vehicle.set_transform(start_tf)
            for _ in range(8):
                world.tick()
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
            world.tick()
            while not q_front.empty(): q_front.get()
            while not q_chase.empty(): q_chase.get()
            while not q_seg.empty(): q_seg.get()

        print("=" * 78)
        print("[2/4] CARLA에서 3가지 언어 프롬프트별 주행 시연 데이터 수집 중...")
        print("=" * 78)

        demo_imgs6, demo_tokens, demo_states, demo_wps, demo_ctrls = [], [], [], [], []

        for cmd_idx, prompt in enumerate(CARLA_PROMPTS):
            tok = tokenize_prompt(prompt)
            reset_ego_pose()

            ep_imgs6, ep_states, ep_wps, ep_ctrls = [], [], [], []
            for t in range(args.steps + chunk_size):
                world.tick()
                img_f = q_front.get(timeout=5.0)
                _ = q_chase.get(timeout=5.0)
                img_s = q_seg.get(timeout=5.0)

                vel = ego_vehicle.get_velocity()
                speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                s, d, _ = compute_frenet_s_d(ego_vehicle.get_location(), start_tf)
                curr_ctrl = ego_vehicle.get_control()

                ctrl, local_wps = expert_urban_controller(ego_vehicle, start_tf, cmd_idx)
                ego_vehicle.apply_control(ctrl)

                ep_imgs6.append(preprocess_rgb_seg_128(img_f, img_s, device).squeeze(0).cpu())
                ep_states.append(torch.tensor([speed_ms / 10.0, curr_ctrl.steer, s / 40.0, d / 4.0], dtype=torch.float32))
                ep_wps.append(torch.from_numpy(local_wps))
                ep_ctrls.append(torch.tensor([ctrl.throttle, ctrl.steer, ctrl.brake], dtype=torch.float32))

            ep_ctrl_stack = torch.stack(ep_ctrls)
            for t in range(args.steps):
                demo_imgs6.append(ep_imgs6[t])
                demo_tokens.append(tok)
                demo_states.append(ep_states[t])
                demo_wps.append(ep_wps[t])
                demo_ctrls.append(ep_ctrl_stack[t : t + chunk_size])

            s_end, d_end, _ = compute_frenet_s_d(ego_vehicle.get_location(), start_tf)
            print(f"  - Prompt #{cmd_idx + 1} 시연 완료 (최종 전진 s={s_end:.1f}m, 횡방향 오프셋 d={d_end:+.2f}m): \"{prompt}\"")

        imgs6_t = torch.stack(demo_imgs6).to(device)
        tokens_t = torch.stack(demo_tokens).to(device)
        states_t = torch.stack(demo_states).to(device)
        wps_t = torch.stack(demo_wps).to(device)
        ctrls_t = torch.stack(demo_ctrls).to(device)

        print("\n" + "=" * 78)
        print("[3/4] RTX 4090에서 CARLA End-to-End VLA 신경망 학습 시작!")
        print("      입력: [RGB + Seg 6ch 이미지, 언어 프롬프트, 차량 상태] -> 출력: [3D 웨이포인트 청크, 조향/페달 청크]")
        print("=" * 78)

        vla = CarlaEndToEndVLA(vocab_size=len(VOCAB) + 2, d_model=256, chunk_size=chunk_size).to(device)
        optimizer = torch.optim.AdamW(vla.parameters(), lr=1e-3, weight_decay=1e-4)
        epochs = 120
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loss_fn = nn.L1Loss()

        batch_size = 32
        num_samples = imgs6_t.shape[0]
        t0 = time.perf_counter()

        vla.train()
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(num_samples, device=device)
            ep_loss = 0.0
            steps = 0
            for i in range(0, num_samples, batch_size):
                idx = perm[i : i + batch_size]
                noisy_state = states_t[idx] + 0.015 * torch.randn_like(states_t[idx])
                pred_wp, pred_ctrl = vla(imgs6_t[idx], tokens_t[idx], noisy_state)
                loss = loss_fn(pred_wp / 15.0, wps_t[idx] / 15.0) + loss_fn(pred_ctrl, ctrls_t[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ep_loss += loss.item()
                steps += 1
            scheduler.step()

            if epoch % 30 == 0 or epoch == 1:
                print(f"  Epoch {epoch:03d}/{epochs} | CARLA VLA Loss: {ep_loss / steps:.4f} | 경과 시간: {time.perf_counter() - t0:.1f}s")

        print("\n" + "=" * 78)
        print("[4/4] 오토파일럿 OFF! 학습된 VLA 신경망으로만 3가지 언어 명령 실제 주행 평가 중...")
        print("=" * 78)

        vla.eval()
        panel_videos = []

        for cmd_idx, prompt in enumerate(CARLA_PROMPTS):
            tok_in = tokenize_prompt(prompt, device=device).unsqueeze(0)
            reset_ego_pose()
            frames = []
            all_time_ctrls = np.zeros((args.steps, args.steps + chunk_size, 3), dtype=np.float32)

            for t in range(args.steps):
                world.tick()
                img_f = q_front.get(timeout=5.0)
                img_c = q_chase.get(timeout=5.0)
                img_s = q_seg.get(timeout=5.0)

                vel = ego_vehicle.get_velocity()
                speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                speed_kmh = speed_ms * 3.6
                s, d, _ = compute_frenet_s_d(ego_vehicle.get_location(), start_tf)
                curr_ctrl = ego_vehicle.get_control()

                with torch.inference_mode():
                    img6_in = preprocess_rgb_seg_128(img_f, img_s, device)
                    st_in = torch.tensor([[speed_ms / 10.0, curr_ctrl.steer, s / 40.0, d / 4.0]], dtype=torch.float32, device=device)
                    pred_wp, pred_ctrl = vla(img6_in, tok_in, st_in)
                    wp_np = pred_wp.squeeze(0).cpu().numpy()
                    ctrl_chunk = pred_ctrl.squeeze(0).cpu().numpy()

                # Temporal Ensembling으로 부드러운 조향각 및 페달 명령 산출
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

                panel = draw_vla_predicted_carpet(
                    img_f, img_c, img_s, ego_vehicle, cam_front, K, wp_np, cmd_idx, prompt, t + 1, args.steps, speed_kmh, applied_ctrl
                )
                frames.append(panel)

            panel_videos.append(frames)
            s_end, d_end, _ = compute_frenet_s_d(ego_vehicle.get_location(), start_tf)
            vel_end = ego_vehicle.get_velocity()
            v_end_kmh = 3.6 * math.sqrt(vel_end.x**2 + vel_end.y**2 + vel_end.z**2)
            print(f"  ★ VLA 주행 #{cmd_idx + 1} 완료 -> 전진 s={s_end:.1f}m | 횡방향 d={d_end:+.2f}m | 종료 속도={v_end_kmh:.1f} km/h | \"{prompt}\"")

        # 3개 언어 명령 실행 화면을 가로(좌/중/우)로 결합
        combined_frames = [
            np.concatenate([panel_videos[0][t], panel_videos[1][t], panel_videos[2][t]], axis=1)
            for t in range(args.steps)
        ]

        out_dir = Path(__file__).resolve().parent / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = out_dir / "carla_vla_3commands.mp4"
        snap_path = out_dir / "carla_vla_3commands_snapshot.png"
        imageio.mimsave(str(mp4_path), combined_frames, fps=20)
        imageio.imwrite(str(snap_path), combined_frames[95])

        print("=" * 78)
        print(f"CARLA End-to-End VLA 3가지 언어 명령 비교 영상 저장 완료: {mp4_path}")
        print(f"CARLA End-to-End VLA 스냅샷 저장 완료                  : {snap_path}")
        print("=" * 78)

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
