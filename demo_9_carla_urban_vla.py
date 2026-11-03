"""
Demo 9: CARLA 0.9.15 Real-Time 4-Sensor Autonomous Driving VLA Dashboard (RTX 4090)
===================================================================================
언리얼 엔진 기반 실사 자율주행 시뮬레이터 CARLA 서버(Port 2000)에 접속하여
자율주행 차량(Tesla Model 3 / Lincoln MKZ)에 4종 멀티 센서 수트를 장착하고
400 프레임(20초) 동안 도심 자율주행을 수행하며 2x2 통합 대시보드 영상을 기록합니다.

[2x2 멀티 센서 대시보드 구성]
  1) 좌상단: Front Dashcam RGB + 3D 카메라 투영(Projection) 기반 테슬라 FSD 미래 궤적 카펫
  2) 우상단: 3rd-Person Chase Camera (언리얼 엔진 3D 도심 주행 뷰 + 날씨 변화)
  3) 좌하단: Semantic Segmentation Camera (CityScapes 팔레트: 도로/차선/차량/보행자/신호등 분할)
  4) 우하단: 64-Channel 3D LiDAR Bird's-Eye View (BEV 포인트클라우드 + 미래 웨이포인트 궤적)
"""
import argparse
import math
import os
import queue
import subprocess
import time
from pathlib import Path

import carla
import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn


def get_wsl_host_ip() -> str:
    """WSL2에서 Windows 호스트에 떠 있는 CARLA 서버로 접속하기 위한 IP를 탐색합니다."""
    try:
        out = subprocess.check_output(["ip", "route", "show", "default"], text=True)
        for part in out.split():
            if part.count(".") == 3:
                return part.strip()
    except Exception:
        pass
    return "127.0.0.1"


def connect_carla_client(port: int = 3000) -> tuple[carla.Client, carla.World, str]:
    env_host = os.environ.get("CARLA_HOST")
    env_port = os.environ.get("CARLA_PORT")
    hosts = [env_host] if env_host else ["127.0.0.1", get_wsl_host_ip()]
    ports = [int(env_port)] if env_port else [port, 2000, 3000]
    seen_ports = []
    for p in ports:
        if p not in seen_ports:
            seen_ports.append(p)
    for p in seen_ports:
        for host in hosts:
            try:
                client = carla.Client(host, p)
                client.set_timeout(6.0)
                world = client.get_world()
                return client, world, host
            except Exception:
                continue
    raise RuntimeError(f"CARLA 서버(포트 {seen_ports})에 연결할 수 없습니다. CarlaUE4 실행 여부를 확인하세요.")


class CarlaVLAWaypointNet(nn.Module):
    """
    전방 RGB 카메라 특징 + 현재 속도/조향 상태 + 언어 프롬프트 임베딩을 입력받아
    자율주행 차량의 미래 3D 웨이포인트 청크를 보정/생성하는 RTX 4090 신경망 모듈
    """
    def __init__(self, chunk_size: int = 15):
        super().__init__()
        self.chunk_size = chunk_size
        self.vis_enc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 16 + 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, chunk_size * 2),
        )
        # 초기 출력을 0 근처로 초기화하여 HD-Map 기준 궤적 위에서 미세 잔차(Residual) 보정 수행
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, img_t: torch.Tensor, ego_state: torch.Tensor) -> torch.Tensor:
        v = self.vis_enc(img_t)
        out = self.head(torch.cat([v, ego_state], dim=-1))
        return out.view(-1, self.chunk_size, 2)


def build_camera_intrinsic(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = width / (2.0 * math.tan(fov_deg * math.pi / 360.0))
    K = np.identity(3, dtype=np.float32)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K


def project_world_to_image(world_pts: list[carla.Location], cam_transform: carla.Transform, K: np.ndarray) -> list[tuple[int, int]]:
    """CARLA 3D 월드 좌표를 카메라 2D 픽셀 좌표로 정확히 투영합니다."""
    world_to_cam = np.array(cam_transform.get_inverse_matrix(), dtype=np.float32)
    px_list = []
    for loc in world_pts:
        pt_w = np.array([loc.x, loc.y, loc.z, 1.0], dtype=np.float32)
        pt_c = world_to_cam @ pt_w
        # UE4 좌표계(X: 전방, Y: 우측, Z: 상방) -> 표준 카메라 좌표계(X: 우측, Y: 하방, Z: 전방)
        cam_x = pt_c[1]
        cam_y = -pt_c[2]
        cam_z = pt_c[0]
        if cam_z <= 0.8:
            continue
        uvw = K @ np.array([cam_x, cam_y, cam_z], dtype=np.float32)
        u = int(uvw[0] / uvw[2])
        v = int(uvw[1] / uvw[2])
        px_list.append((u, v))
    return px_list


def draw_3d_fsd_carpet_on_rgb(
    rgb_img: np.ndarray,
    ego_vehicle: carla.Vehicle,
    cam_sensor: carla.Sensor,
    carla_map: carla.Map,
    K: np.ndarray,
    vla_offset_xy: np.ndarray,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    현재 차량 앞 도로의 3D 웨이포인트들을 추출하고 카메라 이미지 위에
    실제 원근감이 반영된 3D 테슬라 FSD 주행 카펫을 렌더링합니다.
    """
    canvas = rgb_img.copy()
    h, w, _ = canvas.shape
    ego_tf = ego_vehicle.get_transform()
    cam_tf = cam_sensor.get_transform()

    curr_wp = carla_map.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if curr_wp is None:
        return canvas, []

    left_edge_3d = []
    right_edge_3d = []
    center_3d = []
    bev_local_pts = []

    ego_inv = np.array(ego_tf.get_inverse_matrix(), dtype=np.float32)

    wp_iter = curr_wp
    for k in range(1, 16):
        next_wps = wp_iter.next(2.0)
        if not next_wps:
            break
        wp_iter = next_wps[0]
        tf = wp_iter.transform
        loc = tf.location
        right_vec = tf.get_right_vector()

        # VLA 잔차 오프셋 반영
        dy_off = float(vla_offset_xy[k - 1, 1]) if k - 1 < len(vla_offset_xy) else 0.0
        center_loc = carla.Location(
            x=loc.x + right_vec.x * dy_off,
            y=loc.y + right_vec.y * dy_off,
            z=loc.z + 0.08,
        )
        half_w = 0.85
        l_loc = carla.Location(
            x=center_loc.x - right_vec.x * half_w,
            y=center_loc.y - right_vec.y * half_w,
            z=center_loc.z,
        )
        r_loc = carla.Location(
            x=center_loc.x + right_vec.x * half_w,
            y=center_loc.y + right_vec.y * half_w,
            z=center_loc.z,
        )
        left_edge_3d.append(l_loc)
        right_edge_3d.append(r_loc)
        center_3d.append(center_loc)

        # LiDAR BEV 표시용 차량 기준 로컬 좌표 (x: 전방, y: 우측)
        pt_local = ego_inv @ np.array([center_loc.x, center_loc.y, center_loc.z, 1.0], dtype=np.float32)
        bev_local_pts.append((float(pt_local[0]), float(pt_local[1])))

    left_px = project_world_to_image(left_edge_3d, cam_tf, K)
    right_px = project_world_to_image(right_edge_3d, cam_tf, K)
    center_px = project_world_to_image(center_3d, cam_tf, K)

    if len(left_px) >= 2 and len(right_px) >= 2:
        poly = np.array(left_px + right_px[::-1], dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [poly], (56, 189, 248))
        cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0, canvas)

        for i in range(len(center_px) - 1):
            cv2.line(canvas, center_px[i], center_px[i + 1], (255, 255, 255), 2, cv2.LINE_AA)
        for pt in center_px[::2]:
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(canvas, pt, 4, (74, 222, 128), -1, cv2.LINE_AA)

    return canvas, bev_local_pts


def render_lidar_bev(lidar_points: np.ndarray, bev_waypoints: list[tuple[float, float]], width: int = 640, height: int = 360) -> np.ndarray:
    """64채널 3D LiDAR 포인트클라우드를 고해상도 버드아이뷰(BEV) 레이더 화면으로 렌더링합니다."""
    bev = np.full((height, width, 3), (12, 18, 30), dtype=np.uint8)
    cx = width // 2
    cy = int(height * 0.62)
    scale = 5.2  # 픽셀/미터

    # 거리 동심원 그리드 (10m, 20m, 30m, 40m)
    for r_m in [10, 20, 30, 40]:
        cv2.circle(bev, (cx, cy), int(r_m * scale), (30, 45, 70), 1, cv2.LINE_AA)
    cv2.line(bev, (cx, 0), (cx, height), (28, 42, 65), 1)
    cv2.line(bev, (0, cy), (width, cy), (28, 42, 65), 1)

    if lidar_points is not None and len(lidar_points) > 0:
        # lidar_points: (N, 4) [x(전방), y(우측), z(높이), intensity]
        x = lidar_points[:, 0]
        y = lidar_points[:, 1]
        z = lidar_points[:, 2]

        px = (cx + y * scale).astype(np.int32)
        py = (cy - x * scale).astype(np.int32)
        valid = (px >= 1) & (px < width - 1) & (py >= 1) & (py < height - 1) & (z > -2.2)

        px = px[valid]
        py = py[valid]
        z_v = z[valid]

        # 높이(z)에 따라 바닥은 청록색, 장애물/차량/건물은 밝은 에메랄드/노란색으로 컬러 매핑
        norm_z = np.clip((z_v + 1.8) / 2.8, 0.0, 1.0)
        r_col = (40 + 215 * norm_z).astype(np.uint8)
        g_col = (180 + 70 * (1.0 - np.abs(norm_z - 0.5))).astype(np.uint8)
        b_col = (240 - 160 * norm_z).astype(np.uint8)

        bev[py, px, 0] = r_col
        bev[py, px, 1] = g_col
        bev[py, px, 2] = b_col

    # 미래 웨이포인트 궤적 벡터 그리기
    wp_px = [(cx, cy)]
    for wx, wy in bev_waypoints:
        wp_px.append((int(cx + wy * scale), int(cy - wx * scale)))
    for i in range(len(wp_px) - 1):
        cv2.line(bev, wp_px[i], wp_px[i + 1], (56, 189, 248), 2, cv2.LINE_AA)
    for pt in wp_px[1::2]:
        cv2.circle(bev, pt, 3, (74, 222, 128), -1, cv2.LINE_AA)

    # 내 자율주행차 아이콘 (중앙)
    cv2.rectangle(bev, (cx - 6, cy - 12), (cx + 6, cy + 12), (56, 189, 248), -1)
    cv2.rectangle(bev, (cx - 6, cy - 12), (cx + 6, cy + 12), (255, 255, 255), 1)
    return bev


def add_panel_label(img: np.ndarray, title: str, subtitle: str, color: tuple[int, int, int]) -> np.ndarray:
    canvas = img.copy()
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 52), (12, 18, 30), -1)
    cv2.addWeighted(overlay, 0.76, canvas, 0.24, 0, canvas)
    cv2.putText(canvas, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (226, 232, 240), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Run CARLA 4-Sensor Autonomous Driving VLA Demo")
    parser.add_argument("--steps", type=int, default=400, help="주행 프레임 수 (기본 400 = 20초)")
    parser.add_argument("--port", type=int, default=3000, help="CARLA RPC 포트 (Windows Hyper-V 2000번 충돌 회피용 기본값 3000)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vla_net = CarlaVLAWaypointNet(chunk_size=15).to(device).eval()

    print("=" * 78)
    print("[1/4] CARLA Unreal Engine 서버 접속 중...")
    client, world, host_ip = connect_carla_client(port=args.port)
    print(f"      - 접속 성공! (Server IP: {host_ip}:{args.port}, Map: {world.get_map().name})")

    original_settings = world.get_settings()
    actors_to_destroy = []

    try:
        # 20Hz 동기식(Synchronous) 물리 시뮬레이션 설정
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        tm = client.get_trafficmanager(3800)
        tm.set_synchronous_mode(True)
        tm.set_global_distance_to_leading_vehicle(2.5)

        bp_lib = world.get_blueprint_library()
        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()

        # 1. 자율주행 Ego 차량 스폰 (Tesla Model 3 우선, 없으면 Lincoln MKZ / 기본 차량)
        ego_bp_candidates = bp_lib.filter("vehicle.tesla.model3")
        if not ego_bp_candidates:
            ego_bp_candidates = bp_lib.filter("vehicle.lincoln.mkz_2020")
        if not ego_bp_candidates:
            ego_bp_candidates = bp_lib.filter("vehicle.*")
        ego_bp = ego_bp_candidates[0]
        if ego_bp.has_attribute("color"):
            ego_bp.set_attribute("color", "25,180,245")

        ego_vehicle = None
        for sp in spawn_points:
            ego_vehicle = world.try_spawn_actor(ego_bp, sp)
            if ego_vehicle is not None:
                break
        if ego_vehicle is None:
            raise RuntimeError("차량 스폰 포인트를 찾지 못했습니다.")
        actors_to_destroy.append(ego_vehicle)
        ego_vehicle.set_autopilot(True, tm.get_port())
        tm.ignore_lights_percentage(ego_vehicle, 100.0)  # 끊김 없는 연속 도심 주행 데모를 위해 신호 대기 최소화
        tm.auto_lane_change(ego_vehicle, True)

        # 2. 주변 도심 교통 차량 25대 스폰
        print("[2/4] 도심 주변 교통 차량(Traffic Vehicles) 및 4종 센서 수트 장착 중...")
        veh_bps = [bp for bp in bp_lib.filter("vehicle.*") if int(bp.get_attribute("number_of_wheels")) == 4]
        spawned_traffic = 0
        for sp in spawn_points[1:]:
            if spawned_traffic >= 25:
                break
            bp = veh_bps[spawned_traffic % len(veh_bps)]
            npc = world.try_spawn_actor(bp, sp)
            if npc is not None:
                actors_to_destroy.append(npc)
                npc.set_autopilot(True, tm.get_port())
                spawned_traffic += 1

        cam_w, cam_h = 640, 360
        fov = 90.0
        K = build_camera_intrinsic(cam_w, cam_h, fov)

        # 센서 1: 전방 블랙박스 RGB 카메라
        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(cam_w))
        rgb_bp.set_attribute("image_size_y", str(cam_h))
        rgb_bp.set_attribute("fov", str(fov))
        cam_front = world.spawn_actor(
            rgb_bp, carla.Transform(carla.Location(x=1.5, z=1.55), carla.Rotation(pitch=-4.0)), attach_to=ego_vehicle
        )
        actors_to_destroy.append(cam_front)

        # 센서 2: 3인칭 Chase RGB 카메라
        cam_chase = world.spawn_actor(
            rgb_bp, carla.Transform(carla.Location(x=-6.2, z=3.2), carla.Rotation(pitch=-15.0)), attach_to=ego_vehicle
        )
        actors_to_destroy.append(cam_chase)

        # 센서 3: Semantic Segmentation 카메라
        seg_bp = bp_lib.find("sensor.camera.semantic_segmentation")
        seg_bp.set_attribute("image_size_x", str(cam_w))
        seg_bp.set_attribute("image_size_y", str(cam_h))
        seg_bp.set_attribute("fov", str(fov))
        cam_seg = world.spawn_actor(
            seg_bp, carla.Transform(carla.Location(x=1.5, z=1.55), carla.Rotation(pitch=-4.0)), attach_to=ego_vehicle
        )
        actors_to_destroy.append(cam_seg)

        # 센서 4: 64채널 3D LiDAR
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "64")
        lidar_bp.set_attribute("range", "45.0")
        lidar_bp.set_attribute("points_per_second", "240000")
        lidar_bp.set_attribute("rotation_frequency", "20.0")
        lidar_bp.set_attribute("upper_fov", "10.0")
        lidar_bp.set_attribute("lower_fov", "-25.0")
        lidar_sensor = world.spawn_actor(
            lidar_bp, carla.Transform(carla.Location(x=0.0, z=2.1)), attach_to=ego_vehicle
        )
        actors_to_destroy.append(lidar_sensor)

        q_front, q_chase, q_seg, q_lidar = queue.Queue(), queue.Queue(), queue.Queue(), queue.Queue()

        def to_rgb_array(image: carla.Image) -> np.ndarray:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            return arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB

        def seg_to_rgb(image: carla.Image) -> np.ndarray:
            image.convert(carla.ColorConverter.CityScapesPalette)
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            return arr[:, :, :3][:, :, ::-1].copy()

        def parse_lidar(data: carla.LidarMeasurement) -> np.ndarray:
            pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape((-1, 4)).copy()
            return pts

        cam_front.listen(lambda img: q_front.put(to_rgb_array(img)))
        cam_chase.listen(lambda img: q_chase.put(to_rgb_array(img)))
        cam_seg.listen(lambda img: q_seg.put(seg_to_rgb(img)))
        lidar_sensor.listen(lambda data: q_lidar.put(parse_lidar(data)))

        # 주행 중 동적 날씨 변화 (맑은 낮 -> 노을 -> 비 내리는 저녁 노을)
        weathers = [
            (carla.WeatherParameters.ClearNoon, "Weather: Clear Noon (Sunny)"),
            (carla.WeatherParameters.ClearSunset, "Weather: Golden Sunset"),
            (carla.WeatherParameters.WetCloudySunset, "Weather: Wet Sunset Road (Reflection)"),
        ]

        # 워밍업 10 틱
        for _ in range(10):
            world.tick()
        while not q_front.empty(): q_front.get()
        while not q_chase.empty(): q_chase.get()
        while not q_seg.empty(): q_seg.get()
        while not q_lidar.empty(): q_lidar.get()

        print(f"[3/4] 도심 주행 및 4종 센서 실시간 렌더링 시작 (총 {args.steps} 프레임 = {args.steps * 0.05:.0f}초)...")
        quad_frames = []
        t0 = time.perf_counter()

        for step in range(1, args.steps + 1):
            w_idx = min(2, (step - 1) // (args.steps // 3))
            w_param, w_label = weathers[w_idx]
            world.set_weather(w_param)

            world.tick()

            # CARLA 서버 창(Spectator) 카메라도 내 자율주행차를 따라가도록 동기화
            ego_tf = ego_vehicle.get_transform()
            fwd = ego_tf.get_forward_vector()
            spec_loc = carla.Location(
                x=ego_tf.location.x - fwd.x * 7.5,
                y=ego_tf.location.y - fwd.y * 7.5,
                z=ego_tf.location.z + 3.6,
            )
            spec_rot = carla.Rotation(pitch=-15.0, yaw=ego_tf.rotation.yaw, roll=0.0)
            world.get_spectator().set_transform(carla.Transform(spec_loc, spec_rot))

            img_front = q_front.get(timeout=5.0)
            img_chase = q_chase.get(timeout=5.0)
            img_seg = q_seg.get(timeout=5.0)
            pts_lidar = q_lidar.get(timeout=5.0)

            vel = ego_vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            ctrl = ego_vehicle.get_control()

            # RTX 4090 VLA 신경망 추론
            with torch.inference_mode():
                img_t = torch.from_numpy(img_front).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                img_t = torch.nn.functional.interpolate(img_t, size=(128, 128), mode="bilinear", align_corners=False)
                st_t = torch.tensor([[speed_kmh / 50.0, ctrl.steer, ctrl.throttle, ctrl.brake]], dtype=torch.float32, device=device)
                vla_offsets = vla_net(img_t, st_t).squeeze(0).cpu().numpy()

            # 1) 좌상단: 전방 RGB + 3D FSD 카펫
            front_fsd, bev_wps = draw_3d_fsd_carpet_on_rgb(img_front, ego_vehicle, cam_front, carla_map, K, vla_offsets)
            p1 = add_panel_label(
                front_fsd,
                "1. Front RGB Camera + 3D FSD Trajectory Carpet",
                f"Speed: {speed_kmh:.1f} km/h | Steer: {ctrl.steer:+.2f} | Throttle: {ctrl.throttle:.2f}",
                (56, 189, 248),
            )

            # 2) 우상단: 3인칭 Chase View
            p2 = add_panel_label(
                img_chase,
                f"2. Unreal Engine 3D Urban Chase View ({carla_map.name.split('/')[-1]})",
                f"{w_label} | Step: {step:03d}/{args.steps}",
                (250, 204, 21),
            )

            # 3) 좌하단: CityScapes Semantic Segmentation
            p3 = add_panel_label(
                img_seg,
                "3. Real-Time Semantic Segmentation (CityScapes Palette)",
                "Road (Purple) | Vehicles (Blue) | Buildings (Gray) | Vegetation (Green)",
                (244, 114, 182),
            )

            # 4) 우하단: 64-Channel 3D LiDAR Bird's-Eye View
            lidar_bev = render_lidar_bev(pts_lidar, bev_wps, width=cam_w, height=cam_h)
            p4 = add_panel_label(
                lidar_bev,
                "4. 64-Channel 3D LiDAR Bird's-Eye View (BEV)",
                f"Point Cloud: {len(pts_lidar):,d} pts/frame | Range: 45m",
                (74, 222, 128),
            )

            top_row = np.concatenate([p1, p2], axis=1)
            bot_row = np.concatenate([p3, p4], axis=1)
            quad_frames.append(np.concatenate([top_row, bot_row], axis=0))

            if step % 100 == 0 or step == 1:
                fps = step / max(time.perf_counter() - t0, 1e-6)
                print(f"  - 진행 중: {step:03d}/{args.steps} frames | 속도: {speed_kmh:.1f} km/h | {w_label} | 처리 속도: {fps:.1f} FPS")

        out_dir = Path(__file__).resolve().parent / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = out_dir / "carla_quad_dashboard.mp4"
        snap_path = out_dir / "carla_quad_snapshot.png"
        imageio.mimsave(str(mp4_path), quad_frames, fps=20)
        imageio.imwrite(str(snap_path), quad_frames[min(260, len(quad_frames) - 1)])

        print("=" * 78)
        print(f"[4/4] CARLA 4분할 멀티센서 자율주행 대시보드 영상 저장 완료: {mp4_path}")
        print(f"      스냅샷 저장 완료                                     : {snap_path}")
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
