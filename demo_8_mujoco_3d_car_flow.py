"""
Demo 8: Long-Run MuJoCo 3D Autonomous Driving (220m 4-Stage Multi-Obstacle Slalom Course)
        Front Dashcam PIP + 3D Chase View + Live Highway Mini-Map (600 Frames / 20 Seconds)
===========================================================================================
기존 4초(40m) 코스를 20초(600프레임, 220m 장거리 4연속 장애물 슬라럼 코스)로 대폭 확장한 3D 자율주행 시뮬레이션입니다!

[220m 4단계 장애물 코스 구성]
- Stage 1 (x = 35m) : 중앙 차선(y=0m) 빨간 승용차 고장 정차 -> 좌(+4m) 또는 우(-4m) 회피 필요 (L2 회귀는 0m 평균 직진으로 1차 충돌!)
- Stage 2 (x = 85m) : 좌측(+4m) & 우측(-4m) 차선 동시 통제 -> 중앙 차선(y=0m) 좁은 통로로 복귀 통과
- Stage 3 (x = 135m): 중앙 차선(y=0m) 대형 화물 트럭 정차 -> 교차 슬라럼(Mode A는 우측 -4m, Mode B는 좌측 +4m) 회피 (L2 회귀는 또 2차 충돌!)
- Stage 4 (x = 185m): 좌측(+4m) & 우측(-4m) 통제 구간 통과 후 x=215m 피니시 라인 완주!
"""
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np
import torch
import torch.nn as nn


def build_long_highway_xml() -> str:
    lane_Paint = []
    for x in range(6, 235, 8):
        lane_Paint.append(f'<geom type="box" pos="{x} 2.0 0.01" size="2.0 0.06 0.01" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>')
        lane_Paint.append(f'<geom type="box" pos="{x} -2.0 0.01" size="2.0 0.06 0.01" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>')
    lane_xml = "\n    ".join(lane_Paint)

    return f"""
<mujoco model="autonomous_driving_long_3d">
  <option timestep="0.02" gravity="0 0 -9.81"/>
  <visual>
    <headlight ambient="0.45 0.45 0.52" diffuse="0.72 0.72 0.75" specular="0.3 0.3 0.3"/>
    <rgba haze="0.07 0.11 0.18 1"/>
    <global offwidth="640" offheight="480"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.08 0.14 0.26" rgb2="0.02 0.04 0.09" width="512" height="512"/>
    <texture name="road_tex" type="2d" builtin="checker" rgb1="0.16 0.18 0.22" rgb2="0.14 0.16 0.20" width="256" height="256"/>
    <material name="road_mat" texture="road_tex" texrepeat="120 4" reflectance="0.08"/>
  </asset>

  <worldbody>
    <light pos="110 0 40" dir="0 0 -1" directional="true" diffuse="0.85 0.85 0.90"/>
    <!-- 250m 3차선 아스팔트 고속도로 -->
    <geom name="road" type="plane" pos="120 0 0" size="135 6.5 0.1" material="road_mat"/>
    <geom name="curb_left" type="box" pos="120 6.3 0.22" size="135 0.15 0.22" rgba="0.35 0.42 0.52 1"/>
    <geom name="curb_right" type="box" pos="120 -6.3 0.22" size="135 0.15 0.22" rgba="0.35 0.42 0.52 1"/>

    {lane_xml}

    <!-- Stage 1 (x=35m): 중앙 차선 빨간 고장 승용차 -->
    <body name="obs_stage1_center" pos="35.0 0.0 0.55">
      <geom type="box" size="1.85 0.92 0.42" rgba="0.92 0.22 0.25 1"/>
      <geom type="box" pos="-0.2 0 0.55" size="0.95 0.78 0.32" rgba="0.75 0.16 0.20 1"/>
      <geom type="box" pos="-1.86 0.65 0.1" size="0.05 0.22 0.12" rgba="1.0 0.85 0.1 1"/>
      <geom type="box" pos="-1.86 -0.65 0.1" size="0.05 0.22 0.12" rgba="1.0 0.85 0.1 1"/>
    </body>

    <!-- Stage 2 (x=85m): 좌/우 양끝 차선(+4m, -4m) 통제 차량들 (중앙 0m만 통과 가능!) -->
    <body name="obs_stage2_left" pos="85.0 4.0 0.55">
      <geom type="box" size="1.9 0.92 0.44" rgba="0.95 0.55 0.15 1"/>
      <geom type="box" pos="-0.2 0 0.55" size="1.0 0.78 0.32" rgba="0.75 0.42 0.12 1"/>
    </body>
    <body name="obs_stage2_right" pos="85.0 -4.0 0.55">
      <geom type="box" size="1.9 0.92 0.44" rgba="0.95 0.55 0.15 1"/>
      <geom type="box" pos="-0.2 0 0.55" size="1.0 0.78 0.32" rgba="0.75 0.42 0.12 1"/>
    </body>

    <!-- Stage 3 (x=135m): 중앙 차선 대형 화물 트럭 정차 -->
    <body name="obs_stage3_truck" pos="135.0 0.0 0.85">
      <geom type="box" size="2.8 1.02 0.75" rgba="0.88 0.25 0.30 1"/>
      <geom type="box" pos="2.2 0 -0.15" size="0.8 0.96 0.58" rgba="0.70 0.20 0.24 1"/>
      <geom type="box" pos="-2.82 0.72 -0.2" size="0.05 0.25 0.15" rgba="1.0 0.85 0.1 1"/>
      <geom type="box" pos="-2.82 -0.72 -0.2" size="0.05 0.25 0.15" rgba="1.0 0.85 0.1 1"/>
    </body>

    <!-- Stage 4 (x=185m): 좌/우 차선 통제 차량들 (다시 중앙 0m 복귀 통과!) -->
    <body name="obs_stage4_left" pos="185.0 4.0 0.55">
      <geom type="box" size="1.85 0.92 0.42" rgba="0.65 0.35 0.88 1"/>
      <geom type="box" pos="-0.2 0 0.55" size="0.95 0.78 0.32" rgba="0.50 0.25 0.70 1"/>
    </body>
    <body name="obs_stage4_right" pos="185.0 -4.0 0.55">
      <geom type="box" size="1.85 0.92 0.42" rgba="0.65 0.35 0.88 1"/>
      <geom type="box" pos="-0.2 0 0.55" size="0.95 0.78 0.32" rgba="0.50 0.25 0.70 1"/>
    </body>

    <!-- 피니시 아치 게이트 (x=215m) -->
    <body name="finish_gate" pos="215.0 0.0 0.0">
      <geom type="cylinder" pos="0 6.1 2.5" size="0.18 2.5" rgba="0.2 0.9 0.5 1"/>
      <geom type="cylinder" pos="0 -6.1 2.5" size="0.18 2.5" rgba="0.2 0.9 0.5 1"/>
      <geom type="box" pos="0 0 4.8" size="0.2 6.3 0.35" rgba="0.15 0.85 0.48 1"/>
    </body>

    <!-- 자율주행 차량 (Ego Autonomous Vehicle) -->
    <body name="ego_car" pos="0 0 0.55">
      <joint name="ego_x" type="slide" axis="1 0 0" damping="1.0"/>
      <joint name="ego_y" type="slide" axis="0 1 0" damping="1.0"/>
      <joint name="ego_yaw" type="hinge" axis="0 0 1" damping="1.0"/>

      <geom name="ego_chassis" type="box" size="1.85 0.90 0.40" rgba="0.15 0.75 0.95 1"/>
      <geom name="ego_cabin" type="box" pos="-0.15 0 0.52" size="0.95 0.76 0.30" rgba="0.10 0.25 0.40 1"/>
      <geom name="lidar_dome" type="cylinder" pos="0.1 0 0.90" size="0.22 0.10" rgba="0.2 0.95 0.6 1"/>

      <!-- 1인칭 전방 블랙박스 카메라 (Front Dashcam) -->
      <camera name="dashcam" pos="0.85 0 0.88" xyaxes="0 -1 0 0.10 0 1" fovy="58"/>
      <!-- 3인칭 후방 추적 카메라 (3D Chase Cam) -->
      <camera name="chasecam" pos="-11.0 0 7.0" xyaxes="0 -1 0 0.46 0 0.89" fovy="52"/>
    </body>
  </worldbody>
</mujoco>
"""


def smooth_step(x: np.ndarray, center: float, width: float = 4.5) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x - center) / width))


def generate_multi_stage_slalom_trajectories(steps: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """
    220m 4단계 장애물 코스를 통과하는 두 가지 모드의 전문가 궤적 생성 (600 스텝 = 20초)
    - Mode A: Stage 1 좌측(+4m) -> Stage 2 중앙(0m) -> Stage 3 우측(-4m) -> Stage 4 중앙(0m)
    - Mode B: Stage 1 우측(-4m) -> Stage 2 중앙(0m) -> Stage 3 좌측(+4m) -> Stage 4 중앙(0m)
    """
    x = np.linspace(0.0, 220.0, steps)

    # x=20m에서 회피 시작 -> x=62m에서 중앙 복귀 -> x=112m에서 2차 회피 -> x=162m에서 중앙 복귀
    y_mode_a = (
        4.0 * smooth_step(x, 20.0, 3.2)
        - 4.0 * smooth_step(x, 62.0, 3.2)
        - 4.0 * smooth_step(x, 112.0, 3.2)
        + 4.0 * smooth_step(x, 162.0, 3.2)
    )
    y_mode_b = -y_mode_a

    dx = np.gradient(x)
    dy_a = np.gradient(y_mode_a)
    yaw_a = np.arctan2(dy_a, dx)

    traj_a = np.stack([x, y_mode_a, yaw_a], axis=-1).astype(np.float32)
    traj_b = np.stack([x, y_mode_b, -yaw_a], axis=-1).astype(np.float32)
    return traj_a, traj_b


class L2RegressionPlanner(nn.Module):
    def __init__(self, steps: int = 600):
        super().__init__()
        self.steps = steps
        self.net = nn.Sequential(
            nn.Linear(4, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, steps * 3),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond).view(-1, self.steps, 3)


class FlowMatchingPlanner(nn.Module):
    def __init__(self, steps: int = 600):
        super().__init__()
        self.steps = steps
        dim = steps * 3
        self.net = nn.Sequential(
            nn.Linear(dim + 4 + 1, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, dim),
        )

    def forward(self, x_tau: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b = x_tau.shape[0]
        inp = torch.cat([x_tau.view(b, -1), cond, tau[:, None]], dim=-1)
        return self.net(inp).view(b, self.steps, 3)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, z_init: torch.Tensor, ode_steps: int = 12) -> torch.Tensor:
        x = z_init.clone()
        dt = 1.0 / ode_steps
        for s in range(ode_steps):
            tau = torch.full((cond.shape[0],), s * dt, device=cond.device)
            v = self.forward(x, tau, cond)
            x = x + v * dt
        return x


def draw_minimap(canvas: np.ndarray, curr_x: float, curr_y: float, color: tuple[int, int, int], crashed: bool):
    """화면 하단에 220m 전체 코스와 장애물 위치, 현재 차량 위치를 보여주는 실시간 미니맵을 렌더링합니다."""
    h, w, _ = canvas.shape
    map_x0, map_y0 = 14, h - 54
    map_w, map_h = w - 28, 42

    overlay = canvas.copy()
    cv2.rectangle(overlay, (map_x0, map_y0), (map_x0 + map_w, map_y0 + map_h), (12, 18, 32), -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (map_x0, map_y0), (map_x0 + map_w, map_y0 + map_h), (60, 75, 100), 1)

    # 차선 점선
    for ly in [map_y0 + 14, map_y0 + 28]:
        cv2.line(canvas, (map_x0 + 4, ly), (map_x0 + map_w - 4, ly), (70, 85, 110), 1)

    def to_px(xm: float, ym: float) -> tuple[int, int]:
        px = int(map_x0 + 8 + (xm / 220.0) * (map_w - 16))
        # ym in [-4, +4] -> 위쪽이 +4m(좌측 차선), 아래쪽이 -4m(우측 차선)
        py = int(map_y0 + map_h // 2 - (ym / 6.0) * (map_h // 2 - 4))
        return px, py

    # 4개 스테이지 장애물 표시 (빨간/주황 블록)
    obstacles = [
        (35.0, 0.0, (240, 70, 70)),
        (85.0, 4.0, (245, 150, 50)),
        (85.0, -4.0, (245, 150, 50)),
        (135.0, 0.0, (240, 70, 70)),
        (185.0, 4.0, (180, 100, 240)),
        (185.0, -4.0, (180, 100, 240)),
    ]
    for ox, oy, ocol in obstacles:
        opx, opy = to_px(ox, oy)
        cv2.rectangle(canvas, (opx - 5, opy - 4), (opx + 5, opy + 4), ocol, -1)

    # 현재 내 차 위치
    epx, epy = to_px(curr_x, curr_y)
    car_col = (248, 113, 113) if crashed else color
    cv2.circle(canvas, (epx, epy), 5, car_col, -1, cv2.LINE_AA)
    cv2.circle(canvas, (epx, epy), 6, (255, 255, 255), 1, cv2.LINE_AA)


def render_car_rollout(
    mj_model: mujoco.MjModel,
    renderer_chase: mujoco.Renderer,
    renderer_dash: mujoco.Renderer,
    traj: np.ndarray,
    title: str,
    subtitle: str,
    banner_color: tuple[int, int, int],
    is_l2_mode: bool = False,
) -> list[np.ndarray]:
    mj_data = mujoco.MjData(mj_model)
    frames = []
    crash_count = 0

    for t in range(traj.shape[0]):
        x, y, yaw = traj[t]
        crashed_now = False

        # 중앙 차선 장애물(x=35m 승용차, x=135m 대형 트럭) 충돌 체크
        if is_l2_mode:
            if 31.2 <= x <= 48.0 and abs(y) < 1.6:
                crashed_now = True
                x_vis = 31.2  # 1차 장애물 앞에서 정면충돌 상태로 멈춤 표시 후 리스폰!
                if t == 86:
                    crash_count += 1
            elif 130.5 <= x <= 152.0 and abs(y) < 1.7:
                crashed_now = True
                x_vis = 130.5  # 3차 트럭 장애물 앞에서 2차 정면충돌!
                if t == 358:
                    crash_count += 1
            else:
                x_vis = x
        else:
            x_vis = x

        mj_data.qpos[0] = x_vis
        mj_data.qpos[1] = y
        mj_data.qpos[2] = 0.10 if crashed_now else yaw
        mujoco.mj_forward(mj_model, mj_data)

        renderer_chase.update_scene(mj_data, camera="chasecam")
        chase_img = renderer_chase.render().copy()

        renderer_dash.update_scene(mj_data, camera="dashcam")
        dash_img = renderer_dash.render().copy()
        dash_small = cv2.resize(dash_img, (176, 114))

        border_col = (248, 113, 113) if crashed_now else banner_color
        cv2.rectangle(dash_small, (0, 0), (175, 113), border_col, 2)
        cv2.putText(dash_small, "Front Dashcam", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        chase_img[70:184, chase_img.shape[1] - 188 : chase_img.shape[1] - 12] = dash_small

        # 상단 배너
        overlay = chase_img.copy()
        cv2.rectangle(overlay, (0, 0), (chase_img.shape[1], 62), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.80, chase_img, 0.20, 0, chase_img)

        cv2.putText(chase_img, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54, banner_color, 2, cv2.LINE_AA)
        if crashed_now:
            status_str = f"COLLISION #{crash_count}! (Mode Averaging y=0m)"
            status_col = (248, 113, 113)
        else:
            status_str = f"{subtitle} | Pos: {x_vis:.0f}m/220m | Crashes: {crash_count}"
            status_col = (226, 232, 240)
        cv2.putText(chase_img, status_str, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_col, 1, cv2.LINE_AA)

        # 하단 실시간 220m 코스 미니맵 렌더링
        draw_minimap(chase_img, x_vis, y, banner_color, crashed_now)
        frames.append(chase_img)

    return frames


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = 600  # 600 frames @ 30fps = 20초 장거리 슬라럼 주행!

    print("=" * 78)
    print(f"[1/3] 220m 4단계 장애물 슬라럼 코스 시연 데이터 생성 중 ({steps} 프레임)...")
    print("=" * 78)

    traj_a, traj_b = generate_multi_stage_slalom_trajectories(steps=steps)

    conds = torch.zeros((200, 4), device=device)
    conds[:, 0] = 35.0
    conds[:, 1] = 85.0
    conds[:, 2] = 135.0
    conds[:, 3] = 185.0

    targets = []
    for i in range(200):
        base = traj_a if i < 100 else traj_b
        noise = np.random.normal(0.0, 0.04, size=base.shape).astype(np.float32)
        targets.append(torch.from_numpy(base + noise))
    targets_t = torch.stack(targets).to(device)

    print("[2/3] RTX 4090에서 (A) L2 회귀 플래너 vs (B) Flow-Matching 슬라럼 플래너 학습 중...")
    l2_planner = L2RegressionPlanner(steps=steps).to(device)
    opt_l2 = torch.optim.AdamW(l2_planner.parameters(), lr=2e-3)
    for _ in range(180):
        pred = l2_planner(conds)
        loss = nn.functional.mse_loss(pred, targets_t)
        opt_l2.zero_grad()
        loss.backward()
        opt_l2.step()

    flow_planner = FlowMatchingPlanner(steps=steps).to(device)
    opt_flow = torch.optim.AdamW(flow_planner.parameters(), lr=2e-3)
    for _ in range(320):
        mode_sign = torch.where(torch.arange(200, device=device) < 100, 1.0, -1.0)[:, None, None]
        z_0 = mode_sign * 0.35 + torch.randn_like(targets_t) * 0.06
        tau = torch.rand(targets_t.shape[0], device=device)
        tau_v = tau[:, None, None]
        x_tau = (1.0 - tau_v) * z_0 + tau_v * targets_t
        v_target = targets_t - z_0

        v_pred = flow_planner(x_tau, tau, conds)
        loss_f = nn.functional.mse_loss(v_pred, v_target)
        opt_flow.zero_grad()
        loss_f.backward()
        opt_flow.step()

    l2_planner.eval()
    flow_planner.eval()
    test_cond = conds[:1]
    with torch.inference_mode():
        traj_l2 = l2_planner(test_cond).squeeze(0).cpu().numpy()
        z_a = torch.full((1, steps, 3), 0.35, device=device)
        traj_flow_a = flow_planner.sample(test_cond, z_a, ode_steps=12).squeeze(0).cpu().numpy()
        z_b = torch.full((1, steps, 3), -0.35, device=device)
        traj_flow_b = flow_planner.sample(test_cond, z_b, ode_steps=12).squeeze(0).cpu().numpy()

    print(f"  - Stage 1(x=35m) 예측 y좌표: L2={traj_l2[95, 1]:+.2f}m(충돌!) | Flow-A={traj_flow_a[95, 1]:+.2f}m | Flow-B={traj_flow_b[95, 1]:+.2f}m")
    print(f"  - Stage 2(x=85m) 예측 y좌표: L2={traj_l2[230, 1]:+.2f}m       | Flow-A={traj_flow_a[230, 1]:+.2f}m | Flow-B={traj_flow_b[230, 1]:+.2f}m (중앙 게이트 통과!)")
    print(f"  - Stage 3(x=135m)예측 y좌표: L2={traj_l2[368, 1]:+.2f}m(충돌!) | Flow-A={traj_flow_a[368, 1]:+.2f}m | Flow-B={traj_flow_b[368, 1]:+.2f}m (교차 슬라럼!)")

    print("\n[3/3] 600 프레임(20초) MuJoCo 3D 장거리 슬라럼 주행 + 실시간 미니맵 렌더링 중...")
    mj_model = mujoco.MjModel.from_xml_string(build_long_highway_xml())
    renderer_chase = mujoco.Renderer(mj_model, height=432, width=440)
    renderer_dash = mujoco.Renderer(mj_model, height=240, width=360)

    frames_l2 = render_car_rollout(
        mj_model, renderer_chase, renderer_dash, traj_l2,
        "1. Standard BC (L2 Regression)", "Mode Averaging -> Straight", (248, 113, 113), is_l2_mode=True
    )
    frames_flow_a = render_car_rollout(
        mj_model, renderer_chase, renderer_dash, traj_flow_a,
        "2. Flow-Matching (Slalom Mode A)", "Left -> Center -> Right -> Finish", (56, 189, 248), is_l2_mode=False
    )
    frames_flow_b = render_car_rollout(
        mj_model, renderer_chase, renderer_dash, traj_flow_b,
        "3. Flow-Matching (Slalom Mode B)", "Right -> Center -> Left -> Finish", (74, 222, 128), is_l2_mode=False
    )

    renderer_chase.close()
    renderer_dash.close()

    combined = [
        np.concatenate([frames_l2[t], frames_flow_a[t], frames_flow_b[t]], axis=1)
        for t in range(steps)
    ]

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / "mujoco_3d_car_flow_vs_l2.mp4"
    snap_path = out_dir / "mujoco_3d_car_snapshot.png"
    imageio.mimsave(str(mp4_path), combined, fps=30)
    imageio.imwrite(str(snap_path), combined[365])

    print("=" * 78)
    print(f"20초(600 프레임) MuJoCo 3D 장거리 슬라럼 영상 저장 완료: {mp4_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
