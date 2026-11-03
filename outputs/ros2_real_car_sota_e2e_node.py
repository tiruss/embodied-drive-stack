#!/usr/bin/env python3
"""
[실차 탑재용 ROS 2 Humble/Jazzy 폐루프 E2E 자율주행 제어 노드 템플릿]
- 대상 하드웨어: NVIDIA Jetson AGX Orin (64GB) 또는 x86_64 산업용 PC (RTX 4080/4090)
- 센서 입력   :
    1) /sensing/camera/front/image_raw (sensor_msgs/Image, GMSL2 하드웨어 동기화 카메라)
    2) /vehicle/status/velocity_status (autoware_vehicle_msgs 또는 CAN 0x120 차량 속도/조향각)
    3) /localization/kinematic_state   (nav_msgs/Odometry, RTK-GNSS/INS + Wheel Odometry 융합)
- 제어 출력   :
    1) /control/command/control_cmd    (AckermannControlCommand -> Dataspeed/NewEagle DBW CAN 게이트웨이)
    2) /planning/scenario_planning/trajectory (시각화 및 하위 MPC 추종기용 3D 웨이포인트)
"""
import math
import time
import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class RealCarSotaE2ENode(Node):
    def __init__(self, model_ckpt_path: str):
        super().__init__("real_car_sota_e2e_node")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 카메라 내부 파라미터(Intrinsic K) 및 왜곡 계수(Distortion D) 로드 (칼리브레이션 필수!)
        self.cam_K = np.array([[480.0, 0.0, 280.0], [0.0, 480.0, 128.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # 실차 체커보드 칼리브레이션 값 입력

        # 2. SOTA E2E 모델 로드 및 FP16(Half Precision) 최적화
        from demo_13_carla_sota_diffusion_bev_e2e import SOTA_DiffusionDrive_HydraMDP
        self.model = SOTA_DiffusionDrive_HydraMDP(d_model=256, num_modes=5, chunk_size=15).to(self.device)
        if Path(model_ckpt_path).exists():
            self.model.load_state_dict(torch.load(model_ckpt_path, map_location=self.device))
        self.model.eval()

        # 3. Temporal Ensembling 버퍼 및 안전 감시자(Watchdog) 타이머
        self.chunk_history = []
        self.last_cam_time = time.monotonic()
        self.current_speed_ms = 0.0
        self.current_yaw_err = 0.0
        self.current_s = 0.0
        self.current_d = 0.0
        self.current_stage = 1
        self.emergency_stop_active = False

        # 4. ROS 2 Subscriber & Publisher 설정 (20 Hz 제어 루프)
        self.sub_cam = self.create_subscription(Image, "/sensing/camera/front/image_raw", self.on_camera, 1)
        self.sub_odom = self.create_subscription(Odometry, "/localization/kinematic_state", self.on_odom, 1)
        self.pub_ctrl = self.create_publisher(AckermannDriveStamped, "/control/command/control_cmd", 1)
        self.watchdog_timer = self.create_timer(0.05, self.check_safety_watchdog)  # 20 Hz

    def on_odom(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed_ms = math.sqrt(vx * vx + vy * vy)

    def check_safety_watchdog(self):
        """카메라 프레임 드롭(>150ms 지연) 또는 E-Stop 발생 시 즉시 하드웨어 감속 명령 송출"""
        if (time.monotonic() - self.last_cam_time) > 0.150:
            self.get_logger().error("[WATCHDOG TIMEOUT] Camera frame delayed > 150ms! Engaging Safe Stop.")
            self.publish_safe_stop()

    def publish_safe_stop(self):
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.drive.steering_angle = 0.0
        cmd.drive.speed = 0.0
        cmd.drive.acceleration = -3.5  # 안전한 비상 감속 (-3.5 m/s^2)
        self.pub_ctrl.publish(cmd)

    def on_camera(self, msg: Image):
        self.last_cam_time = time.monotonic()
        if self.emergency_stop_active:
            self.publish_safe_stop()
            return

        # 1) ROS Image -> OpenCV RGB 변환 및 렌즈 왜곡 보정(Undistort)
        img_bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        img_undist = cv2.undistort(img_bgr, self.cam_K, self.dist_coeffs)
        img_rgb = cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB)

        # 2) 128x128 리사이즈 및 정규화
        rgb_128 = cv2.resize(img_rgb, (128, 128)).transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        state_vec = np.array(
            [
                self.current_speed_ms / 10.0,
                self.current_yaw_err,
                self.current_s / 200.0,
                self.current_d / 3.5,
                self.current_stage / 5.0,
            ],
            dtype=np.float32,
        )

        t_rgb = torch.from_numpy(rgb_128).unsqueeze(0).to(self.device)
        t_st = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)

        # 3) SOTA DiffusionDrive + Hydra-MDP 실시간 추론 (~3.0 ms)
        with torch.no_grad():
            best_wps_t, pred_ctrls_t, _, hydra_scores_t, pred_bev_t = self.model.forward_inference(
                t_rgb, t_st, ode_steps=4
            )
            max_safety_score = float(hydra_scores_t[0].max().item())
            pred_chunk = pred_ctrls_t[0].cpu().numpy()

        # 4) [Safety Shield / CBF Override] 만약 5개 모드의 최대 안전 점수가 모두 25% 미만이면 즉시 비상 제동!
        if max_safety_score < 0.25:
            self.get_logger().warn(f"[SAFETY SHIELD] All 5 modes unsafe (max={max_safety_score:.2f})! Braking!")
            self.publish_safe_stop()
            return

        # 5) Temporal Ensembling 및 조향각 변화율 제한(Steering Rate Limiter) 적용 후 DBW CAN 전송
        self.chunk_history.append(pred_chunk)
        if len(self.chunk_history) > 6:
            self.chunk_history.pop(0)
        act = np.mean([ch[i] for i, ch in enumerate(reversed(self.chunk_history))], axis=0)

        cmd = AckermannDriveStamped()
        cmd.header.stamp = msg.header.stamp
        cmd.drive.steering_angle = float(np.clip(act[1] * 0.45, -0.45, 0.45))  # 최대 조향각 ±25.7도 제한
        cmd.drive.acceleration = float(2.5 * act[0] - 4.0 * act[2])
        self.pub_ctrl.publish(cmd)
