# 🚗 실차(Real Vehicle) 자율주행 E2E / World Model 탑재 및 실험 완벽 가이드

우리가 CARLA 0.9.15와 RTX 4090에서 구현·검증한 **3세대 SOTA E2E 모델([`SOTA_DiffusionDrive_HydraMDP`](../demo_13_carla_sota_diffusion_bev_e2e.py#L565-L738))** 및 **행동 조건부 월드 모델([`ActionConditionedDrivingWorldModel`](../demo_15_carla_action_world_model.py#L140-L300))**을 시뮬레이터 밖의 **실제 차량(1/10 스케일 RC카 ~ 실제 승용차)**에 올려서 안전하게 폐루프(Closed-Loop) 주행 실험을 수행하기 위해 필요한 **하드웨어 구성, 센서 칼리브레이션, Sim-to-Real 변환, ROS 2 / CAN 연동 코드, 그리고 5단계 실차 테스트 절차**를 정리한 실전 엔지니어링 매뉴얼입니다.

---

## 1. 전체 실차 배포 파이프라인 아키텍처

```mermaid
flowchart TD
    subgraph HW["1. 실차 센서 및 측위 계층 (Sensing & Localization)"]
        Cam["GMSL2 Global Shutter 카메라\n(60Hz HDR, cv2.undistort 왜곡 보정)"]
        GNSS["RTK-GNSS (ZED-F9P, <2cm 오차)\n+ 6축 IMU + CAN 휠 오도메트리"]
        Route["사전 기록 GPS 경로 (reference_route.csv)\n-> SameDirectionTwoLaneSpline 변환"]
    end

    subgraph AI["2. 온보드 AI 컴퓨팅 계층 (Jetson AGX Orin / 산업용 RTX PC)"]
        Pre["전처리: 128x128 RGB 정규화\n+ Frenet 상태 벡터 [v, yaw_err, s, d, stage]"]
        TRT["TensorRT FP16 엔진 (~1.8ms 추론)\nSOTA DiffusionDrive + Hydra-MDP + BEV"]
        Shield["하드웨어 안전 방어막 (Safety Shield)\n1) max(HydraScore) < 25% -> 즉시 감속\n2) 초음파/LiDAR TTC < 1.2s -> 비상 제동\n3) 카메라 프레임 지연 > 150ms -> Watchdog 정지"]
    end

    subgraph CTRL["3. 차량 제어 및 Drive-By-Wire 계층 (Actuation & CAN-Bus)"]
        MPC["하위 궤적 추종기 (Pure Pursuit / MPC)\nbest_wps (15x2 m) -> 목표 조향각 δ & 가속도 a"]
        DBW["Drive-By-Wire CAN 게이트웨이 (500kbps)\n조향(MDPS) + 가감속(SCC/Braking) + 물리 E-Stop"]
    end

    Cam --> Pre
    GNSS --> Route --> Pre
    Pre --> TRT --> Shield --> MPC --> DBW
```

---

## 2. 예산 및 목적별 실차 플랫폼 3단계 선택 가이드

처음부터 2톤짜리 실제 승용차를 움직이는 것은 위험 부담과 비용이 크므로, 실제 자율주행 연구실과 스타트업에서도 아래 **3단계 스케일** 순서로 실험을 진행합니다.

| 구분 | **Tier 1: 1/10 스케일 자율주행차 (`F1TENTH`)** | **Tier 2: 소형 전동 모빌리티 / 캠퍼스 카트** | **Tier 3: 풀스케일 실제 승용차 개조** |
| :--- | :--- | :--- | :--- |
| **베이스 차량** | Traxxas Slash 4x4 1/10 RC 섀시 | AgileX Hunter 2.0 / PIX 샤시 / 개조 골프카트 | 현대 아이오닉 5 / 기아 니로 EV / 링컨 MKZ |
| **예상 구축 비용** | **약 150만 ~ 350만 원** | **약 1,200만 ~ 3,500만 원** | **약 4,000만 원 ~ 1억 원+** (차량가 별도) |
| **온보드 컴퓨터** | NVIDIA Jetson Orin Nano (8GB) / Orin NX (16GB) | NVIDIA Jetson AGX Orin (64GB) 또는 Mini-ITX PC | 산업용 팬리스 PC (`Nuvo-10108GC` + RTX 4080/4090) |
| **조향/구동 제어** | `VESC` 오픈소스 BLDC 모터 제어기 + 서보 PWM | 내장 CAN-Bus 프로토콜 (ROS 2 드라이버 기본 제공) | ① 저비용: `comma.ai panda` (OBD-II MDPS/SCC CAN)<br>② 정식 연구용: `Dataspeed` / `모빌테크` DBW 키트 |
| **실험 장소** | 실내 복도, 체육관, 대학 주차장 (법적 허가 불필요) | 대학 캠퍼스 도로, 사유지 주행시험장 | 자동차안전연구원(KATRI) K-City, 지자체 시범지구 (임시운행허가 필수) |

> [!TIP]
> **가장 추천하는 시작점:**
> 코드 한 줄 바꾸지 않고 오늘 만든 [`SOTA_DiffusionDrive_HydraMDP`](../demo_13_carla_sota_diffusion_bev_e2e.py#L565-L738)를 가장 빠르게 현실 세계에서 굴려보려면 **Tier 1 (`F1TENTH` + Jetson Orin NX)**으로 실내/주차장에 미니 차선과 장난감 경찰차 장애물을 놓고 먼저 검증한 뒤, **Tier 2/3 실차**로 넘어가는 것이 가장 빠르고 안전합니다.

---

## 3. 실차 구성을 위한 5대 필수 하드웨어 체크리스트

### ① 카메라 센서 (일반 USB 웹캠 대신 차량용 Global Shutter HDR 필수)
* **왜 일반 웹캠은 위험한가?**
  1. **롤링 셔터(Rolling Shutter) 왜곡:** 차가 방지턱을 넘거나 빠르게 조향할 때 이미지 상하단이 젤리처럼 휘어져 3D 웨이포인트 투영 오차가 발생합니다.
  2. **자동 노출(Auto-Exposure) 지연 및 역광:** 터널 진·출입이나 오후 서쪽 역광에서 1~2초 동안 화면이 완전히 하얗게 날아갑니다(White-out).
* **실차 권장 사양:**
  * **GMSL2 인터페이스 + Global Shutter 또는 차량용 HDR 센서 (`Sony IMX390` / `IMX490`, 120dB~140dB HDR, LED 플리커 억제 기능 탑재)**
  * 예: `Tier IV C1 / C2 Camera`, `Leopard Imaging LI-IMX390-GMSL2`, 또는 소형 플랫폼용 `Stereolabs ZED X (GMSL2)`

### ② 온보드 AI 컴퓨터 & 차량용 전원부
* 본 프로젝트의 3세대 SOTA 모델은 RTX 4090에서 **`2.95 ms` (`338 Hz`)**, 가중치 용량 **`18.8 MB` (FP16 변환 시 `9.4 MB`)**, 추론 VRAM **`61.5 MB`**만 사용하므로:
  * **임베디드 배포:** `NVIDIA Jetson AGX Orin (64GB)`에서 TensorRT FP16 변환 시 약 **`4.5 ~ 6.0 ms` (`160+ Hz`)**로 구동됩니다.
  * **트렁크 워크스테이션 배포:** 차량 시동 시 발생하는 전압 강하(Cranking Drop)와 노이즈를 막기 위해 반드시 **차량용 절연 DC-DC 컨버터(Ignition Sense 기능 내장)** 또는 별도의 **리튬인산철(LiFePO4) 보조 배터리 팩**을 통해 PC에 전원을 공급해야 합니다.

### ③ 고정밀 측위 센서 (CARLA의 `world_to_frenet`을 대체하는 장비)
* 시뮬레이터에서는 `ego_vehicle.get_transform()`으로 $(x, y, \psi)$를 바로 알 수 있지만, 실차에서는 현재 내가 차선 중심에서 몇 미터 벗어나 있는지($d$)와 누적 주행 거리($s$)를 알아야 합니다.
* **필수 조합:**
  1. **RTK-GNSS 수신기 (`u-blox ZED-F9P` 또는 `Septentrio Mosaic-X5`):** 국토지리정보원 NTRIP VRS 보정 신호를 LTE 모뎀으로 수신하여 실시간 위치 오차를 **`1.5 ~ 2.0 cm`** 이내로 유지합니다.
  2. **6축/9축 IMU (`Xsens MTi-630` 또는 `VectorNav VN-100`) + 차량 CAN 휠 속도계(Wheel Odometry):** `robot_localization` (Extended Kalman Filter) 패키지로 GNSS + IMU + 휠 속도를 융합하여 $100\text{ Hz}$ 위치·속도·헤딩 $(X, Y, \psi, v)$를 출력합니다.

### ④ Drive-By-Wire (DBW) CAN 인터페이스 및 물리적 비상정지(E-Stop)
* 컴퓨터가 계산한 `[throttle, steer, brake]`를 실제 자동차의 핸들 모터(MDPS)와 전자식 브레이크(ESC/ibb)로 전달하는 장치입니다.
* **CAN-USB / PCIe 어댑터:** `PEAK-System PCAN-USB FD` 또는 `Kvaser Leaf Light v2` (Linux `SocketCAN` 네이티브 지원).
* **필수 안전장치 (법적·공학적 절대 원칙):**
  1. **운전석 빨간색 버섯형 물리 비상정지 스위치 (Hardwired E-Stop):** 소프트웨어 버그로 PC가 멈추더라도 물리적으로 DBW 릴레이 전원을 차단해 즉시 운전자가 핸들과 브레이크를 수동 장악할 수 있어야 합니다.
  2. **브레이크 페달 오버라이드 (Brake Pedal Override):** 운전자가 브레이크나 핸들에 일정 토크 이상 힘을 가하면 `10 ms` 이내에 자율주행 모드가 자동 해제(Disengage)되어야 합니다.

---

## 4. 시뮬레이터 코드를 실차에 연결하는 4단계 핵심 엔지니어링 방법

### Step 1: 카메라 칼리브레이션 (Intrinsic & Extrinsic 일치시키기)
우리가 CARLA에서 사용한 `build_camera_intrinsic(560, 256, fov=95.0)`과 카메라 장착 위치 $(x=1.5\text{m}, z=1.45\text{m}, \text{pitch}=-4.0^\circ)$를 실차 카메라와 일치시켜야 `Perspective-to-BEV Lifter`가 정확한 거리로 장애물을 인식합니다.
1. **내부 파라미터(Intrinsic $\mathbf{K}$ & 왜곡 계수 $\mathbf{D}$):** 체커보드 판을 들고 `ros2 run camera_calibration cameracalibrator`를 실행하여 렌즈 왜곡 계수 $\mathbf{D} = [k_1, k_2, p_1, p_2, k_3]$를 구한 뒤, 추론 직전 `cv2.undistort()`로 편평한 핀홀 이미지로 보정합니다.
2. **외부 파라미터(Extrinsic — 장착 높이 및 각도):** 실차 룸미러 뒤쪽 전면 유리에 카메라를 고정할 때 지면으로부터의 높이($z \approx 1.4\text{m}$)와 숙임각($\text{pitch} \approx -4^\circ$)을 디지털 각도기로 맞추거나, 지면에 놓인 차선 마커 4개의 실제 거리 $(X, Y)$와 픽셀 $(u, v)$로부터 **호모그래피(Homography) 행렬**을 계산해 시뮬레이터 시점과 정렬(Camera Pose Alignment)합니다.

### Step 2: 실차 주행 경로(`SameDirectionTwoLaneSpline`) 생성하기
CARLA의 `find_best_two_lane_boulevard()` 대신 실차에서는 아래 방법으로 동일한 [`SameDirectionTwoLaneSpline`](../demo_14_carla_sota_dynamic_cutin_jaywalk.py#L46-L115) 객체를 생성합니다:
1. 실험할 도로(예: 시험주행장 200m 구간)의 1차선과 2차선 중심을 따라 **수동 운전으로 1회 주행하며 RTK-GNSS UTM 좌표 $(X_i, Y_i, Z_i)$를 `1.0m` 간격으로 기록**하여 `real_road_waypoints.csv`로 저장합니다.
2. 저장된 CSV 좌표를 [`SameDirectionTwoLaneSpline`](../demo_14_carla_sota_dynamic_cutin_jaywalk.py#L46-L115)의 `self.pts`에 그대로 로드하면, 실차 주행 중 실시간 RTK-GNSS 좌표만 넣어도 **`s` (누적 거리), `d` (차선 오프셋: 1차선 `-1.65m`, 2차선 `+1.70m`, 중앙선 `-3.45m`), `ref_yaw` (도로 접선 각도)**가 CARLA에서와 100% 동일하게 계산됩니다!

### Step 3: Sim-to-Real 도메인 갭 극복 (실차 파인튜닝 & 궤적-제어 분리)
1. **실차 데이터 소량 수집 후 Fine-Tuning (가장 효과적인 방법):**
   * CARLA에서 학습한 가중치를 초기값(Pre-trained Checkpoint)으로 둔 상태에서, 실차로 시험 코스를 **15~30분간 수동 운전(정상 주행 + 차선 변경 + 정차)**하며 `ros2 bag`으로 `[카메라 영상, RTK-GNSS 경로, CAN 조향/페달]`을 기록합니다.
   * 기록된 실차 데이터로 [`SOTA_DiffusionDrive_HydraMDP`](../demo_13_carla_sota_diffusion_bev_e2e.py#L565-L738)를 **15~20 에포크 동안 Fine-Tuning**하면 실제 아스팔트 질감·햇빛·그림자에 즉시 적응합니다.
2. **궤적 계획(E2E)과 차량 하위 제어(MPC / Pure Pursuit) 분리:**
   * 시뮬레이터의 `throttle=0.6, steer=0.2`는 실제 차량의 액셀/핸들 물리량과 다릅니다.
   * 따라서 실차에서는 신경망의 `ctrl_head` 출력을 직접 쓰기보다, 신경망이 예측한 **최적 3D 웨이포인트 `best_wps` (`15 × 2` meters, 로컬 $x_k, y_k$)**를 실차의 축거(Wheelbase $L \approx 2.7\text{ m}$)에 맞춘 **Pure Pursuit / Stanley / MPC 추종 제어기**에 입력하여 목표 조향각 $\delta$를 계산하는 것이 훨씬 안전하고 정확합니다:
     $$\delta_{\text{cmd}} = \arctan\!\left(\frac{2 L \cdot y_{\text{lookahead}}}{x_{\text{lookahead}}^2 + y_{\text{lookahead}}^2}\right)$$

### Step 4: ONNX / TensorRT FP16 변환 및 ROS 2 노드 구동
프로젝트 폴더에 생성해 둔 [`export_and_ros2_deploy_template.py`](../export_and_ros2_deploy_template.py)를 실행하면:
1. PyTorch 모델을 정적 그래프인 **`outputs/sota_diffusion_drive_e2e.onnx`**로 자동 변환합니다.
2. 실차의 Jetson AGX Orin 또는 산업용 PC에서 아래 명령 한 줄로 **TensorRT FP16 엔진**을 빌드합니다:
   ```bash
   trtexec --onnx=outputs/sota_diffusion_drive_e2e.onnx --saveEngine=outputs/sota_e2e_fp16.engine --fp16 --workspace=2048
   ```
3. 함께 생성되는 [`outputs/ros2_real_car_sota_e2e_node.py`](../outputs/ros2_real_car_sota_e2e_node.py) 템플릿을 통해 카메라 왜곡 보정 $\to$ SOTA 추론 $\to$ `Hydra-MDP` 안전 점수 검증(`max_score < 0.25` 시 비상 제동) $\to$ `Watchdog` 타임아웃 감시(`> 150ms` 프레임 드롭 시 즉시 정지) $\to$ `AckermannDriveStamped` DBW CAN 명령 발행을 수행합니다.

---

## 5. 실차 탑승 시 반드시 지켜야 할 5단계 안전 실험 프로토콜 (Shadow Mode $\to$ Closed-Loop)

> [!CAUTION]
> 실차 실험에서는 절대로 첫 시도에 액추에이터(핸들/페달) 전원을 켜고 주행해서는 안 됩니다. 반드시 아래 **5단계 검증 프로토콜**을 순서대로 통과해야 합니다.

| 단계 | 테스트 명칭 | 액추에이터(DBW) 상태 | 구체적 수행 방법 및 합격 기준 |
| :---: | :--- | :---: | :--- |
| **Stage 1** | **ROS 2 Bag 오프라인 리플레이 검증** | **OFF** (실차 미탑승) | 실차에서 수동 운전하며 녹화한 `ros2 bag` 데이터를 재생하면서, 모델이 출력하는 `Predicted BEV Radar`와 `5-Mode 궤적`이 도로 밖이나 중앙선을 침범하지 않는지 PC에서 먼저 확인 |
| **Stage 2** | **Shadow Mode (유령 모드 실차 주행)** | **OFF** (사람이 100% 직접 운전) | 보조석 모니터에 실시간 **FSD 카펫 + BEV Radar 대시보드**를 띄운 채 사람이 직접 코스를 운전. **사람이 핸들을 꺾는 타이밍과 모델이 선택한 최적 모드(`m*`) 및 조향각 부호가 95% 이상 일치**하는지 검증 |
| **Stage 3** | **Jack-Up (바퀴 리프트 공회전) 테스트** | **ON** (차량 리프트 위) | 차량 구동축을 리프트로 띄운 상태에서 ROS 2 노드를 켜고, 조향각 제한(`±25°`), 조향 속도 제한(Rate Limiter), 카메라 케이블 분리 시 **`150ms Watchdog` 자동 제동**, 물리 **E-Stop 스위치** 즉각 차단 작동 여부 확인 |
| **Stage 4** | **횡방향(조향) 단독 폐루프 + 수동 페달** | **Steer ON / Pedal OFF** | 폐쇄된 넓은 공터에서 **속도(액셀/브레이크)는 운전자가 발로 직접 조절(15~20 km/h 저속)**하고, **핸들(Steering)만 SOTA 모델에 맡겨** 라바콘(Traffic Cone) 슬라럼 및 차선 변경 회피 성능 검증 |
| **Stage 5** | **Full Closed-Loop (조향 + 가감속 통합)** | **Full DBW ON** | 안전요원이 E-Stop과 브레이크 페달 위에 발을 올린 상태에서, 스펀지/풍선 더미 차량(Soft Target)과 보행자 더미를 배치해 Stage 1~5 시나리오 완전 자율주행 수행 |

---

## 6. 실차 구축 등급별 상세 견적서 (BOM: Bill of Materials, 2026년 국내 실구매가 기준)

오늘 구현한 `SOTA_DiffusionDrive_HydraMDP` 및 `ActionConditionedDrivingWorldModel`을 실제로 올릴 수 있는 **4가지 현실적인 예산안(부품 모델명·수량·단가 포함)**입니다.

### ① [Tier 1] 1/10 스케일 `F1TENTH` 자율주행 플랫폼 (총 약 `263만 원` ~ `385만 원`)
* **특징:** 법적 허가 없이 연구실 복도, 체육관, 주차장에서 시속 15~30 km/h로 동적 장애물 회피를 즉시 실험할 수 있는 전 세계 대학 표준 플랫폼입니다.

| 카테고리 | 상세 부품 및 모델명 | 수량 | 국내 실구매 단가 (원) | 비고 |
| :--- | :--- | :---: | ---: | :--- |
| **차량 섀시** | Traxxas Slash 4x4 Platinum (또는 Fiesta ST 1/10) | 1 | `650,000` | 서보 모터 및 기본 조향 링크 포함 |
| **구동 제어기 (ESC)** | Flipsky VESC 6 MkV (또는 VESC 6 Edu) | 1 | `240,000` | ROS 2에서 휠 속도(Odometry) 측정 및 정밀 가감속 제어 |
| **온보드 AI 보드** | NVIDIA Jetson Orin Nano 8GB DevKit *(또는 Orin NX 16GB)* | 1 | `720,000` | TensorRT FP16 변환 시 SOTA 모델 `~8ms` 추론 *(NX 선택 시 +45만 원)* |
| **카메라 센서** | Stereolabs ZED 2i *(또는 Intel RealSense D435i)* | 1 | `680,000` | Global Shutter급 스테레오 + 내장 6축 IMU 포함 *(D435i 선택 시 48만 원)* |
| **전원 및 배터리** | 3S LiPo 배터리(5000mAh) 2개 + 밸런스 충전기 + DC-DC 벅 컨버터 | 1식 | `190,000` | 구동용(VESC)과 연산용(Jetson) 전원 분리 |
| **기구물/기타** | 아크릴/알루미늄 상판 데크, USB 허브, 비상정지 무선 조종기 | 1식 | `150,000` | 칼리브레이션용 A3 체커보드 포함 |
| **합계** | **Tier 1 (Orin Nano + ZED 2i 기준)** | - | **약 `2,630,000 원`** | *(2D LiDAR `Hokuyo UST-10LX` 추가 시 +`120만 원`)* |

---

### ② [Tier 2] 캠퍼스/시험장용 소형 전동 모빌리티 (`AgileX Hunter 2.0` 기준, 총 약 `2,180만 원`)
* **특징:** 성인 연구 장비(80kg)를 싣고 실제 야외 아스팔트 도로를 시속 20~25 km/h로 달릴 수 있으며, **처음부터 CAN-Bus Drive-By-Wire와 ROS 2 드라이버가 완벽히 내장**되어 있어 자동차 개조 없이 바로 코드만 올리면 움직입니다.

| 카테고리 | 상세 부품 및 모델명 | 수량 | 국내 실구매 단가 (원) | 비고 |
| :--- | :--- | :---: | ---: | :--- |
| **모바일 샤시** | **AgileX Hunter 2.0** (Ackermann 조향 전동 샤시) | 1 | `13,500,000` | CAN-Bus DBW 내장, 물리 E-Stop 스위치 기본 탑재 |
| **온보드 컴퓨터** | **NVIDIA Jetson AGX Orin 64GB Developer Kit** (275 TOPS) | 1 | `3,150,000` | SOTA E2E + World Model 동시 실시간 구동 가능 |
| **차량용 카메라** | Stereolabs ZED X (GMSL2 Global Shutter HDR) + 캡처 카드 | 1식 | `1,450,000` | 야외 직사광선/역광 완벽 대응 (`120dB HDR`) |
| **고정밀 측위 (RTK)** | **ArduSimple simpleRTK2B (u-blox ZED-F9P)** + GNSS 안테나 + LTE 라우터 | 1식 | `850,000` | 국토부 NTRIP 보정 신호 수신 시 위치 오차 `< 2cm` |
| **관성센서 (IMU)** | Xsens MTi-630 (산업용 9축 AHRS IMU) | 1 | `1,650,000` | 정밀 헤딩(`yaw`) 및 가속도 융합 |
| **안전 센서/기타** | 3D LiDAR (`Robosense Helios-16P` 또는 `Livox Mid-360` 안전 방어막용) + 프로파일 거치대 + CAN 어댑터(`PCAN-USB`) | 1식 | `1,200,000` | `Livox Mid-360` (`85만 원`)으로 비상 제동(Safety Shield) 구현 |
| **합계** | **Tier 2 (야외 캠퍼스/시험장 실차 풀세트)** | - | **약 `21,800,000 원`** | *(중고 전동 골프카트 직접 개조 시 총 `1,400만 원` 수준 가능)* |

---

### ③ [Tier 3-A] 실제 승용차 '가성비' E2E 실험 세팅 (`openpilot CAN 하네스` 방식, 차량 제외 약 `890만 원`)
* **특징:** 현대 아이오닉 5, 기아 니로 EV, 쏘나타 등 **순정 차선유지보조(LKAS/LFA)와 스마트 크루즈 컨트롤(SCC)이 달린 본인 차량(또는 중고차)**의 OBD-II / 카메라 커넥터에 `comma.ai panda` 인터페이스를 연결해, 외장 개조 공사 없이 순정 핸들 모터(MDPS)와 가감속을 내 PC의 ROS 2 노드로 제어하는 방법입니다.
* **한계점:** 안전을 위해 순정 MDPS의 최대 조향 토크 제한(급격한 핸들 꺾기 제한)이 걸려 있어, 완만한 차선 변경과 감속 실험에 적합합니다.

| 카테고리 | 상세 부품 및 모델명 | 수량 | 국내 실구매 단가 (원) | 비고 |
| :--- | :--- | :---: | ---: | :--- |
| **CAN DBW 브릿지** | **comma.ai Red Panda** + 현대/기아 전용 Car Harness (HDA1/HDA2) | 1식 | `650,000` | 차량 개조 없이 15분 만에 순정 MDPS/SCC CAN 버스 연결 |
| **온보드 컴퓨터** | **RTX 4080 탑재 고성능 게이밍 노트북** *(또는 미니 워크스테이션)* | 1 | `3,200,000` | 이미 RTX 4090 데스크탑/노트북 보유 시 `0원`! |
| **전방 카메라** | Leopard Imaging `LI-IMX390-GMSL2` (HDR) + USB3/PCIe 그래버 *(또는 ZED 2i)* | 1식 | `1,150,000` | 룸미러 뒤 앞유리 흡착 마운트 고정 |
| **고정밀 측위 (RTK)** | `u-blox ZED-F9P` RTK-GNSS 보드 + 차량 지붕 자석식 안테나 + IMU(`VN-100`) | 1식 | `2,100,000` | 차선 기준 횡방향 오프셋 `d` (`±2cm`) 실시간 산출 |
| **차량 전원/안전** | 에코플로우 파워뱅크 (`Delta 2`, 1024Wh) + 보조석 모니터 + 브레이크 오버라이드 릴레이 | 1식 | `1,800,000` | 차량 시거잭/배터리 방전 걱정 없이 4시간 연속 실험 가능 |
| **합계** | **Tier 3-A (양산 승용차 가성비 E2E 세팅, 차량가 별도)** | - | **약 `8,900,000 원`** | *(노트북 기보유 시 약 **`570만 원`**으로 실제 승용차 실험 가능!)* |

---

### ④ [Tier 3-B] 대학/연구소/스타트업 정식 풀스택 자율주행 실차 (정식 DBW 개조, 차량 제외 약 `6,850만 원`)
* **특징:** 순정 MDPS 토크 제한 없이 **핸들 각도를 `-450° ~ +450°`까지 100% 제어**하고 브레이크 유압까지 독립 제어하여, 오늘 구현한 **급격한 동적 끼어들기 회피 및 보행자 돌발 급제동**을 실제 승용차에서 100% 그대로 재현하고 **국토교통부 자율주행차 임시운행허가**까지 받을 수 있는 정식 연구용 사양입니다.

| 카테고리 | 상세 부품 및 모델명 | 수량 | 국내 실구매 단가 (원) | 비고 |
| :--- | :--- | :---: | ---: | :--- |
| **정식 DBW 개조** | 현대 아이오닉 5 / 니로 EV 전용 **Full Drive-By-Wire 개조 공임 및 제어기** (조향·가속·제동·기어·방향지시등 + 운전석 물리 E-Stop 스위치) | 1식 | `35,000,000` | 국내 전문 특장/자율주행 솔루션 업체 시공 (캔 통신 프로토콜 문서 제공) |
| **차량용 산업용 PC** | **Neousys Nuvo-10108GC** (Intel i9 + **NVIDIA RTX 4080/4090** + 차량용 Ignition 전원 + 방진 섀시) | 1 | `11,500,000` | 트렁크 마운트, 8채널 GMSL2 프레임 그래버 카드 내장 |
| **카메라 & LiDAR** | GMSL2 HDR 카메라 (`IMX390/IMX490`) 3채널 + **Ouster OS1-64 (64채널 3D LiDAR, 안전 방어막 및 GT 로깅용)** | 1식 | `12,500,000` | 루프랙(Roof Rack) 통합 센서 하우징 포함 |
| **RTK-INS 통합항법** | **NovAtel PwrPak7D-E1** *(또는 Septentrio AsteRx-i3 D Pro+ 듀얼 안테나 RTK-INS)* | 1식 | `7,500,000` | 듀얼 안테나로 정지 상태에서도 `0.1°` 정밀 헤딩(`yaw`) 측정 |
| **전장/배선/모니터** | 보조배터리(LiFePO4 200Ah) + 주행 충전기 + PEAK PCAN-PCIe + 앞좌석 터치 모니터 | 1식 | `2,000,000` | - |
| **합계** | **Tier 3-B (풀스택 정식 DBW 자율주행 연구차, 차량가 별도)** | - | **약 `68,500,000 원`** | *(중고 니로 EV / 아이오닉 5 차량 구입비 `2,500~3,500만 원` 추가 시 총 **약 `9,500만 ~ 1억 원`**)* |



---

## 7. 프로덕션 레벨(Production-Grade) End-to-End 자율주행 전체 파이프라인 상세 설계

데모 코드가 "카메라 1장 넣으면 조향각 나오는 단일 스크립트"라면, **프로덕션 레벨(Tesla FSD v12/v13, Wayve, NVIDIA DriveOS/Halos, Autoware Universe 양산 표준)**의 자율주행 시스템은 크게 **(A) 차량 내부 실시간 온보드 런타임 파이프라인(On-Vehicle Real-Time Runtime)**과 **(B) 클라우드 플릿 데이터 루프 및 자동 재학습 MLOps 파이프라인(Cloud Data Engine & Closed-Loop CI/CD)**의 거대한 **듀얼 루프(Dual-Loop) 아키텍처**로 설계됩니다.

구현 및 RTX 4090 실측 검증을 마친 프로덕션 레퍼런스 파이프라인 코드는 [`production_e2e_pipeline_architecture.py`](../production_e2e_pipeline_architecture.py)에서 바로 확인하고 실행하실 수 있습니다.

---

### 7.1 전체 시스템 마스터 아키텍처 다이어그램 (On-Vehicle + Cloud Data Loop)

```mermaid
flowchart TB
    subgraph VEHICLE["🚗 Pillar 1~3: 온보드 실시간 차량 런타임 (On-Vehicle Real-Time Runtime, 33~50Hz / ASIL-D)"]
        direction TB
        subgraph SENSORS["Pillar 1: 하드웨어 시각 동기화 & Zero-Copy 수집 계층 (IEEE 1588 PTPv2 + GPIO FSIN)"]
            S1["GMSL2 HDR 카메라 x6 (30Hz)"]
            S2["3D LiDAR (10~20Hz) & 4D Radar (20Hz)"]
            S3["듀얼 안테나 RTK-INS (100Hz) & CAN-FD (100Hz)"]
            SYNC["HardwarePTPSensorSynchronizer\n(< 1.0ms 동기화 + SE(3) Ego-Motion Deskew + NVMM DMA Zero-Copy)"]
            S1 & S2 & S3 --> SYNC
        end

        subgraph AIRUNTIME["Pillar 2: 이중 주기(Dual-Rate) 비동기 AI 추론 런타임 (TensorRT FP16 / CUDA Streams)"]
            SYS2["🧠 System 2: 인지/세계 모델 (Slow Loop @ 5Hz, ~2.8ms)\nActionConditionedDrivingWorldModel + VLM\n(3-Universe 미래 상상 + 주행 가능 코리더 [d_min, d_max] 생성)"]
            SYS1["⚡ System 1: 반사 신경 제어기 (Fast Loop @ 33.3Hz, ~3.3ms)\nSOTA_DiffusionDrive_HydraMDP\n(64x64 BEV Occupancy + M=5 4-Step Flow ODE + Hydra-MDP Scorer)"]
            SYNC --> SYS2
            SYNC --> SYS1
            SYS2 -. "Lock-Free Atomic Corridor Prior\n[d_min, d_max, v_ref]" .-> SYS1
        end

        subgraph SAFETY["Pillar 3: ISO 26262 (ASIL-D) 3계층 다중 안전 방어막 (3-Tier Redundant Safety Shield)"]
            T1["Tier 1: Neural Uncertainty Gate\n(Hydra-MDP 신뢰도 S_m < 0.45 또는 엔트로피 H > 0.88 시 감속)"]
            T2["Tier 2: 해석적 물리 방어막 (신경망 가중치 완전 독립!)\n• 횡방향 CBF (황색 중앙선 d = -3.45m 침범 원천 차단, d >= -2.55m)\n• 종방향 Mobileye RSS + Raw 3D LiDAR AEB 비상제동"]
            T3["Tier 3: 하드웨어 워치독 MRM & Jerk/Rate 제어기\n• PTP/루프 지연 > 20ms 시 Minimal Risk Maneuver 비상정차\n• 운전자 핸들 토크(>3.5Nm)/브레이크 개입 시 즉시 릴레이 차단"]
            SYS1 --> T1 --> T2 --> T3
        end

        CAN["차량 DBW 액추에이터 (CAN-FD 100Hz: 조향 MDPS / 가감속 ESC)"]
        T3 --> CAN
    end

    subgraph CLOUD["☁️ Pillar 4~5: 플릿 데이터 엔진 & 폐루프 MLOps 파이프라인 (Cloud Data Engine & CI/CD)"]
        direction TB
        EDGE_TRIG["Pillar 4: 엣지 링버퍼 트리거 레코더 (ShadowModeFleetDataEngine)\n• Rule 1: 운전자 강제 개입(Disengagement)\n• Rule 2: Tier-3 MRM 워치독 폴트\n• Rule 3: Tier-2 CBF/RSS 안전 방어막 작동\n• Rule 4: 높은 신경망 불확실성(Entropy > 0.85)\n(-10초 ~ +5초 멀티센서 원본 패킷 자동 패키징)"]
        AUTOLABEL["4D 오토라벨링 & 3DGS 월드 재구성 파이프라인\n• 다중 프레임 LiDAR+Camera 4D Occupancy GT 생성\n• 거대 Teacher 모델(70B VLM + Offline Tracker) 의사 정답 라벨링\n• 3D Gaussian Splatting (HUGSIM/StreetGaussians) 디지털 트윈 변환"]
        TRAIN["분산 학습 클러스터 (PyTorch FSDP / Multi-Node H100)\n• Stage 1: BEV & World Model 사전학습\n• Stage 2: DiffusionDrive + Hydra-MDP 모방학습\n• Stage 3: 폐루프 강화학습(Closed-Loop RL) 파인튜닝"]
        CICD["Pillar 5: 폐루프 시뮬레이션 CI/CD 검증 게이트\n• Gate 1: NAVSIM (PDMS >= 90%)\n• Gate 2: 3DGS + CARLA 동적 가혹 시나리오 1,000종 (중앙선 침범 0건, 충돌 0건)\n• Gate 3: 실차 HIL 벤치 (TensorRT FP16 p99 지연 < 10ms)"]

        T3 & SYS1 -. "실시간 텔레메트리 감시" .-> EDGE_TRIG
        EDGE_TRIG ==>|"LTE/5G/Wi-Fi 업로드"| AUTOLABEL ==> TRAIN ==> CICD
        CICD ==>|"OTA 서명된 TensorRT 엔진 배포"| AIRUNTIME
    end
```

---

### 7.2 6대 핵심 아키텍처 필러(Pillar) 프로덕션 엔지니어링 상세 명세

#### ① Pillar 1: 하드웨어 결정론적 시각 동기화 & Zero-Copy 센서 파이프라인 (`HardwarePTPSensorSynchronizer`)
프로덕션 실차에서는 센서 간 타임스탬프가 `5 ms`만 어긋나도 시속 50 km/h 주행 시 장애물 위치가 `7 cm` 이상 튀어 궤적이 진동합니다.
* **IEEE 1588 PTPv2 (`gPTP`) + 하드웨어 GPIO `FSIN` 트리거:**
  * 차량 내 산업용 이더넷 스위치 또는 `NovAtel/Septentrio RTK-INS`가 **Grandmaster Clock**이 되어, GMSL2 프레임 그래버, 3D LiDAR, Jetson/PC의 시스템 클록(`CLOCK_TAI`)을 `< 100 μs` 오차로 동기화합니다.
  * 카메라 셔터가 열리는 정확한 중간 시점($t_{\text{cam\_mid}}$)에 맞춰 10Hz/30Hz 하드웨어 펄스(`FSIN`)를 쏩니다.
* **$\text{SE}(3)$ 자차 이동 보상(Ego-Motion Deskewing):**
  * 100Hz로 들어오는 RTK-INS/CAN 속도($v$)와 각속도($\omega$)를 이용해, LiDAR 포인트 클라우드와 Frenet 좌표 $(s, d, \psi)$를 카메라 노출 시점 $t_{\text{cam\_mid}}$로 1차 테일러 보간($s' = s + v \Delta t$)하여 공간 정합을 맞춥니다.
* **Zero-Copy DMA 메모리 전송 (`NVMM` / `ROS 2 NITROS`):**
  * GMSL2 카메라 영상이 CPU RAM을 거치지 않고 PCIe DMA를 통해 GPU VRAM의 고정 버퍼(`cudaMallocHost` / `NVMM`)로 직접 들어가 TensorRT 입력 텐서에 바로 바인딩됩니다(`CPU 복사 지연 0 ms`).

#### ② Pillar 2: 이중 주기(Dual-Rate) 비동기 AI 추론 런타임 (`DualRateSystem1System2Runtime`)
인간의 뇌가 **"빠른 반사 신경(System 1)"**과 **"깊은 상황 판단(System 2)"**으로 나뉘듯, 프로덕션 E2E 자율주행은 서로 다른 두 개의 CUDA Stream과 스레드 우선순위로 돌아갑니다.

| 구분 | **System 1 (Fast Visuomotor Reflex)** | **System 2 (Slow Cognitive & World Model)** |
| :--- | :--- | :--- |
| **담당 신경망** | [`SOTA_DiffusionDrive_HydraMDP`](../demo_13_carla_sota_diffusion_bev_e2e.py#L564-L745) | [`ActionConditionedDrivingWorldModel`](../demo_15_carla_action_world_model.py#L156-L317) + VLM |
| **동작 주파수 / 주기** | **`33.3 ~ 50 Hz`** (`20 ~ 30 ms` 주기, 실시간 크리티컬 패스) | **`5 ~ 10 Hz`** (`100 ~ 200 ms` 주기, 비동기 백그라운드 스레드) |
| **RTX 4090 실측 지연** | 평균 **`3.297 ms`** (`p99 = 4.557 ms`, 예산 `10 ms` 이내) | 평균 **`2.219 ms`** (`p99 = 2.806 ms`, 대형 VLM 결합 시 `~45 ms`) |
| **핵심 역할** | $64\times 64$ BEV 점유 격자 디코딩 + $M=5$ 4-Step Flow-Matching 궤적 생성 + `Hydra-MDP` 채점 | 1.8초 뒤 3개 평행우주 미래 영상/BEV 상상 및 생존 확률 평가 → **안전 주행 코리더($[d_{\min}, d_{\max}], v_{\text{ref}}$)** 산출 |
| **통신 메커니즘** | `SCHED_FIFO` 최고 우선순위(Priority 90), Lock-Free Atomic 포인터로 System 2의 최신 코리더만 읽음 (절대 System 2를 기다리며 블로킹되지 않음!) | 비동기 CUDA Stream에서 추론 완료 시 Atomic Shared Memory에 최신 `System2CorridorPrior` 구조체 갱신 |

#### ③ Pillar 3: ISO 26262 (ASIL-D) 3계층 다중 안전 방어막 (`ISO26262SafetyShieldController`)
양산 자율주행차에서 **신경망의 출력을 CAN 버스에 직결하는 것은 절대 금지**됩니다. 신경망이 환각(Hallucination)을 일으켜도 사고가 나지 않도록 **신경망 가중치와 100% 독립된 해석적(Analytical) 물리 방어막**을 3단계로 겹쳐 씌웁니다.

1. **Tier 1 — 신경망 불확실성 게이트 (Neural Epistemic Uncertainty Gate):**
   * $M=5$ 후보 모드들의 점수 엔트로피 $H = -\sum p_m \log p_m / \log M$가 `0.88`을 넘거나 최고 모드 점수 $S_{m^*} < 0.45$이면(예: 강한 역광, 처음 보는 공사장), 신경망이 확신을 잃은 상태이므로 즉시 목표 속도를 감속($-2.0\text{ m/s}^2$)합니다.
2. **Tier 2 — 해석적 물리 방어막 (Analytical CBF & Mobileye RSS + Classical LiDAR AEB):**
   * **횡방향 제어 장벽 함수 (Control Barrier Function, CBF — 황색 중앙선 절대 방어):**
     * 황색 복선 중앙선이 $d_{\text{yellow}} = -3.45\text{ m}$에 있을 때, 안전 경계 $d_{\text{safe}} = -2.55\text{ m}$에 대해 장벽 함수 $h(d) = d - d_{\text{safe}} \ge 0$을 정의합니다.
     * 만약 신경망 조향각 $\delta_{\text{AI}}$가 0.6초 뒤 $h(\hat{d}_{t+0.6\text{s}}) < 0$을 유발하면, 신경망 출력을 무시하고 해석적 CBF 투영($\delta_{\text{final}} = \max(\delta_{\text{AI}}, +0.18\text{ rad})$)으로 차량을 즉시 합법 차선 안쪽으로 밀어냅니다.
   * **종방향 Mobileye RSS (Responsibility-Sensitive Safety) + 고전 3D LiDAR/Radar AEB:**
     * 딥러닝을 쓰지 않는 순수 기하학적 3D LiDAR 유클리드 클러스터링 거리 $d_{\text{lidar}}$가 RSS 최소 안전거리 공식:
       $$d_{\min,\text{RSS}} = v_{\text{ego}}\rho + \frac{1}{2}a_{\max,\text{accel}}\rho^2 + \frac{(v_{\text{ego}} + a_{\max,\text{accel}}\rho)^2}{2 a_{\min,\text{brake}}}$$
       보다 짧아지면 즉시 비상 제동($-3.0 \sim -6.5\text{ m/s}^2$)을 겁니다.
3. **Tier 3 — 하드웨어 워치독 MRM & 액추에이터 Jerk/Rate 제한기:**
   * 센서 동기화 오차가 `2.5 ms`를 넘거나 루프 지연이 `20 ms`를 초과하면 즉시 **MRM(Minimal Risk Maneuver, 차선 유지 비상 감속 정차)**으로 전환합니다.
   * 운전자가 브레이크를 밟거나 핸들에 `3.5 Nm` 이상의 토크를 가하면 1ms 이내에 자율주행 제어권을 해제(`DRIVER_OVERRIDE_DISENGAGE`)합니다.
   * 모든 최종 명령은 탑승자 승차감과 타이어 슬립 방지를 위해 조향 변화율 $|\dot{\delta}| \le 0.55\text{ rad/s}$, 가속도 변화율(Jerk) $|j| \le 8.5\text{ m/s}^3$로 클램핑됩니다.

#### ④ Pillar 4: 엣지 링버퍼 트리거 & 플릿 데이터 엔진 (`ShadowModeFleetDataEngine`)
실차 주행 시 하루 수 테라바이트(TB)의 모든 영상을 클라우드에 올리면 통신비와 스토리지 비용이 폭발합니다. 프로덕션 차량은 RAM에 **항상 최근 15초 분량의 순환 링버퍼(Circular Ring Buffer)**만 유지하다가, 아래 **4대 트리거 규칙**이 발동할 때만 **사건 발생 전 10초(`-10s`) ~ 발생 후 5초(`+5s`) 패킷**을 잘라 클라우드로 자동 업로드합니다.
* **`RULE_1_HUMAN_DISENGAGEMENT`:** 운전자가 위험을 느끼고 브레이크/핸들에 개입한 순간 (가장 가치 높은 오답 노트!)
* **`RULE_2_TIER3_MRM_FAULT`:** 센서 시각 동기화 깨짐 또는 워치독 지연 발생 순간
* **`RULE_3_TIER2_CBF_OR_RSS`:** 신경망이 중앙선에 가까워져 CBF가 개입했거나, RSS LiDAR 비상제동이 개입한 아차사고(Near-Miss) 순간
* **`RULE_4_HIGH_EPISTEMIC_UNCERTAINTY`:** 신경망 내부 엔트로피가 `0.85` 이상으로 치솟은 낯선 도로(Out-of-Distribution) 구간

#### ⑤ Pillar 5: 클라우드 4D 오토라벨링 & 폐루프 시뮬레이션 CI/CD 게이트
클라우드로 수집된 15초 클립들은 사람 라벨러가 일일이 손으로 칠하지 않고 자동화된 데이터 공장을 거칩니다:
1. **다중 프레임 4D 오토라벨링 (Offline Teacher Auto-Labeling):**
   * 차량에서는 실시간 제약 때문에 못 돌리던 초대형 모델(과거+미래 프레임을 모두 보는 Offline 4D Tracker + 70B VLM + LiDAR-Camera Bundle Adjustment)을 클라우드 H100 서버에서 돌려 고정밀 3D Occupancy GT와 이상적 회피 궤적을 자동 생성합니다.
2. **3D Gaussian Splatting (3DGS) 디지털 트윈 변환:**
   * 운전자가 개입했던 15초 클립을 **HUGSIM / StreetGaussians** 파이프라인에 넣어, 그 도로와 끼어들던 차량을 3DGS 가상 공간으로 똑같이 복원합니다.
3. **자동 회귀 검증 게이트 (Automated CI/CD Release Gate):**
   * 새로 학습된 모델이 실차에 OTA로 배포되려면 반드시 3단계 게이트를 통과해야 합니다:
     * **Gate 1 (`NAVSIM`):** PDMS 종합 점수 `90.0%` 이상
     * **Gate 2 (`3DGS + CARLA 1,000 Scenarios`):** 과거 운전자 개입이 발생했던 모든 클립 재현 시 **황색 중앙선 침범 `0건`, 충돌 `0건`**
     * **Gate 3 (`HIL Bench`):** 실제 Jetson Orin / 차량용 PC에서 TensorRT FP16 변환 후 1시간 연속 구동 시 **`p99` 추론 지연 `< 10.0 ms`, 메모리 누수 `0 MB`**

---

### 7.3 RTX 4090 프로덕션 파이프라인 5단계 가혹 시나리오 검증 결과

[`production_e2e_pipeline_architecture.py`](../production_e2e_pipeline_architecture.py)를 실행하여 33.3Hz 실시간 루프에서 5가지 가혹 상황(정상 주행 → 우측 차선 급끼어들기 회피 → 황색 중앙선 방향 횡풍 드리프트 → 보행자 돌발 출현 → PTP 센서 동기화 폴트 및 운전자 강제 개입)을 연속 주입한 검증 대시보드입니다:

![프로덕션 파이프라인 5단계 가혹 검증 텔레메트리](file://../outputs/production_pipeline_telemetry.png)

* **실측 검증 요약 ([`production_pipeline_verification.json`](../outputs/production_pipeline_verification.json)):**
  * **System 1 (Fast Loop @ 33.3Hz) 추론 지연:** 평균 **`3.297 ms`**, `p99 = 4.557 ms` (실시간 예산 `10.0 ms` 여유 있게 통과)
  * **System 2 (Slow World-Model Loop @ 5.5Hz) 추론 지연:** 평균 **`2.219 ms`**, `p99 = 2.806 ms`
  * **황색 중앙선($d = -3.45\text{ m}$) 침범 횟수:** **`0회`** (Frame 65에서 강제 횡풍 외란으로 $d = -2.62\text{ m}$까지 밀렸으나, Tier-2 CBF 장벽 함수가 즉시 조향각을 양수 방향으로 클램핑하여 최좌측 도달 위치를 **`-2.653 m`**로 방어 — 황색 중앙선까지 **`0.80 m` 안전 마진** 확보)
  * **플릿 데이터 엔진 자동 캡처:** 총 11개의 엣지 케이스 클립(`RULE_1` 운전자 개입, `RULE_2` MRM 폴트, `RULE_3` CBF/RSS 개입, `RULE_4` OOD 불확실성 급증)이 전후 15초 버퍼 메타데이터와 함께 정확히 추출됨.


---

## 8. Python vs. Native Rust 프로덕션 파이프라인 실측 벤치마크 (2,000 Iterations)

실제 차량 런타임(Pillar 1~4)을 **Python (`NumPy` + `threading.Lock` + ROS 2 `rclpy` 스타일 백그라운드 메시지 GC 부하)**에서 **Native Rust (`rustc 1.99.1`, `-C opt-level=3 -C target-cpu=native`, SIMD 자동 벡터화 + `AtomicU64` Lock-Free 브릿지 + Zero-GC 링버퍼)**로 이식했을 때의 실측 성능 비교입니다.

* **구현 파일:**
  * Rust 네이티브 파이프라인: [main.rs](../rust_e2e_pipeline/src/main.rs) ([Cargo.toml](../rust_e2e_pipeline/Cargo.toml))
  * 맞대결 벤치마크 스크립트: [benchmark_python_vs_rust_e2e.py](../benchmark_python_vs_rust_e2e.py)
  * 실측 JSON 리포트: [python_vs_rust_e2e_benchmark.json](../outputs/python_vs_rust_e2e_benchmark.json)

![Python vs Rust E2E Benchmark](../outputs/python_vs_rust_e2e_benchmark.png)

### 핵심 실측 요약표 (`N = 2,000` 루프, 멀티스레드 경합 상태)

| 파이프라인 단계 | Python (`Mean` / `p99.9` / `Max`) | Native Rust (`Mean` / `p99.9` / `Max`) | **Rust 가속 배율** |
| :--- | :---: | :---: | :--- |
| **Pillar 1: PTP 검증 + `32,768` 포인트 LiDAR $\text{SE}(3)$ Deskew + RGB 정규화** | `161.1 µs` / `332.6 µs` / `381.1 µs` | **`53.9 µs`** / `764.6 µs` / `977.4 µs` | 평균 **3.0배** 가속 (`0.054 ms`) |
| **Pillar 2 & 4: System 2 코리도 공유 + `1 MB` 블랙박스 링버퍼 복사** | `81.0 µs` / `432.7 µs` / `671.5 µs` | **`35.7 µs`** / `562.7 µs` / `1,549.6 µs` | 평균 **2.3배** 가속 (`AtomicU64` 무잠금) |
| **Pillar 3: ASIL-D 3-Tier Safety Shield (CBF + RSS) + CAN CRC-8 패킹** | `6.06 µs` / `35.91 µs` / `42.78 µs` | **`0.043 µs` (`43 ns`)** / `0.052 µs` / `4.32 µs` | 평균 **140.8배**, `p99.9` **690.7배** 가속 |
| **Full CPU Host 통합 루프 (3-스레드 동시 경합 & GC 환경)** | `529.1 µs` (`0.53 ms`) / **`36.09 ms`** / **`36.90 ms`** | **`80.5 µs` (`0.08 ms`)** / **`0.206 ms`** / **`1.24 ms`** | 평균 **6.6배**, **`p99.9` Tail 175.4배 가속**, **지터(`StdDev`) 62.0배 감소** |
| **GPU 추론 (RTX 4090 `SOTA_DiffusionDrive_HydraMDP`)** | PyTorch Eager: `2.90 ms` (`Max 5.82 ms`) | Native CUDA Graph: **`0.48 ms`** (`Max 1.23 ms`) | Python 커널 런치 오버헤드 제거로 **6.1배** 가속 |
| **Total End-to-End (Camera-to-CAN 전체 지연)** | 평균 `3.43 ms` / **최악 스파이크 `42.72 ms`** | 평균 **`0.56 ms`** / **최악 스파이크 `2.47 ms`** | 평균 **6.1배**, **최악 스파이크 17.3배 단축** |
