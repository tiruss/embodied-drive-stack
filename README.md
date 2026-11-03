# End-to-End Robot Imitation Learning & Autonomous Driving Monorepo (MuJoCo $\rightarrow$ CARLA SOTA E2E $\rightarrow$ World Model $\rightarrow$ Native Rust Production Pipeline)

이 레포지토리는 **로봇 모방학습(MuJoCo + LeRobot)** 기초부터 시작하여 **CARLA 0.9.15 도심 자율주행 VLA / 3세대 SOTA Diffusion + BEV + Hydra-MDP E2E 아키텍처 / Action-Conditioned World Model (상상 기반 미래 예측)**, 그리고 최종적으로 **ISO 26262 ASIL-D 프로덕션 파이프라인 및 Native Rust 실시간 이식**까지의 전 과정을 단계별로 추적(Tracing)하고 재현할 수 있도록 통합된 단일 모노레포(Monorepo)입니다.

---

## 📁 레포지토리 디렉토리 구조

```text
mujoco-imitation-demo/
├── README.md                                     # 프로젝트 전체 아키텍처 및 단계별 추적 인덱스
├── docs/                                         # 상세 이론, 벤치마크, 실차 배포 및 예산 가이드 문서
│   ├── mujoco_imitation_study_guide.md           # [문서 1] Step 1 ~ Step 17 전 과정 이론 & 실습 스터디 가이드
│   └── real_vehicle_deployment_guide.md          # [문서 2] 실차 탑재, 캘리브레이션, 예산(BOM), 6-Pillar 프로덕션 설계 & Rust 벤치마크
├── check_env.py                                  # GPU (RTX 4090), PyTorch, MuJoCo, LeRobot 환경 검증
├── demo_1_interactive_viewer.py                  # [Phase 1] MuJoCo 3D 물리 엔진 & ALOHA 양팔 로봇 뷰어
├── demo_2_act_pretrained.py                      # [Phase 1] Action Chunking with Transformers (ACT) 추론
├── demo_3_train_mini_bc.py                       # [Phase 1] Behavior Cloning (BC) 학습 전/후 비교 & Temporal Ensembling
├── demo_4_bimanual_insertion.py                  # [Phase 1] 양팔 협업 Peg-in-Hole 정밀 삽입 제어
├── generate_study_visuals.py                     # [Phase 1] 모방학습 학습 곡선 및 스테이지별 스트립 시각화 생성
├── demo_5_diffusion_flow_policy.py               # [Phase 2] Flow-Matching / Diffusion Policy (다중 모드 확률 분포 학습)
├── demo_6_vla_language_conditioned.py            # [Phase 2] Vision-Language-Action (VLA) 자연어 명령 조건부 로봇 제어
├── demo_7_highway_vla_fsd.py                     # [Phase 3] 2D Highway-env 자율주행 FSD VLA
├── demo_8_mujoco_3d_car_flow.py                  # [Phase 3] MuJoCo 3D 차량 물리 엔진: L2 평균 회귀 충돌 vs Flow-Matching 회피
├── demo_9_carla_urban_vla.py                     # [Phase 4] CARLA 0.9.15 도심 4대 센서(RGB/Depth/Seg/LiDAR) 통합 대시보드
├── demo_10_carla_end_to_end_vla.py               # [Phase 4] CARLA Closed-Loop End-to-End VLA (3가지 주행 명령 수행)
├── demo_11_carla_pretrained_vlm_vla.py           # [Phase 4] HuggingFace 사전학습 CLIP + MiniLM Zero-Shot 미학습 명령어 주행
├── demo_12_carla_e2e_gen1_vs_gen2.py             # [Phase 5] 1세대 Black-Box (PilotNet) vs 2세대 Modular E2E (UniAD/VAD 스타일)
├── demo_13_carla_sota_diffusion_bev_e2e.py       # [Phase 5] 3세대 SOTA: Perspective-to-BEV + Truncated Flow-Matching + Hydra-MDP Scorer
├── benchmark_e2e_models.py                       # [Phase 5] RTX 4090 1세대 vs 2세대 vs 3세대 SOTA 하드웨어 정밀 벤치마크
├── demo_14_carla_sota_dynamic_cutin_jaywalk.py   # [Phase 6] 중앙선 침범 방지(Same-Direction 2차선) + 동적 칼치기/무단횡단/급정거 회피
├── demo_15_carla_action_world_model.py           # [Phase 6] Action-Conditioned World Model (3개 평행우주 미래 카메라/BEV 상상)
├── export_and_ros2_deploy_template.py            # [Phase 7] SOTA 모델 Static ONNX 변환 및 ROS 2 Humble/Jazzy 실차 CAN 노드 생성
├── production_e2e_pipeline_architecture.py       # [Phase 7] 6-Pillar 프로덕션 아키텍처 (PTP 동기화 + Dual-Rate + ASIL-D CBF/RSS + 블랙박스)
├── benchmark_python_vs_rust_e2e.py               # [Phase 8] Python vs Native Rust + CUDA Graph 프로덕션 파이프라인 맞대결 벤치마크
├── rust_e2e_pipeline/                            # [Phase 8] Native Rust 초저지연(Zero-GC, SIMD, Lock-Free Atomic) 차량 런타임
│   ├── Cargo.toml
│   └── src/main.rs
└── outputs/                                      # 모든 데모 실행 영상(.mp4), 텔레메트리 차트(.png), JSON 리포트, ONNX 가중치
```

---

## 🔍 Git 커밋 히스토리 기반 단계별 추적 (Tracing Guide)

이 레포지토리는 기술 발전 단계별로 커밋이 분리되어 있어 `git log --oneline --graph` 명령으로 각 단계의 코드 변경점과 결과물을 완벽히 추적할 수 있습니다.

1. **`Phase 1`**: MuJoCo ALOHA 모방학습 기초 (ACT, BC, Temporal Ensembling, Bimanual Insertion)
2. **`Phase 2`**: 생성형 행동 정책 (Flow-Matching ODE) 및 언어 조건부 VLA 로봇 제어
3. **`Phase 3`**: 로봇 팔에서 자율주행으로의 전이 (`highway-env` 2D & MuJoCo 3D 차량 L2 vs Flow-Matching)
4. **`Phase 4`**: CARLA 0.9.15 도심 주행 센서 퓨전 및 사전학습 VLM (CLIP + MiniLM) Zero-Shot VLA
5. **`Phase 5`**: 자율주행 E2E 세대별 비교 (Gen1 PilotNet vs Gen2 UniAD/VAD vs Gen3 SOTA DiffusionDrive + Hydra-MDP) 및 RTX 4090 하드웨어 벤치마크
6. **`Phase 6`**: 동적 돌발 시나리오 (칼치기/보행자 무단횡단/선행차 급정거) 및 Action-Conditioned World Model (미래 RGB/BEV Dreaming)
7. **`Phase 7`**: 실차 배포용 Static ONNX 추출, ROS 2 DBW 노드, 그리고 6-Pillar ISO 26262 ASIL-D 프로덕션 파이프라인
8. **`Phase 8`**: Native Rust (`rustc 1.99.1`, SIMD + Lock-Free `AtomicU64` + Zero-GC) 포팅 및 Python 대비 실시간 Latency/Jitter 맞대결 벤치마크

---

## 🚀 빠른 실행 방법 (WSL2 Ubuntu + RTX 4090)

```bash
# 1. WSL2 가상환경 활성화
source /home/tiruss/.venvs/mujoco-lerobot/bin/activate
cd /mnt/c/Users/tirus/.gemini/antigravity/scratch/mujoco-imitation-demo

# 2. 6-Pillar 프로덕션 파이프라인 검증 실행
python production_e2e_pipeline_architecture.py

# 3. Native Rust 빌드 및 Python vs. Rust 맞대결 벤치마크 실행
cd rust_e2e_pipeline && RUSTFLAGS="-C target-cpu=native" ~/.cargo/bin/cargo build --release && cd ..
python benchmark_python_vs_rust_e2e.py
```

---

## 📚 핵심 문서 바로가기
* 📖 **[로봇 모방학습 ~ 자율주행 World Model 종합 스터디 가이드 (`docs/mujoco_imitation_study_guide.md`)](docs/mujoco_imitation_study_guide.md)**
* 🚗 **[실차 탑재, 캘리브레이션, 예산(BOM), 6-Pillar 프로덕션 아키텍처 & Rust 벤치마크 가이드 (`docs/real_vehicle_deployment_guide.md`)](docs/real_vehicle_deployment_guide.md)**
