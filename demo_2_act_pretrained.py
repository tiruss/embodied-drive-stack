"""
Demo 2: Pre-trained ACT (Action Chunking with Transformers) on MuJoCo ALOHA
===========================================================================
최신 양팔 로봇 모방학습의 대표 모델인 ACT (Action Chunking with Transformers)를
RTX 4090 GPU에서 추론하여 '한쪽 팔로 큐브를 집어 반대쪽 팔에 공중 전달(Transfer Cube)'하는 데모입니다.

실행 방법:
  1) MP4 고화질 영상으로 저장 (기본):
     MUJOCO_GL=egl ~/.venvs/mujoco-lerobot/bin/python demo_2_act_pretrained.py

  2) 실시간 화면 창(OpenCV GUI)까지 함께 띄우며 실행:
     MUJOCO_GL=egl ~/.venvs/mujoco-lerobot/bin/python demo_2_act_pretrained.py --live
"""
import argparse
import time
from pathlib import Path

import cv2
import gymnasium as gym
import gym_aloha  # noqa: F401
import imageio
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file


def load_norm_stats(model_id: str, device: torch.device) -> dict[str, torch.Tensor]:
    """체크포인트(model.safetensors) 내에 저장된 입력/출력 정규화 통계량(mean, std)을 로드합니다."""
    safetensors_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
    sd = load_file(safetensors_path, device=str(device))
    return {
        "img_mean": sd["normalize_inputs.buffer_observation_images_top.mean"],
        "img_std": sd["normalize_inputs.buffer_observation_images_top.std"],
        "state_mean": sd["normalize_inputs.buffer_observation_state.mean"],
        "state_std": sd["normalize_inputs.buffer_observation_state.std"],
        "action_mean": sd["unnormalize_outputs.buffer_action.mean"],
        "action_std": sd["unnormalize_outputs.buffer_action.std"],
    }


def format_and_normalize_obs(obs: dict, stats: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """gym_aloha의 관측값(카메라 이미지 + 14축 관절 각도)을 정규화된 LeRobot ACT 입력 텐서로 변환합니다."""
    # 1. 로봇 관절 상태: (14,) -> (1, 14) float32 후 평균/표준편차 정규화
    state = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device)
    state = (state - stats["state_mean"]) / stats["state_std"]

    # 2. 상단 카메라 이미지: (H, W, C) uint8 [0..255] -> (1, C, H, W) float32 [0..1] 후 ImageNet 정규화
    img = torch.from_numpy(obs["pixels"]["top"]).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    img = (img - stats["img_mean"]) / stats["img_std"]

    return {
        "observation.state": state,
        "observation.images.top": img,
    }


def add_hud_overlay(frame_rgb: np.ndarray, step: int, max_steps: int, reward: float, success: bool, fps: float) -> np.ndarray:
    """영상 프레임 위에 현재 스텝, 보상, 성공 여부, GPU 추론 FPS를 표시합니다."""
    canvas = frame_rgb.copy()
    status_text = "SUCCESS! (Cube Transferred)" if success else "In Progress..."
    status_color = (40, 220, 80) if success else (255, 215, 0)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 72), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

    cv2.putText(
        canvas,
        f"ACT Policy | AlohaTransferCube-v0 | Step: {step:03d}/{max_steps} | {fps:.1f} FPS",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Reward: {reward:.1f} | Status: {status_text}",
        (14, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        status_color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Run pre-trained ACT policy on MuJoCo ALOHA")
    parser.add_argument("--model-id", type=str, default="lerobot/act_aloha_sim_transfer_cube_human")
    parser.add_argument("--steps", type=int, default=400, help="에피소드 최대 스텝 수 (기본 400 = 8초)")
    parser.add_argument("--seed", type=int, default=1, help="랜덤 시드 (큐브 초기 위치 결정)")
    parser.add_argument("--replan-steps", type=int, default=20, help="몇 스텝마다 카메라를 다시 보고 궤적을 수정할지 (기본 20)")
    parser.add_argument("--live", action="store_true", help="실시간 OpenCV 창으로 로봇 시점 보기")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"[1/4] 디바이스 설정: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    print(f"[2/4] Hugging Face에서 사전 학습된 ACT 모델 로드 중: {args.model_id}")
    policy = ACTPolicy.from_pretrained(args.model_id)
    # 핵심: 100스텝 오픈루프 대신 args.replan_steps 마다 카메라를 다시 보고 폐루프(Closed-loop) 제어 수행
    policy.config.n_action_steps = args.replan_steps
    policy.to(device)
    policy.eval()

    stats = load_norm_stats(args.model_id, device)
    print(f"      - Action Chunk 크기 (chunk_size)     : {policy.config.chunk_size} 스텝")
    print(f"      - 실행 스텝 단위 (n_action_steps)    : {policy.config.n_action_steps} 스텝")

    print("[3/4] MuJoCo ALOHA 양팔 로봇 환경 초기화 (gym_aloha/AlohaTransferCube-v0)...")
    env = gym.make(
        "gym_aloha/AlohaTransferCube-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=args.steps,
    )

    obs, info = env.reset(seed=args.seed)
    policy.reset()

    frames = []
    max_reward = 0.0
    success_step = None

    print(f"[4/4] 모방학습 정책 추론 시작 (총 {args.steps} 스텝)...")
    t_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        # 1. 현재 관측값을 텐서로 변환 후 정규화
        batch = format_and_normalize_obs(obs, stats, device)

        # 2. ACT 정책으로 다음 관절 목표값(14-DoF) 추론 및 역정규화
        with torch.inference_mode():
            action_norm = policy.select_action(batch)
            action_tensor = action_norm * stats["action_std"] + stats["action_mean"]
        action_np = action_tensor.squeeze(0).cpu().numpy()

        # 3. MuJoCo 물리 시뮬레이션 1스텝 진행
        obs, reward, terminated, truncated, info = env.step(action_np)
        max_reward = max(max_reward, float(reward))
        is_success = float(reward) >= 4.0  # AlohaTransferCube에서 최고 보상(전달 완료)은 4.0
        if is_success and success_step is None:
            success_step = step
            print(f"  ★ Step {step:03d}: 양팔 큐브 공중 전달(Handover) 성공! (Reward = {reward})")

        elapsed = time.perf_counter() - t_start
        fps = step / max(elapsed, 1e-6)

        # 4. 상단 카메라 프레임에 HUD 정보 오버레이
        frame_hud = add_hud_overlay(obs["pixels"]["top"], step, args.steps, float(reward), is_success, fps)
        frames.append(frame_hud)

        if args.live:
            cv2.imshow("MuJoCo ALOHA - ACT Imitation Learning", cv2.cvtColor(frame_hud, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("사용자에 의해 중단되었습니다.")
                break

        if step % 100 == 0:
            print(f"  - 진행 중: {step}/{args.steps} steps | 현재 보상: {reward:.1f} | 최고 보상: {max_reward:.1f} | 속도: {fps:.1f} FPS")

    if args.live:
        cv2.destroyAllWindows()
    env.close()

    # 결과 MP4 영상 저장
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "aloha_act_transfer_cube.mp4"
    imageio.mimsave(str(video_path), frames, fps=50)

    print("=" * 70)
    print(f"추론 완료! 최고 보상(Max Reward): {max_reward:.1f} / 4.0")
    if success_step is not None:
        print(f"성공 여부: 성공 (Step {success_step}에서 반대쪽 팔로 전달 완료!)")
    print(f"결과 영상 저장 위치: {video_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
