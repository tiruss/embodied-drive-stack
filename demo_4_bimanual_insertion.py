"""
Demo 4: High-Precision Bimanual Peg-in-Hole Insertion (AlohaInsertion-v0)
=========================================================================
양팔 로봇 제어에서 가장 까다로운 작업 중 하나인 '양팔 공중 핀 삽입(Peg-in-Hole Insertion)' 데모입니다.
- 왼쪽 팔: 파란색 소켓(구멍)을 집어 들어 올림
- 오른쪽 팔: 빨간색 핀(Peg)을 집어 들어 올림
- 두 팔이 공중에서 정밀하게 정렬하여 소켓 구멍 안에 핀을 끝까지 꽂아 넣음 (Reward = 4.0)

실행 방법:
  MUJOCO_GL=egl ~/.venvs/mujoco-lerobot/bin/python demo_4_bimanual_insertion.py
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
from demo_2_act_pretrained import format_and_normalize_obs, load_norm_stats
from lerobot.policies.act.modeling_act import ACTPolicy


def add_insertion_hud(frame_rgb: np.ndarray, step: int, max_steps: int, reward: float, fps: float) -> np.ndarray:
    canvas = frame_rgb.copy()
    stage_map = {
        0.0: "Stage 0: Reaching for Peg & Socket",
        1.0: "Stage 1: Grasped Peg & Socket",
        2.0: "Stage 2: Lifted & Aligning in Mid-Air",
        3.0: "Stage 3: Contacting Socket Hole",
        4.0: "SUCCESS! (Peg Fully Inserted into Socket)",
    }
    stage_desc = stage_map.get(float(reward), f"Reward: {reward:.1f}")
    color = (40, 230, 90) if reward >= 4.0 else (255, 210, 60)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 72), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

    cv2.putText(
        canvas,
        f"ACT Policy | AlohaInsertion-v0 (Peg-in-Hole) | Step: {step:03d}/{max_steps} | {fps:.1f} FPS",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Reward: {reward:.1f}/4.0 | {stage_desc}",
        (14, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Run ACT Bimanual Peg-in-Hole Insertion on MuJoCo ALOHA")
    parser.add_argument("--model-id", type=str, default="lerobot/act_aloha_sim_insertion_human")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replan-steps", type=int, default=25)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(f"[1/3] ACT 양팔 핀 삽입 모델 로드 중: {args.model_id}")
    policy = ACTPolicy.from_pretrained(args.model_id)
    policy.config.n_action_steps = args.replan_steps
    policy.to(device).eval()

    stats = load_norm_stats(args.model_id, device)

    print("[2/3] MuJoCo AlohaInsertion-v0 환경 초기화...")
    env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels_agent_pos", max_episode_steps=args.steps)
    obs, _ = env.reset(seed=args.seed)
    policy.reset()

    frames = []
    max_reward = 0.0
    success_step = None
    t0 = time.perf_counter()

    print(f"[3/3] 양팔 핀 삽입(Peg-in-Hole) 추론 시작 (replan_steps={args.replan_steps})...")
    for step in range(1, args.steps + 1):
        batch = format_and_normalize_obs(obs, stats, device)
        with torch.inference_mode():
            act_norm = policy.select_action(batch)
            act = (act_norm * stats["action_std"] + stats["action_mean"]).squeeze(0).cpu().numpy()

        obs, reward, _, _, _ = env.step(act)
        max_reward = max(max_reward, float(reward))
        if float(reward) >= 4.0 and success_step is None:
            success_step = step
            print(f"  ★ Step {step:03d}: 양팔 공중 핀 삽입(Peg-in-Hole Insertion) 완수! (Reward = {reward})")

        fps = step / max(time.perf_counter() - t0, 1e-6)
        frame_hud = add_insertion_hud(obs["pixels"]["top"], step, args.steps, float(reward), fps)
        frames.append(frame_hud)

        if args.live:
            cv2.imshow("MuJoCo ALOHA - Peg-in-Hole Insertion", cv2.cvtColor(frame_hud, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if step % 100 == 0:
            print(f"  - 진행 중: {step}/{args.steps} | 현재 보상: {reward:.1f} | 최고 보상: {max_reward:.1f}")

    if args.live:
        cv2.destroyAllWindows()
    env.close()

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / "aloha_act_insertion.mp4"
    imageio.mimsave(str(out_video), frames, fps=50)

    print("=" * 72)
    print(f"추론 완료! 최고 보상(Max Reward): {max_reward:.1f} / 4.0")
    print(f"결과 영상: {out_video}")
    print("=" * 72)


if __name__ == "__main__":
    main()
