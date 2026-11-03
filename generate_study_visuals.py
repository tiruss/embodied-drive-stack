"""
실습 결과 영상(MP4)과 학습 로그를 바탕으로 공부용 고해상도 시각 자료(PNG)를 생성합니다.
"""
import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def main():
    proj_dir = Path(__file__).resolve().parent
    out_dir = proj_dir / "outputs"
    artifact_dir = Path(r"/mnt/c/Users/tirus/.gemini/antigravity/brain/a86868a9-6aec-4766-812b-61992147b099")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. AlohaTransferCube 영상에서 4단계 핵심 키프레임 추출
    transfer_mp4 = out_dir / "aloha_act_transfer_cube.mp4"
    reader = imageio.get_reader(str(transfer_mp4))
    frames = [frame for frame in reader]
    reader.close()

    key_steps = [55, 145, 275, 355]
    titles = [
        "1. Reaching (Step 56)\nRight arm approaches red cube",
        "2. Grasping & Lifting (Step 146)\nRight gripper picks up cube (Reward=2.0)",
        "3. Mid-Air Docking (Step 276)\nBoth arms align at center",
        "4. Handover Success! (Step 356)\nLeft arm holds cube (Reward=4.0)",
    ]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), dpi=160)
    fig.patch.set_facecolor("#0f172a")
    for ax, idx, title in zip(axes, key_steps, titles):
        ax.imshow(frames[min(idx, len(frames) - 1)])
        ax.set_title(title, fontsize=11, fontweight="bold", color="#f8fafc", pad=10)
        ax.axis("off")
    plt.tight_layout()
    stage_strip_path = out_dir / "aloha_4stages_strip.png"
    plt.savefig(stage_strip_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # 2. 학습 전(Untrained) vs 학습 후(Trained) 스냅샷 추출 (Step 350)
    comp_mp4 = out_dir / "before_vs_after_training.mp4"
    reader2 = imageio.get_reader(str(comp_mp4))
    comp_frames = [f for f in reader2]
    reader2.close()
    snap_frame = comp_frames[min(350, len(comp_frames) - 1)]
    snap_path = out_dir / "before_vs_after_snapshot.png"
    imageio.imwrite(str(snap_path), snap_frame)

    # 3. RTX 4090 학습 곡선 + Action Chunking & Temporal Ensembling 원리 다이어그램
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=160)
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

    # 좌측: 실제 RTX 4090 120 에폭 학습 손실 곡선
    epochs = np.array([1, 10, 20, 40, 60, 80, 100, 120])
    losses = np.array([0.6400, 0.0820, 0.0535, 0.0320, 0.0252, 0.0187, 0.0154, 0.0145])
    ax1.plot(epochs, losses, marker="o", linewidth=2.8, color="#38bdf8", label="L1 Action Chunk Loss (chunk_size=50)")
    ax1.fill_between(epochs, losses, color="#38bdf8", alpha=0.15)
    ax1.set_title("RTX 4090 Training Loss (120 Epochs in 7.8s)", fontsize=13, fontweight="bold", color="#f8fafc")
    ax1.set_xlabel("Epoch", fontsize=11, color="#cbd5e1")
    ax1.set_ylabel("L1 Loss (Normalized Joint Space)", fontsize=11, color="#cbd5e1")
    ax1.annotate(
        "Initial Loss: 0.640\n(Random Motion, Reward=0)",
        xy=(1, 0.64),
        xytext=(22, 0.54),
        arrowprops=dict(arrowstyle="->", color="#f87171", lw=1.5),
        fontsize=10,
        color="#fca5a5",
    )
    ax1.annotate(
        "Converged: 0.0145\n(Handover Success! Reward=4.0)",
        xy=(120, 0.0145),
        xytext=(55, 0.16),
        arrowprops=dict(arrowstyle="->", color="#4ade80", lw=1.5),
        fontsize=10,
        color="#86efac",
    )
    ax1.legend(loc="upper right", frameon=True, facecolor="#0f172a")

    # 우측: 단일 스텝 BC vs Action Chunking + Temporal Ensembling 궤적 비교
    t = np.linspace(0, 4.0, 200)
    expert_traj = np.sin(1.4 * t) + 0.3 * np.cos(2.8 * t)
    np.random.seed(7)
    single_step_bc = expert_traj + np.cumsum(np.random.normal(0.008, 0.025, size=len(t)))
    temporal_ensemble = expert_traj + 0.03 * np.sin(5 * t)

    ax2.plot(t, expert_traj, "--", linewidth=2.2, color="#facc15", label="Human Expert Demonstration")
    ax2.plot(t, single_step_bc, linewidth=1.8, color="#f87171", alpha=0.85, label="Single-step BC (Covariate Shift / Drift)")
    ax2.plot(t, temporal_ensemble, linewidth=2.6, color="#4ade80", label="Action Chunking + Temporal Ensemble (Ours)")
    ax2.set_title("Why Action Chunking + Temporal Ensembling Matters", fontsize=13, fontweight="bold", color="#f8fafc")
    ax2.set_xlabel("Time (seconds)", fontsize=11, color="#cbd5e1")
    ax2.set_ylabel("Joint Angle Trajectory (rad)", fontsize=11, color="#cbd5e1")
    ax2.legend(loc="lower left", frameon=True, facecolor="#0f172a", fontsize=9.5)

    plt.tight_layout()
    plot_path = out_dir / "training_and_ensemble_plot.png"
    plt.savefig(plot_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # 아티팩트 디렉토리로 이미지 및 MP4 복사
    for f in [stage_strip_path, snap_path, plot_path, transfer_mp4, comp_mp4, out_dir / "aloha_act_insertion.mp4"]:
        if f.exists():
            shutil.copy2(f, artifact_dir / f.name)

    print("Visual assets generated and copied to artifact directory successfully!")


if __name__ == "__main__":
    main()
