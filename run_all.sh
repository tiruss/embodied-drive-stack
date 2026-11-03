#!/usr/bin/env bash
set -euo pipefail

# Ensure we are in the repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-quick}"

echo "========================================================================"
echo "  Embodied Drive Stack - One-Click Reproducible Runner (Mode: ${MODE})"
echo "========================================================================"

# 1. Python Environment Setup / Activation
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f ".venv/bin/activate" ]]; then
        echo "[Env] Activating local .venv..."
        source .venv/bin/activate
    elif [[ -d "$HOME/.venvs/mujoco-lerobot" ]]; then
        echo "[Env] Activating cached virtualenv ($HOME/.venvs/mujoco-lerobot)..."
        source "$HOME/.venvs/mujoco-lerobot/bin/activate"
    else
        echo "[Env] Creating new Python virtual environment (.venv) and installing requirements..."
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
else
    echo "[Env] Using active virtualenv: ${VIRTUAL_ENV}"
fi

# 2. Rust Toolchain Setup & Native Release Build
if ! command -v cargo &>/dev/null; then
    if [[ -f "$HOME/.cargo/env" ]]; then
        source "$HOME/.cargo/env"
    else
        echo "[Rust] Installing minimal Rust toolchain via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
        source "$HOME/.cargo/env"
    fi
fi

echo "[Rust] Building rust_e2e_pipeline (--release, native SIMD)..."
(cd rust_e2e_pipeline && RUSTFLAGS="-C target-cpu=native" cargo build --release)

# 3. Execute Requested Pipeline Mode
run_quick_suite() {
    echo ""
    echo ">>> [1/5] Checking Hardware & PyTorch/MuJoCo Environment..."
    python check_env.py

    echo ""
    echo ">>> [2/5] Exporting Static ONNX & Generating ROS 2 DBW Deployment Node..."
    python export_and_ros2_deploy_template.py

    echo ""
    echo ">>> [3/5] Running Gen-1 vs Gen-2 vs Gen-3 SOTA Hardware Benchmark..."
    python benchmark_e2e_models.py

    echo ""
    echo ">>> [4/5] Running 6-Pillar ISO 26262 ASIL-D Production Pipeline Verification..."
    python production_e2e_pipeline_architecture.py

    echo ""
    echo ">>> [5/5] Running Python vs. Native Rust + CUDA Graph Head-to-Head Benchmark..."
    python benchmark_python_vs_rust_e2e.py
}

run_mujoco_suite() {
    echo ""
    echo ">>> Running Phase 1 ~ Phase 3 Standalone MuJoCo & Highway-Env Demos..."
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    python demo_2_act_pretrained.py
    python demo_3_train_mini_bc.py
    python demo_4_bimanual_insertion.py
    python generate_study_visuals.py
    python demo_5_diffusion_flow_policy.py
    python demo_6_vla_language_conditioned.py
    python demo_7_highway_vla_fsd.py
    python demo_8_mujoco_3d_car_flow.py
}

run_carla_suite() {
    echo ""
    echo ">>> Running Phase 4 ~ Phase 6 CARLA 0.9.15 Autonomous Driving Demos..."
    echo "    (Ensure CARLA 0.9.15 server is running on CARLA_PORT=${CARLA_PORT:-3000 or 2000})"
    python demo_9_carla_urban_vla.py
    python demo_10_carla_end_to_end_vla.py
    python demo_11_carla_pretrained_vlm_vla.py
    python demo_12_carla_e2e_gen1_vs_gen2.py
    python demo_13_carla_sota_diffusion_bev_e2e.py
    python demo_14_carla_sota_dynamic_cutin_jaywalk.py
    python demo_15_carla_action_world_model.py
    python demo_16_carla_surround_6cam_bevformer_e2e.py
}

case "$MODE" in
    quick)
        run_quick_suite
        ;;
    mujoco)
        run_mujoco_suite
        ;;
    carla)
        run_carla_suite
        ;;
    all)
        run_quick_suite
        run_mujoco_suite
        run_carla_suite
        ;;
    *)
        echo "Usage: ./run_all.sh [quick|mujoco|carla|all]"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "  All requested tasks completed! Outputs saved to ./outputs/"
echo "========================================================================"
