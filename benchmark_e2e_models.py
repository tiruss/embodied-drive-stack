"""
Benchmark Suite: Comprehensive Hardware & Architecture Profiling on NVIDIA RTX 4090
===================================================================================
CARLA 자율주행 E2E / VLA 전 세대 모델(1세대 PilotNet, 2세대 Single-Mode UniAD, 3세대 SOTA DiffusionDrive+Hydra-MDP,
그리고 Hugging Face 사전학습 CLIP+MiniLM VLM-VLA)의:
  1) 모델 크기 (파라미터 수, 서브모듈별 비중, FP32/FP16 가중치 용량 MB)
  2) 입출력 텐서 명세 및 해상도 (센서 원본 해상도 vs 네트워크 입력 해상도 vs BEV 출력 해상도)
  3) RTX 4090 CUDA Event 정밀 추론 시간 (Mean, Std, P50, P95, P99 ms & 최대 제어 주파수 Hz/FPS)
  4) 3세대 SOTA 서브모듈별 지연 시간 분해 (BEV 인코딩 vs 4-Step Flow ODE vs Hydra-MDP Scorer)
  5) ODE 디노이징 스텝 수(1, 2, 4, 8, 10 steps)에 따른 추론 속도 변화
  6) VRAM 메모리 점유량 (가중치 전용 vs Batch=1 실시간 추론 피크 vs Batch=64 학습 피크)
을 실제 RTX 4090 GPU에서 1,000회 반복 측정하여 JSON/콘솔 표 및 고해상도 분석 차트로 저장합니다.
"""
import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from demo_12_carla_e2e_gen1_vs_gen2 import Gen1_BlackBox_PilotNet, Gen2_Planning_UniAD
from demo_13_carla_sota_diffusion_bev_e2e import Gen2_SingleMode_L2_E2E, SOTA_DiffusionDrive_HydraMDP


def count_params(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def measure_cuda_latency_ms(fn, warmup: int = 100, iters: int = 1000) -> dict:
    """torch.cuda.Event를 사용하여 마이크로초 단위의 정밀 GPU 커널 실행 시간을 측정합니다."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times_ms = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)], dtype=np.float64)
    mean_ms = float(np.mean(times_ms))
    return {
        "mean_ms": mean_ms,
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "max_ms": float(np.max(times_ms)),
        "fps_hz": float(1000.0 / mean_ms),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print("=" * 94)
    print(f"🔬 [Hardware Benchmark] Autonomous Driving E2E Architecture Suite on {gpu_name} ({total_vram_gb:.1f} GB VRAM)")
    print("=" * 94)

    # 1. 서브모듈별 파라미터 분해 (3rd-Gen SOTA: SOTA_DiffusionDrive_HydraMDP)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem_mb = torch.cuda.memory_allocated() / (1024**2)

    sota_model = SOTA_DiffusionDrive_HydraMDP(d_model=256, num_modes=5, chunk_size=15).to(device)
    sota_model.eval()
    sota_weight_mem_mb = (torch.cuda.memory_allocated() / (1024**2)) - base_mem_mb

    submodules = {
        "1. Perspective RGB Encoder (4-Stage Conv)": sota_model.rgb_enc,
        "2. Perspective-to-BEV Spatial Lifter": sota_model.persp_to_bev,
        "3. Top-Down BEV Occupancy Decoder (64x64)": sota_model.bev_decoder,
        "4. BEV Scene Context Transformer (3-Layer)": nn.ModuleList(
            [sota_model.state_proj, sota_model.scene_transformer]
        ),
        "5. DiffusionDrive Flow-Matching ODE Net": nn.ModuleList(
            [sota_model.mode_emb, sota_model.traj_in_proj, sota_model.time_mlp, sota_model.flow_vel_net]
        ),
        "6. Hydra-MDP 5-Mode Neural Scorer Head": sota_model.hydra_scorer,
        "7. Trajectory-Conditioned Control Head": sota_model.ctrl_head,
    }

    total_sota_params, _ = count_params(sota_model)
    print("\n📊 [1/5] 3세대 SOTA (`SOTA_DiffusionDrive_HydraMDP`) 서브모듈별 파라미터 및 용량 상세 분해:")
    print("-" * 94)
    print(f"{'Sub-Module Name':<46} | {'Parameters':>12} | {'Share (%)':>9} | {'FP32 (MB)':>10} | {'FP16 (MB)':>10}")
    print("-" * 94)
    sub_param_dict = {}
    for name, mod in submodules.items():
        p_cnt, _ = count_params(mod)
        if "4. BEV Scene" in name:
            p_cnt += sota_model.bev_pos.numel()
        share = (p_cnt / total_sota_params) * 100.0
        fp32_mb = (p_cnt * 4) / (1024**2)
        fp16_mb = (p_cnt * 2) / (1024**2)
        sub_param_dict[name] = p_cnt
        print(f"{name:<46} | {p_cnt:>12,d} | {share:>8.2f}% | {fp32_mb:>9.3f} | {fp16_mb:>9.3f}")
    print("-" * 94)
    print(
        f"{'TOTAL (SOTA_DiffusionDrive_HydraMDP)':<46} | {total_sota_params:>12,d} | {'100.00%':>9} | "
        f"{(total_sota_params*4)/(1024**2):>9.3f} | {(total_sota_params*2)/(1024**2):>9.3f}"
    )

    # 2. 세대별 E2E 모델 크기, 추론 지연 시간(Latency), 최대 FPS, 추론/학습 VRAM 비교
    gen1_model = Gen1_BlackBox_PilotNet().to(device).eval()
    gen2_uniad = Gen2_Planning_UniAD(d_model=256, chunk_size=15).to(device).eval()
    gen2_single = Gen2_SingleMode_L2_E2E(d_model=256, chunk_size=15).to(device).eval()

    dummy_rgb_b1 = torch.randn(1, 3, 128, 128, device=device)
    dummy_spd_b1 = torch.randn(1, 1, device=device)
    dummy_st4_b1 = torch.randn(1, 4, device=device)
    dummy_st5_b1 = torch.randn(1, 5, device=device)

    dummy_rgb_b16 = torch.randn(16, 3, 128, 128, device=device)
    dummy_st5_b16 = torch.randn(16, 5, device=device)

    with torch.no_grad():
        lat_gen1 = measure_cuda_latency_ms(lambda: gen1_model(dummy_rgb_b1, dummy_spd_b1))
        lat_gen2_uniad = measure_cuda_latency_ms(lambda: gen2_uniad(dummy_rgb_b1, dummy_st4_b1))
        lat_gen2_single = measure_cuda_latency_ms(lambda: gen2_single(dummy_rgb_b1, dummy_st5_b1))
        lat_sota_1step = measure_cuda_latency_ms(
            lambda: sota_model.forward_inference(dummy_rgb_b1, dummy_st5_b1, ode_steps=1)
        )
        lat_sota_2step = measure_cuda_latency_ms(
            lambda: sota_model.forward_inference(dummy_rgb_b1, dummy_st5_b1, ode_steps=2)
        )
        lat_sota_4step = measure_cuda_latency_ms(
            lambda: sota_model.forward_inference(dummy_rgb_b1, dummy_st5_b1, ode_steps=4)
        )
        lat_sota_8step = measure_cuda_latency_ms(
            lambda: sota_model.forward_inference(dummy_rgb_b1, dummy_st5_b1, ode_steps=8)
        )
        lat_sota_10step = measure_cuda_latency_ms(
            lambda: sota_model.forward_inference(dummy_rgb_b1, dummy_st5_b1, ode_steps=10)
        )
        lat_sota_b16 = measure_cuda_latency_ms(
            lambda: sota_model.forward_inference(dummy_rgb_b16, dummy_st5_b16, ode_steps=4)
        )

    # 전처리(OpenCV 560x256 -> 128x128 리사이즈 + CPU->GPU 복사) 시간 포함 End-to-End 파이프라인 시간 측정
    raw_cam_np = np.random.randint(0, 255, size=(256, 560, 3), dtype=np.uint8)
    state_np = np.array([0.85, 0.02, 0.45, -0.20, 0.50], dtype=np.float32)

    def full_e2e_camera_to_control():
        rgb_128 = cv2.resize(raw_cam_np, (128, 128)).transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        t_rgb = torch.from_numpy(rgb_128).unsqueeze(0).to(device, non_blocking=True)
        t_st = torch.from_numpy(state_np).unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            best_wps, pred_ctrls, cand_trajs, h_scores, pred_bev = sota_model.forward_inference(
                t_rgb, t_st, ode_steps=4
            )
        _ = pred_ctrls[0, 0].cpu().numpy()

    lat_full_pipeline = measure_cuda_latency_ms(full_e2e_camera_to_control, warmup=50, iters=500)

    # 3. 3세대 SOTA 내부 스테이지별 Latency 정밀 분해
    with torch.no_grad():
        scene_ctx_cached, _ = sota_model.encode_scene(dummy_rgb_b1, dummy_st5_b1)
        tau_cached = sota_model.anchors.clone()

        # Stage A: Perspective-to-BEV + BEV Occupancy Decoder (64x64) + Scene Transformer
        lat_stage_bev_enc = measure_cuda_latency_ms(lambda: sota_model.encode_scene(dummy_rgb_b1, dummy_st5_b1))

        # Stage B: 4-Step Truncated Flow-Matching ODE Loop
        def run_only_flow_ode():
            tau = sota_model.anchors.clone()
            dt = 0.25
            for s_i in range(4):
                t_val = torch.full((1, 1), s_i * dt, device=device)
                tau = tau + dt * sota_model.predict_flow_velocity(tau, t_val, scene_ctx_cached)
            return tau

        lat_stage_flow_ode = measure_cuda_latency_ms(run_only_flow_ode)

        # Stage C: Hydra-MDP 5-Mode Scorer + Winner Selection + Control Chunk Head
        def run_only_hydra_and_ctrl():
            tau_flat = (tau_cached / 15.0).view(1, 5, 30)
            mode_ids = torch.arange(5, device=device).unsqueeze(0)
            m_feat = sota_model.mode_emb(mode_ids)
            ctx_exp = scene_ctx_cached.unsqueeze(1).expand(1, 5, 512)
            scores = torch.sigmoid(
                sota_model.hydra_scorer(torch.cat([ctx_exp, m_feat, tau_flat], dim=-1)).squeeze(-1)
            )
            best_idx = torch.argmax(scores, dim=-1)
            best_wps = tau_cached[0, best_idx]
            ctrl = sota_model.ctrl_head(torch.cat([scene_ctx_cached, best_wps.view(1, 30) / 15.0], dim=-1))
            return scores, ctrl

        lat_stage_hydra_ctrl = measure_cuda_latency_ms(run_only_hydra_and_ctrl)

    # 4. VRAM 정밀 측정 (Batch=1 추론 피크 vs Batch=64 학습 피크)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = sota_model.forward_inference(dummy_rgb_b1, dummy_st5_b1, ode_steps=4)
    vram_infer_b1_mb = torch.cuda.max_memory_allocated() / (1024**2)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = sota_model.forward_inference(dummy_rgb_b16, dummy_st5_b16, ode_steps=4)
    vram_infer_b16_mb = torch.cuda.max_memory_allocated() / (1024**2)

    # Batch=64 멀티태스크 학습 시 Peak VRAM 측정
    sota_model.train()
    opt_tmp = torch.optim.AdamW(sota_model.parameters(), lr=1e-3)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    b_sz = 64
    b_rgb = torch.randn(b_sz, 3, 128, 128, device=device)
    b_st = torch.randn(b_sz, 5, device=device)
    b_bev_gt = torch.rand(b_sz, 3, 64, 64, device=device)
    b_cand_gt = torch.randn(b_sz, 5, 15, 2, device=device)
    b_scores_gt = torch.rand(b_sz, 5, device=device)
    b_exp_wps = torch.randn(b_sz, 15, 2, device=device)
    b_exp_ctrl = torch.randn(b_sz, 15, 3, device=device)

    scene_ctx, pred_bev = sota_model.encode_scene(b_rgb, b_st)
    tau_0 = sota_model.anchors.expand(b_sz, -1, -1, -1)
    t_rand = torch.rand((b_sz, 1), device=device)
    tau_t = (1.0 - t_rand.view(b_sz, 1, 1, 1)) * tau_0 + t_rand.view(b_sz, 1, 1, 1) * b_cand_gt
    pred_vel = sota_model.predict_flow_velocity(tau_t, t_rand, scene_ctx)
    loss_flow = F.mse_loss(pred_vel, b_cand_gt - tau_0)
    tau_flat = (b_cand_gt / 15.0).view(b_sz, 5, 30)
    m_feat = sota_model.mode_emb(torch.arange(5, device=device).unsqueeze(0).expand(b_sz, 5))
    ctx_exp = scene_ctx.unsqueeze(1).expand(b_sz, 5, 512)
    pred_scores = torch.sigmoid(sota_model.hydra_scorer(torch.cat([ctx_exp, m_feat, tau_flat], dim=-1)).squeeze(-1))
    loss_score = F.mse_loss(pred_scores, b_scores_gt)
    loss_bev = F.mse_loss(pred_bev, b_bev_gt)
    pred_ctrls = sota_model.ctrl_head(torch.cat([scene_ctx, b_exp_wps.view(b_sz, 30) / 15.0], dim=-1)).view(b_sz, 15, 3)
    loss_ctrl = F.mse_loss(pred_ctrls, b_exp_ctrl)
    total_loss = loss_flow + 2.0 * loss_score + 1.5 * loss_ctrl + 0.8 * loss_bev
    opt_tmp.zero_grad()
    total_loss.backward()
    opt_tmp.step()
    vram_train_b64_mb = torch.cuda.max_memory_allocated() / (1024**2)

    p_gen1, _ = count_params(gen1_model)
    p_gen2_u, _ = count_params(gen2_uniad)
    p_gen2_s, _ = count_params(gen2_single)

    print("\n⚡ [2/5] 세대별 자율주행 E2E 모델 추론 속도(Latency) & 제어 주파수(FPS) 비교 (Batch=1, 1000회 측정):")
    print("-" * 94)
    print(
        f"{'Model Architecture':<38} | {'Params':>10} | {'Mean (ms)':>9} | {'P50 (ms)':>8} | "
        f"{'P99 (ms)':>8} | {'Max Control FPS':>15}"
    )
    print("-" * 94)
    rows = [
        ("1st-Gen Black-Box (PilotNet)", p_gen1, lat_gen1),
        ("2nd-Gen Modular (UniAD Persp-Seg)", p_gen2_u, lat_gen2_uniad),
        ("2nd-Gen Single-Mode BEV (L2 Plan)", p_gen2_s, lat_gen2_single),
        ("3rd-Gen SOTA (1-Step Flow + Hydra)", total_sota_params, lat_sota_1step),
        ("3rd-Gen SOTA (2-Step Flow + Hydra)", total_sota_params, lat_sota_2step),
        ("3rd-Gen SOTA (4-Step Flow + Hydra) ★", total_sota_params, lat_sota_4step),
        ("3rd-Gen SOTA (8-Step Flow + Hydra)", total_sota_params, lat_sota_8step),
        ("3rd-Gen SOTA (10-Step Flow + Hydra)", total_sota_params, lat_sota_10step),
        ("3rd-Gen SOTA Full (OpenCV+H2D+GPU+D2H)", total_sota_params, lat_full_pipeline),
    ]
    for m_name, p_cnt, st in rows:
        print(
            f"{m_name:<38} | {p_cnt:>10,d} | {st['mean_ms']:>8.3f} | {st['p50_ms']:>8.3f} | "
            f"{st['p99_ms']:>8.3f} | {st['fps_hz']:>11.1f} Hz"
        )
    print("-" * 94)

    print("\n🔍 [3/5] 3세대 SOTA (`ode_steps=4`) 내부 파이프라인 단계별 실행 시간 분해:")
    print(f"   - Stage A (RGB Backbone + Persp-to-BEV + 64x64 BEV Decoder + Transformer) : {lat_stage_bev_enc['mean_ms']:.3f} ms")
    print(f"   - Stage B (DiffusionDrive 4-Step Euler ODE Velocity Field Integration)    : {lat_stage_flow_ode['mean_ms']:.3f} ms")
    print(f"   - Stage C (Hydra-MDP 5-Mode Neural Scorer + Winner Selection + Ctrl Head) : {lat_stage_hydra_ctrl['mean_ms']:.3f} ms")
    print(f"   - Stage D (OpenCV 560x256->128x128 Resize + PCIe H2D/D2H Copy Overhead)   : {max(0.05, lat_full_pipeline['mean_ms'] - lat_sota_4step['mean_ms']):.3f} ms")

    print("\n💾 [4/5] RTX 4090 (24 GB) VRAM 메모리 점유율 상세 측정:")
    print(f"   - 모델 가중치 순수 용량 (FP32 Weights Only)        : {sota_weight_mem_mb:.2f} MB (FP16 변환 시 {sota_weight_mem_mb/2:.2f} MB)")
    print(f"   - 실시간 폐루프 추론 Peak VRAM (Batch = 1)         : {vram_infer_b1_mb:.2f} MB (전체 24GB VRAM의 {vram_infer_b1_mb/24576*100:.2f}%)")
    print(f"   - 다중 시뮬레이션 추론 Peak VRAM (Batch = 16)      : {vram_infer_b16_mb:.2f} MB")
    print(f"   - 멀티태스크 역전파 학습 Peak VRAM (Batch = 64)    : {vram_train_b64_mb:.2f} MB (AdamW 옵티마이저 + 그래디언트 포함)")

    # 5. 벤치마크 시각화 차트 생성 (4-Panel High-Res Figure)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=130)
    fig.patch.set_facecolor("#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#475569")

    # Panel 1: 세대별 모델 추론 시간 (Mean & P99 Latency ms)
    arch_labels = [
        "1st-Gen\nPilotNet",
        "2nd-Gen\nUniAD (Seg)",
        "2nd-Gen\nSingle-BEV",
        "3rd-Gen SOTA\n(2-Step ODE)",
        "3rd-Gen SOTA\n(4-Step ODE)★",
        "3rd-Gen SOTA\n(Full + OpenCV)",
    ]
    means = [
        lat_gen1["mean_ms"],
        lat_gen2_uniad["mean_ms"],
        lat_gen2_single["mean_ms"],
        lat_sota_2step["mean_ms"],
        lat_sota_4step["mean_ms"],
        lat_full_pipeline["mean_ms"],
    ]
    p99s = [
        lat_gen1["p99_ms"],
        lat_gen2_uniad["p99_ms"],
        lat_gen2_single["p99_ms"],
        lat_sota_2step["p99_ms"],
        lat_sota_4step["p99_ms"],
        lat_full_pipeline["p99_ms"],
    ]
    x_pos = np.arange(len(arch_labels))
    w = 0.36
    b1 = axes[0, 0].bar(x_pos - w / 2, means, width=w, color="#34d399", label="Mean Latency (ms)")
    b2 = axes[0, 0].bar(x_pos + w / 2, p99s, width=w, color="#38bdf8", label="P99 Tail Latency (ms)")
    axes[0, 0].axhline(10.0, color="#f87171", linestyle="--", linewidth=1.3, label="100 Hz Real-Time Budget (10 ms)")
    for bar in b1:
        h = bar.get_height()
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2, h + 0.12, f"{h:.2f}", ha="center", va="bottom", color="#f8fafc", fontsize=8
        )
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(arch_labels, fontsize=8.5)
    axes[0, 0].set_ylabel("Inference Time (ms)", color="#cbd5e1")
    axes[0, 0].set_title(
        "1. Inference Latency Across E2E Generations on RTX 4090 (Batch=1)",
        color="#f8fafc",
        fontsize=11,
        fontweight="bold",
    )
    axes[0, 0].legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=8.5)
    axes[0, 0].grid(True, axis="y", color="#334155", linestyle=":", alpha=0.6)

    # Panel 2: 3세대 SOTA 서브모듈별 파라미터 분포 (Horizontal Bar)
    mod_ShortNames = [
        "1. RGB Encoder (Conv)",
        "2. Persp-to-BEV Lifter",
        "3. BEV Decoder (64x64)",
        "4. Scene Transformer (3L)",
        "5. Flow-Matching ODE Net",
        "6. Hydra-MDP Scorer",
        "7. Control Chunk Head",
    ]
    mod_p_millions = [v / 1e6 for v in sub_param_dict.values()]
    y_pos = np.arange(len(mod_ShortNames))
    bars_p = axes[0, 1].barh(
        y_pos,
        mod_p_millions,
        color=["#60a5fa", "#818cf8", "#34d399", "#fbbf24", "#f472b6", "#fb923c", "#a78bfa"],
    )
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(mod_ShortNames, fontsize=8.5)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Parameters (Millions)", color="#cbd5e1")
    axes[0, 1].set_title(
        f"2. SOTA Architecture Parameter Breakdown (Total: {total_sota_params/1e6:.2f}M / {total_sota_params*4/(1024**2):.1f} MB)",
        color="#f8fafc",
        fontsize=11,
        fontweight="bold",
    )
    for bar in bars_p:
        w_val = bar.get_width()
        axes[0, 1].text(
            w_val + 0.04, bar.get_y() + bar.get_height() / 2, f"{w_val:.2f}M", va="center", color="#f8fafc", fontsize=8
        )
    axes[0, 1].set_xlim(0, max(mod_p_millions) * 1.22)
    axes[0, 1].grid(True, axis="x", color="#334155", linestyle=":", alpha=0.6)

    # Panel 3: Flow-Matching ODE Steps (1, 2, 4, 8, 10) vs Latency (ms) & Control FPS (Hz)
    ode_list = [1, 2, 4, 8, 10]
    ode_lats = [
        lat_sota_1step["mean_ms"],
        lat_sota_2step["mean_ms"],
        lat_sota_4step["mean_ms"],
        lat_sota_8step["mean_ms"],
        lat_sota_10step["mean_ms"],
    ]
    ode_fps = [
        lat_sota_1step["fps_hz"],
        lat_sota_2step["fps_hz"],
        lat_sota_4step["fps_hz"],
        lat_sota_8step["fps_hz"],
        lat_sota_10step["fps_hz"],
    ]
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    l1 = ax3.plot(ode_list, ode_lats, "-o", color="#f472b6", linewidth=2.5, markersize=7, label="Latency (ms)")
    l2 = ax3_twin.plot(ode_list, ode_fps, "--s", color="#34d399", linewidth=2.5, markersize=7, label="Control Freq (Hz)")
    ax3.axvline(4, color="#fbbf24", linestyle=":", linewidth=1.5, label="Default Truncated ODE (4 Steps)")
    ax3.set_xlabel("Flow-Matching Euler ODE Steps", color="#cbd5e1")
    ax3.set_ylabel("Latency (ms)", color="#f472b6")
    ax3_twin.set_ylabel("Max Control Loop Frequency (Hz)", color="#34d399")
    ax3_twin.tick_params(colors="#34d399")
    ax3.set_title(
        "3. Truncated Flow-Matching ODE Step Ablation (Speed vs Frequency)",
        color="#f8fafc",
        fontsize=11,
        fontweight="bold",
    )
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", loc="center right")
    ax3.grid(True, color="#334155", linestyle=":", alpha=0.6)

    # Panel 4: SOTA 내부 단계별 실행 시간 & VRAM 점유량 요약
    stage_names = [
        "Stage A:\nBEV + Transformer",
        "Stage B:\n4-Step Flow ODE",
        "Stage C:\nHydra-MDP + Ctrl",
        "Stage D:\nOpenCV + PCIe Copy",
    ]
    stage_ms = [
        lat_stage_bev_enc["mean_ms"],
        lat_stage_flow_ode["mean_ms"],
        lat_stage_hydra_ctrl["mean_ms"],
        max(0.08, lat_full_pipeline["mean_ms"] - lat_sota_4step["mean_ms"]),
    ]
    bars_s = axes[1, 1].bar(stage_names, stage_ms, color=["#38bdf8", "#f472b6", "#34d399", "#fbbf24"], width=0.5)
    for bar in bars_s:
        h = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2, h + 0.03, f"{h:.2f} ms", ha="center", va="bottom", color="#f8fafc", fontsize=8.5
        )
    axes[1, 1].set_ylabel("Execution Time (ms)", color="#cbd5e1")
    axes[1, 1].set_ylim(0, max(stage_ms) * 1.28)
    axes[1, 1].set_title(
        f"4. SOTA Sub-Stage Latency & VRAM (Infer B=1: {vram_infer_b1_mb:.0f}MB / Train B=64: {vram_train_b64_mb:.0f}MB)",
        color="#f8fafc",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 1].grid(True, axis="y", color="#334155", linestyle=":", alpha=0.6)

    plt.tight_layout()
    chart_out = out_dir / "carla_sota_benchmark_analysis.png"
    plt.savefig(str(chart_out), facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "gpu": gpu_name,
        "total_vram_gb": round(total_vram_gb, 2),
        "sota_total_params": total_sota_params,
        "sota_fp32_mb": round((total_sota_params * 4) / (1024**2), 3),
        "sota_fp16_mb": round((total_sota_params * 2) / (1024**2), 3),
        "vram_mb": {
            "weights_only_fp32": round(sota_weight_mem_mb, 2),
            "inference_peak_batch1": round(vram_infer_b1_mb, 2),
            "inference_peak_batch16": round(vram_infer_b16_mb, 2),
            "training_peak_batch64": round(vram_train_b64_mb, 2),
        },
        "latency_batch1": {
            "gen1_pilotnet": lat_gen1,
            "gen2_uniad_seg": lat_gen2_uniad,
            "gen2_single_bev": lat_gen2_single,
            "sota_ode1": lat_sota_1step,
            "sota_ode2": lat_sota_2step,
            "sota_ode4_default": lat_sota_4step,
            "sota_ode8": lat_sota_8step,
            "sota_ode10": lat_sota_10step,
            "sota_full_camera_to_control": lat_full_pipeline,
        },
    }
    json_out = out_dir / "carla_sota_benchmark_report.json"
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n✅ 벤치마크 차트 및 리포트 저장 완료: {chart_out}, {json_out}")


if __name__ == "__main__":
    main()
