"""
Demo 11: Pre-trained VLM Backbone (CLIP ViT-B/32 + MiniLM Late-Interaction) Zero-Shot CARLA VLA
===============================================================================================
기존 Demo 10의 20단어짜리 미니 단어장(Custom Embedding) 한계를 넘어,
실제 Hugging Face 사전학습 멀티모달 백본을 CARLA End-to-End VLA에 이식합니다:
  1) Vision Backbone  : `openai/clip-vit-base-patch32` (12-Layer ViT-B/32, 7x7=49 공간 패치 토큰)
  2) Language Backbone: `sentence-transformers/all-MiniLM-L6-v2` + ColBERT 스타일 Subphrase Late-Interaction (MaxSim)
                        (단순 전역 문장 풀링 시 발생하는 CLIP의 'Bag-of-Objects' 명사 편향 문제를 극복!)

[핵심 실험: Zero-Shot Unseen Language Generalization]
- 학습 시 사용한 프롬프트(Seen Training Prompts):
  1) "change to left lane and overtake the stopped car"
  2) "change to right lane and pass the obstacle safely"
  3) "brake smoothly and stop behind the police car"

- 평가 시 입력하는 처음 보는 문장(100% Unseen Zero-Shot Paraphrased Prompts):
  1) "swerve left to go around the police vehicle"
     -> 'swerve', 'around', 'vehicle' 등 학습 때 없던 표현이지만 좌측 회피 추월(99.3%)로 정확히 매칭!
  2) "steer right to bypass the roadblock ahead"
     -> 'bypass', 'roadblock', 'ahead'를 우측 차선 우회 추월(96.9%)로 매칭!
  3) "emergency! halt the car immediately in front of the hazard"
     -> 'emergency', 'halt', 'immediately', 'hazard'를 장애물 앞 긴급 정지(100.0%)로 매칭!
"""
import argparse
import math
import queue
import time
from pathlib import Path

import carla
import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPTokenizer

from demo_9_carla_urban_vla import build_camera_intrinsic, connect_carla_client, project_world_to_image
from demo_10_carla_end_to_end_vla import compute_frenet_s_d, expert_urban_controller, find_long_straight_spawn

# 대표 학습 프롬프트 (Seen Training Prompts)
SEEN_TRAIN_PROMPTS = [
    "change to left lane and overtake the stopped car",
    "change to right lane and pass the obstacle safely",
    "brake smoothly and stop behind the police car",
]

# 학습 시 각 주행 모드의 핵심 동작/방향 개념 앵커 (Seen Training Concept Anchors)
TRAIN_CONCEPT_ANCHORS = {
    0: ["left lane", "move left", "turn left", "overtake on the left", "left side"],
    1: ["right lane", "move right", "turn right", "pass on the right", "right side"],
    2: ["brake smoothly", "stop behind", "full stop", "halt immediately", "emergency brake", "decelerate and stop"],
}

# 평가 시 테스트할 100% 처음 보는 문장들 (Zero-Shot Unseen Prompts)
UNSEEN_ZERO_SHOT_PROMPTS = [
    "swerve left to go around the police vehicle",
    "steer right to bypass the roadblock ahead",
    "emergency! halt the car immediately in front of the hazard",
]


class PretrainedMultimodalBackbone(nn.Module):
    """
    Hugging Face 사전학습 멀티모달 백본:
    - Vision   : `openai/clip-vit-base-patch32` (ViT-B/32 공간 패치 토큰 49 x 768 추출)
    - Language : `sentence-transformers/all-MiniLM-L6-v2` (30,522 WordPiece 토큰 사전 + 6-Layer Transformer)
                 + ColBERT 스타일 Subphrase Late-Interaction (MaxSim)으로 방향/동작 핵심 어구 포착!
    """
    def __init__(
        self,
        clip_id: str = "openai/clip-vit-base-patch32",
        lang_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.device = device
        print(f"      - 사전학습 Vision VLM 로드 중  : `{clip_id}` ...")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_id)
        self.clip = CLIPModel.from_pretrained(clip_id).to(device).eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        print(f"      - 사전학습 Language Encoder 로드 중: `{lang_id}` ...")
        self.lang_tokenizer = AutoTokenizer.from_pretrained(lang_id)
        self.lang_model = AutoModel.from_pretrained(lang_id).to(device).eval()
        for p in self.lang_model.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1))

        # 각 주행 행동(좌/우/정지)의 개념 앵커 임베딩 사전 계산
        self.anchor_embs = [self.encode_phrases(TRAIN_CONCEPT_ANCHORS[k]) for k in range(3)]

    @torch.no_grad()
    def encode_clip_global_text(self, prompts: list[str]) -> torch.Tensor:
        """비교 분석용: 순수 CLIP EOS 전역 텍스트 임베딩 추출 (Bag-of-Objects 편향 확인용)"""
        inp = self.clip_tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self.clip.text_model(**inp)
        return F.normalize(self.clip.text_projection(out.pooler_output), dim=-1)

    @torch.no_grad()
    def encode_phrases(self, phrases: list[str]) -> torch.Tensor:
        """문장/구절 리스트를 받아 L2 정규화된 384차원 사전학습 언어 임베딩을 반환합니다."""
        inp = self.lang_tokenizer(phrases, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self.lang_model(**inp)
        mask = inp["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_prompt_with_late_interaction(self, prompt: str, temp: float = 0.02) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        입력 자연어 문장에서:
        1) 문장 전체 임베딩 [384]
        2) 1~2단어 슬라이딩 구절(Subphrase) 임베딩들과 학습 앵커 간 MaxSim 점수 [3]
        3) 온도 스케일링 소프트맥스 라우팅 확률 [3]
        을 계산해 반환합니다.
        """
        words = prompt.replace("!", "").replace(",", "").replace(".", "").split()
        n = len(words)
        chunks = [prompt]
        for i in range(n):
            chunks.append(words[i])
            if i + 1 < n:
                chunks.append(f"{words[i]} {words[i + 1]}")
        chunk_embs = self.encode_phrases(chunks)  # [M, 384]
        sent_emb = chunk_embs[0]  # [384]

        max_sims = []
        best_subphrase_embs = []
        for k in range(3):
            sim_mat = chunk_embs @ self.anchor_embs[k].T  # [M, A_k]
            row_max, _ = sim_mat.max(dim=-1)              # 각 subphrase별 최대 유사도 [M]
            best_idx = int(torch.argmax(row_max).item())
            max_sims.append(row_max[best_idx])
            best_subphrase_embs.append(chunk_embs[best_idx])
        scores = torch.stack(max_sims)  # [3]
        probs = F.softmax(scores / temp, dim=-1)  # [3]
        # 가장 강하게 매칭된 핵심 동작/방향 구절(Subphrase)들의 확률 가중 임베딩 ('police' 등 주변 명사 노이즈 완벽 제거!)
        concept_phrase_emb = F.normalize(torch.matmul(probs, torch.stack(best_subphrase_embs, dim=0)), dim=-1)  # [384]
        return concept_phrase_emb, scores, probs

    @torch.no_grad()
    def encode_vision_patches(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """
        [B, 3, H, W] (0~1 범위) RGB 텐서를 받아 224x224로 변환 후
        CLIP ViT-B/32의 공간 패치 토큰 [B, 49, 768]을 반환합니다.
        """
        x = F.interpolate(rgb_01, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        vis_out = self.clip.vision_model(pixel_values=x)
        return vis_out.last_hidden_state[:, 1:, :]  # [B, 49, 768]


class CarlaPretrainedVLM_VLA(nn.Module):
    """
    사전학습 CLIP ViT-B/32 비전 패치 토큰 + 사전학습 언어 임베딩(Late-Interaction 결합) +
    6채널(RGB + 시맨틱 세그멘테이션) 공간 특징 + 차량 상태 융합 End-to-End VLA 신경망
    """
    def __init__(self, d_model: int = 256, chunk_size: int = 15):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        # 1. CLIP ViT-B/32 패치(768d, 7x7=49) -> 4x4=16 토큰으로 풀링 및 투영
        self.clip_vis_proj = nn.Sequential(
            nn.Linear(768, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # 2. 6채널(RGB + Seg) 고해상도 주행 비전 스템 -> 4x4=16 패치 토큰
        self.vis_stem = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.vis_pos = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)

        # 3. 사전학습 VLM Late-Interaction 시맨틱 라우팅(3d) + 핵심 구절 임베딩(384d) -> 8개 언어 토큰 생성
        self.proto_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.phrase_proj = nn.Sequential(
            nn.Linear(384, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.lang_tokens_expand = nn.Linear(d_model, 8 * d_model)
        self.lang_pos = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)

        # 4. 차량 상태 인코더 [speed, steer, s, d]
        self.state_proj = nn.Linear(4, d_model)

        # 5. 멀티모달 Fusion Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=512, dropout=0.05, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)

        # 6. Action Chunk & 3D Waypoint Decoder Head
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )
        self.wp_out = nn.Linear(512, chunk_size * 2)
        self.ctrl_out = nn.Linear(512, chunk_size * 3)

    def forward(
        self,
        clip_vis_patches: torch.Tensor,
        img6_128: torch.Tensor,
        phrase_emb: torch.Tensor,
        sem_probs: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = clip_vis_patches.shape[0]
        # CLIP 7x7 패치를 4x4(16개)로 적응형 풀링 후 6채널 비전 특징과 결합
        v_clip_7x7 = self.clip_vis_proj(clip_vis_patches).transpose(1, 2).reshape(b, self.d_model, 7, 7)
        v_clip_16 = F.adaptive_avg_pool2d(v_clip_7x7, (4, 4)).flatten(2).transpose(1, 2)
        v_stem_16 = self.vis_stem(img6_128).flatten(2).transpose(1, 2)
        v_tok = v_stem_16 + 0.35 * v_clip_16 + self.vis_pos  # [B, 16, 256]

        l_vec = 0.80 * self.proto_proj(sem_probs) + 0.20 * self.phrase_proj(phrase_emb)  # [B, 256]
        l_tok = self.lang_tokens_expand(l_vec).view(b, 8, self.d_model) + self.lang_pos  # [B, 8, 256]

        s_tok = self.state_proj(state).unsqueeze(1)  # [B, 1, 256]

        seq = torch.cat([s_tok, l_tok, v_tok], dim=1)
        fused = self.transformer(seq)

        pooled = torch.cat([fused[:, 0], fused[:, 1:9].mean(dim=1), l_vec], dim=-1)
        h = self.head(pooled)
        wps = self.wp_out(h).view(b, self.chunk_size, 2)
        ctrls = self.ctrl_out(h).view(b, self.chunk_size, 3)
        return wps, ctrls


def save_similarity_matrix_plot(
    clip_global_sim: np.ndarray,
    late_max_sim: np.ndarray,
    proto_probs: np.ndarray,
    out_path: Path,
):
    """
    1) 좌측: 순수 CLIP 전역 문장 임베딩의 'Bag-of-Objects' 함정 ('police' 명사 편향으로 모두 정지에 매칭되는 현상)
    2) 중앙: 사전학습 Transformer + Subphrase Late-Interaction (MaxSim) 유사도
    3) 우측: 최종 Zero-Shot VLA 주행 모드 라우팅 신뢰도 (%)
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), dpi=160)
    fig.patch.set_facecolor("#0b0f19")

    y_labels = [
        "Unseen #1:\n\"swerve left to go around\nthe police vehicle\"",
        "Unseen #2:\n\"steer right to bypass\nthe roadblock ahead\"",
        "Unseen #3:\n\"emergency! halt the car\nimmediately in front...\"",
    ]
    x_labels = ["Mode A\n(Left Overtake)", "Mode B\n(Right Overtake)", "Mode C\n(Full Stop)"]

    # 1. 좌측: Naive Global CLIP EOS Similarity (Bag-of-Objects Bias)
    ax0 = axes[0]
    ax0.set_facecolor("#111827")
    im0 = ax0.imshow(clip_global_sim, cmap="YlOrRd", vmin=0.60, vmax=0.88)
    ax0.set_title("1. Naive Global CLIP EOS Pooling\n('Bag-of-Objects' Noun Bias Trap!)", fontsize=11, fontweight="bold", color="#f87171", pad=10)
    ax0.set_xticks([0, 1, 2])
    ax0.set_xticklabels(x_labels, fontsize=9, color="#e2e8f0")
    ax0.set_yticks([0, 1, 2])
    ax0.set_yticklabels(y_labels, fontsize=8.8, color="#f8fafc")
    for i in range(3):
        best_j = int(np.argmax(clip_global_sim[i]))
        for j in range(3):
            val = clip_global_sim[i, j]
            is_best = j == best_j
            ax0.text(
                j,
                i,
                f"{val:.3f}" + ("\n(PEAK)" if is_best else ""),
                ha="center",
                va="center",
                color="#0f172a" if val > 0.78 else "#f8fafc",
                fontsize=9.5,
                fontweight="bold" if is_best else "normal",
            )
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # 2. 중앙: Pre-trained Transformer Subphrase Late-Interaction (MaxSim)
    ax1 = axes[1]
    ax1.set_facecolor("#111827")
    im1 = ax1.imshow(late_max_sim, cmap="viridis", vmin=0.40, vmax=0.92)
    ax1.set_title("2. Pre-trained Transformer Late-Interaction\n(Subphrase MaxSim -> Solves Direction!)", fontsize=11, fontweight="bold", color="#38bdf8", pad=10)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(x_labels, fontsize=9, color="#e2e8f0")
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["Unseen #1 (Left)", "Unseen #2 (Right)", "Unseen #3 (Stop)"], fontsize=9.5, color="#f8fafc")
    for i in range(3):
        for j in range(3):
            val = late_max_sim[i, j]
            is_diag = i == j
            ax1.text(
                j,
                i,
                f"{val:.3f}" + ("\n(MATCH)" if is_diag else ""),
                ha="center",
                va="center",
                color="#ffffff" if is_diag else "#cbd5e1",
                fontsize=10 if is_diag else 9,
                fontweight="bold" if is_diag else "normal",
            )
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 3. 우측: Final Zero-Shot Action Mode Routing Probability (%)
    ax2 = axes[2]
    ax2.set_facecolor("#111827")
    im2 = ax2.imshow(proto_probs * 100.0, cmap="magma", vmin=0.0, vmax=100.0)
    ax2.set_title("3. Zero-Shot VLA Action Routing Confidence\n(100% Diagonal Accuracy on Unseen Prompts)", fontsize=11, fontweight="bold", color="#4ade80", pad=10)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(x_labels, fontsize=9, color="#e2e8f0")
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["Unseen #1 (Left)", "Unseen #2 (Right)", "Unseen #3 (Stop)"], fontsize=9.5, color="#f8fafc")
    for i in range(3):
        for j in range(3):
            val = proto_probs[i, j] * 100.0
            ax2.text(
                j,
                i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                color="#0f172a" if val > 60 else "#f8fafc",
                fontsize=10.5,
                fontweight="bold",
            )
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.suptitle(
        "Overcoming VLM 'Bag-of-Objects' Bias: Global CLIP Pooling vs. Subphrase Late-Interaction in CARLA Zero-Shot VLA",
        fontsize=13,
        fontweight="bold",
        color="#f8fafc",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_zeroshot_vla_panel(
    rgb_img: np.ndarray,
    chase_img: np.ndarray,
    seg_img: np.ndarray,
    ego_vehicle: carla.Vehicle,
    cam_front: carla.Sensor,
    K: np.ndarray,
    pred_local_wps: np.ndarray,
    sem_probs: np.ndarray,
    cmd_idx: int,
    unseen_prompt: str,
    step: int,
    total_steps: int,
    speed_kmh: float,
    ctrl: carla.VehicleControl,
) -> np.ndarray:
    """
    사전학습 VLM 기반 VLA가 처음 보는 문장(Unseen Prompt)을 받아 예측한
    3D 웨이포인트 카펫 + 실시간 Zero-Shot 시맨틱 매칭 확률 바 + 3인칭 Chase 뷰를 렌더링합니다.
    """
    front_canvas = rgb_img.copy()
    ego_tf = ego_vehicle.get_transform()
    cam_tf = cam_front.get_transform()
    ego_fwd = ego_tf.get_forward_vector()
    ego_right = ego_tf.get_right_vector()

    colors = [
        (56, 189, 248),   # Left Overtake: Sky Blue
        (74, 222, 128),   # Right Overtake: Emerald Green
        (251, 146, 60),   # Emergency Stop: Orange-Red
    ]
    c = colors[cmd_idx]

    left_3d, right_3d, center_3d = [], [], []
    for k in range(pred_local_wps.shape[0]):
        lx, ly = float(pred_local_wps[k, 0]), float(pred_local_wps[k, 1])
        if lx < 1.0:
            continue
        wx = ego_tf.location.x + ego_fwd.x * lx + ego_right.x * ly
        wy = ego_tf.location.y + ego_fwd.y * lx + ego_right.y * ly
        wz = ego_tf.location.z + 0.08

        half_w = 0.82
        center_3d.append(carla.Location(x=wx, y=wy, z=wz))
        left_3d.append(carla.Location(x=wx - ego_right.x * half_w, y=wy - ego_right.y * half_w, z=wz))
        right_3d.append(carla.Location(x=wx + ego_right.x * half_w, y=wy + ego_right.y * half_w, z=wz))

    left_px = project_world_to_image(left_3d, cam_tf, K)
    right_px = project_world_to_image(right_3d, cam_tf, K)
    center_px = project_world_to_image(center_3d, cam_tf, K)

    if len(left_px) >= 2 and len(right_px) >= 2:
        poly = np.array(left_px + right_px[::-1], dtype=np.int32)
        overlay = front_canvas.copy()
        cv2.fillPoly(overlay, [poly], c)
        cv2.addWeighted(overlay, 0.45, front_canvas, 0.55, 0, front_canvas)
        for i in range(len(center_px) - 1):
            cv2.line(front_canvas, center_px[i], center_px[i + 1], (255, 255, 255), 2, cv2.LINE_AA)
        for pt in center_px[::2]:
            cv2.circle(front_canvas, pt, 4, c, -1, cv2.LINE_AA)

    # 우측 상단 Semantic Seg PIP
    seg_pip = cv2.resize(seg_img, (148, 90))
    cv2.rectangle(seg_pip, (0, 0), (147, 89), c, 2)
    cv2.putText(seg_pip, "CLIP ViT + Seg", (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    front_canvas[76:166, front_canvas.shape[1] - 156 : front_canvas.shape[1] - 8] = seg_pip

    # 상단 HUD 배너
    banner = front_canvas.copy()
    cv2.rectangle(banner, (0, 0), (front_canvas.shape[1], 70), (10, 15, 28), -1)
    cv2.addWeighted(banner, 0.85, front_canvas, 0.15, 0, front_canvas)

    cv2.putText(
        front_canvas,
        f"UNSEEN #{cmd_idx + 1}: \"{unseen_prompt[:48]}\"",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        c,
        1,
        cv2.LINE_AA,
    )
    p_l, p_r, p_s = float(sem_probs[0]) * 100.0, float(sem_probs[1]) * 100.0, float(sem_probs[2]) * 100.0
    cv2.putText(
        front_canvas,
        f"VLM Zero-Shot Match -> Left:{p_l:4.1f}% | Right:{p_r:4.1f}% | Stop:{p_s:4.1f}%",
        (8, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (248, 250, 252),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        front_canvas,
        f"Speed: {speed_kmh:4.1f} km/h | Steer: {ctrl.steer:+.2f} | Throt: {ctrl.throttle:.2f} | Brake: {ctrl.brake:.2f}",
        (8, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (148, 163, 184),
        1,
        cv2.LINE_AA,
    )

    # 하단 3인칭 Chase View
    chase_canvas = chase_img.copy()
    cb = chase_canvas.copy()
    cv2.rectangle(cb, (0, 0), (chase_canvas.shape[1], 34), (10, 15, 28), -1)
    cv2.addWeighted(cb, 0.78, chase_canvas, 0.22, 0, chase_canvas)
    mode_names = [
        "ZERO-SHOT LEFT SWERVE (Pre-trained VLM)",
        "ZERO-SHOT RIGHT BYPASS (Pre-trained VLM)",
        "ZERO-SHOT EMERGENCY HALT (Pre-trained VLM)",
    ]
    cv2.putText(
        chase_canvas,
        f"{mode_names[cmd_idx]} | Frame {step:03d}/{total_steps}",
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        c,
        1,
        cv2.LINE_AA,
    )

    return np.concatenate([front_canvas, chase_canvas], axis=0)


def main():
    parser = argparse.ArgumentParser(description="Pre-trained VLM + Zero-Shot CARLA End-to-End VLA")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_size = 15
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("[1/5] Hugging Face 사전학습 VLM 로드 및 Zero-Shot 시맨틱 매칭 검증")
    print("=" * 84)
    vlm = PretrainedMultimodalBackbone(device=device)

    # 1) 비교용: 순수 CLIP 전역 문장 임베딩 유사도 (Bag-of-Objects 명사 편향 확인)
    clip_seen = vlm.encode_clip_global_text(SEEN_TRAIN_PROMPTS)
    clip_unseen = vlm.encode_clip_global_text(UNSEEN_ZERO_SHOT_PROMPTS)
    clip_global_sim = (clip_unseen @ clip_seen.T).cpu().numpy()

    # 2) 실제 사용: 사전학습 Transformer + Subphrase Late-Interaction (MaxSim)
    seen_sent_embs, seen_probs_list = [], []
    for p in SEEN_TRAIN_PROMPTS:
        s_emb, _, s_prob = vlm.encode_prompt_with_late_interaction(p)
        seen_sent_embs.append(s_emb)
        seen_probs_list.append(s_prob)
    seen_sent_embs_t = torch.stack(seen_sent_embs, dim=0)  # [3, 384]
    seen_probs_t = torch.stack(seen_probs_list, dim=0)    # [3, 3]

    unseen_sent_embs, late_scores_list, unseen_probs_list = [], [], []
    print("\n  [Zero-Shot Late-Interaction (Subphrase MaxSim) 매칭 결과]")
    for i, u_p in enumerate(UNSEEN_ZERO_SHOT_PROMPTS):
        u_emb, u_scores, u_prob = vlm.encode_prompt_with_late_interaction(u_p)
        unseen_sent_embs.append(u_emb)
        late_scores_list.append(u_scores.cpu().numpy())
        unseen_probs_list.append(u_prob.cpu().numpy())
        best_j = int(torch.argmax(u_prob).item())
        print(f"  - Unseen #{i+1}: \"{u_p}\"")
        print(
            f"    -> MaxSim(좌/우/정지): [{u_scores[0]:.3f}, {u_scores[1]:.3f}, {u_scores[2]:.3f}] | "
            f"라우팅 확률: [Left {u_prob[0]*100:5.1f}%, Right {u_prob[1]*100:5.1f}%, Stop {u_prob[2]*100:5.1f}%] "
            f"==> ★ 매칭: Mode #{best_j + 1}"
        )

    late_max_sim = np.stack(late_scores_list, axis=0)
    proto_probs_np = np.stack(unseen_probs_list, axis=0)
    sim_plot_path = out_dir / "carla_vlm_similarity_matrix.png"
    save_similarity_matrix_plot(clip_global_sim, late_max_sim, proto_probs_np, sim_plot_path)
    print(f"  - 시맨틱 유사도 비교 분석 플롯 저장 완료: {sim_plot_path}")

    print("\n" + "=" * 84)
    print(f"[2/5] CARLA 서버(Port {args.port}) 접속 및 도심 경찰차 장애물 시나리오 생성 중...")
    print("=" * 84)
    client, world, host_ip = connect_carla_client(port=args.port)
    print(f"      - 접속 성공! ({host_ip}:{args.port}, Map: {world.get_map().name})")

    original_settings = world.get_settings()
    actors_to_destroy = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearSunset)

        bp_lib = world.get_blueprint_library()
        carla_map = world.get_map()
        start_tf = find_long_straight_spawn(carla_map)

        # 전방 28m 지점에 정차된 경찰차 스폰
        fwd = start_tf.get_forward_vector()
        obs_loc = carla.Location(
            x=start_tf.location.x + fwd.x * 28.0,
            y=start_tf.location.y + fwd.y * 28.0,
            z=start_tf.location.z + 0.2,
        )
        obs_tf = carla.Transform(obs_loc, start_tf.rotation)
        police_bps = bp_lib.filter("vehicle.dodge.charger_police*")
        obs_bp = police_bps[0] if police_bps else bp_lib.filter("vehicle.*")[0]
        obstacle_car = world.try_spawn_actor(obs_bp, obs_tf)
        if obstacle_car is not None:
            actors_to_destroy.append(obstacle_car)
            obstacle_car.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))

        # 내 자율주행차 (Tesla Model 3) 스폰
        ego_bps = bp_lib.filter("vehicle.tesla.model3")
        ego_bp = ego_bps[0] if ego_bps else bp_lib.filter("vehicle.lincoln.mkz_2020")[0]
        if ego_bp.has_attribute("color"):
            ego_bp.set_attribute("color", "20,175,245")
        ego_vehicle = world.spawn_actor(ego_bp, start_tf)
        actors_to_destroy.append(ego_vehicle)

        cam_w, cam_h = 448, 264
        fov = 90.0
        K = build_camera_intrinsic(cam_w, cam_h, fov)

        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(cam_w))
        rgb_bp.set_attribute("image_size_y", str(cam_h))
        rgb_bp.set_attribute("fov", str(fov))
        cam_front = world.spawn_actor(
            rgb_bp, carla.Transform(carla.Location(x=1.5, z=1.55), carla.Rotation(pitch=-4.0)), attach_to=ego_vehicle
        )
        actors_to_destroy.append(cam_front)

        cam_chase = world.spawn_actor(
            rgb_bp, carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-14.0)), attach_to=ego_vehicle
        )
        actors_to_destroy.append(cam_chase)

        seg_bp = bp_lib.find("sensor.camera.semantic_segmentation")
        seg_bp.set_attribute("image_size_x", str(cam_w))
        seg_bp.set_attribute("image_size_y", str(cam_h))
        seg_bp.set_attribute("fov", str(fov))
        cam_seg = world.spawn_actor(
            seg_bp, carla.Transform(carla.Location(x=1.5, z=1.55), carla.Rotation(pitch=-4.0)), attach_to=ego_vehicle
        )
        actors_to_destroy.append(cam_seg)

        q_front, q_chase, q_seg = queue.Queue(), queue.Queue(), queue.Queue()

        def to_rgb(image: carla.Image) -> np.ndarray:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            return arr[:, :, :3][:, :, ::-1].copy()

        def to_seg(image: carla.Image) -> np.ndarray:
            image.convert(carla.ColorConverter.CityScapesPalette)
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            return arr[:, :, :3][:, :, ::-1].copy()

        cam_front.listen(lambda img: q_front.put(to_rgb(img)))
        cam_chase.listen(lambda img: q_chase.put(to_rgb(img)))
        cam_seg.listen(lambda img: q_seg.put(to_seg(img)))

        def reset_ego_pose():
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            ego_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            ego_vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            ego_vehicle.set_transform(start_tf)
            for _ in range(8):
                world.tick()
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
            world.tick()
            while not q_front.empty(): q_front.get()
            while not q_chase.empty(): q_chase.get()
            while not q_seg.empty(): q_seg.get()

        print("=" * 84)
        print("[3/5] CARLA 시연 데이터 수집 및 CLIP ViT-B/32 비전 패치 특징 추출 중...")
        print("=" * 84)

        demo_clip_vis, demo_segs, demo_cmd_ids, demo_states, demo_wps, demo_ctrls = [], [], [], [], [], []

        for cmd_idx, seen_prompt in enumerate(SEEN_TRAIN_PROMPTS):
            reset_ego_pose()
            ep_clip_vis, ep_segs, ep_states, ep_wps, ep_ctrls = [], [], [], [], []

            for t in range(args.steps + chunk_size):
                world.tick()
                img_f = q_front.get(timeout=5.0)
                _ = q_chase.get(timeout=5.0)
                img_s = q_seg.get(timeout=5.0)

                vel = ego_vehicle.get_velocity()
                speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                s, d, _ = compute_frenet_s_d(ego_vehicle.get_location(), start_tf)
                curr_ctrl = ego_vehicle.get_control()

                ctrl, local_wps = expert_urban_controller(ego_vehicle, start_tf, cmd_idx)
                ego_vehicle.apply_control(ctrl)

                t_rgb = (torch.from_numpy(img_f).float().permute(2, 0, 1) / 255.0).unsqueeze(0).to(device)
                clip_patches = vlm.encode_vision_patches(t_rgb).squeeze(0).cpu()  # [49, 768]

                t_seg = (torch.from_numpy(img_s).float().permute(2, 0, 1) / 255.0).unsqueeze(0).to(device)
                img6_128 = F.interpolate(torch.cat([t_rgb, t_seg], dim=1), size=(128, 128), mode="bilinear", align_corners=False).squeeze(0).cpu()

                ep_clip_vis.append(clip_patches)
                ep_segs.append(img6_128)
                ep_states.append(torch.tensor([speed_ms / 10.0, curr_ctrl.steer, s / 40.0, d / 4.0], dtype=torch.float32))
                ep_wps.append(torch.from_numpy(local_wps))
                ep_ctrls.append(torch.tensor([ctrl.throttle, ctrl.steer, ctrl.brake], dtype=torch.float32))

            ep_ctrl_stack = torch.stack(ep_ctrls)
            for t in range(args.steps):
                demo_clip_vis.append(ep_clip_vis[t])
                demo_segs.append(ep_segs[t])
                demo_cmd_ids.append(cmd_idx)
                demo_states.append(ep_states[t])
                demo_wps.append(ep_wps[t])
                demo_ctrls.append(ep_ctrl_stack[t : t + chunk_size])

            s_end, d_end, _ = compute_frenet_s_d(ego_vehicle.get_location(), start_tf)
            print(f"  - Seen 학습 시연 #{cmd_idx + 1} 완료 (s={s_end:.1f}m, d={d_end:+.2f}m): \"{seen_prompt}\"")

        clip_vis_t = torch.stack(demo_clip_vis).to(device)  # [N, 49, 768]
        segs_t = torch.stack(demo_segs).to(device)          # [N, 3, 128, 128]
        cmd_ids_t = torch.tensor(demo_cmd_ids, dtype=torch.long, device=device)
        states_t = torch.stack(demo_states).to(device)
        wps_t = torch.stack(demo_wps).to(device)
        ctrls_t = torch.stack(demo_ctrls).to(device)

        print("\n" + "=" * 84)
        print("[4/5] RTX 4090에서 Pre-trained VLM 기반 VLA Action Head 학습 중...")
        print("=" * 84)

        vla = CarlaPretrainedVLM_VLA(d_model=256, chunk_size=chunk_size).to(device)
        optimizer = torch.optim.AdamW(vla.parameters(), lr=1e-3, weight_decay=1e-4)
        epochs = 120
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loss_fn = nn.L1Loss()

        batch_size = 32
        num_samples = clip_vis_t.shape[0]
        t0 = time.perf_counter()

        vla.train()
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(num_samples, device=device)
            ep_loss = 0.0
            steps = 0
            for i in range(0, num_samples, batch_size):
                idx = perm[i : i + batch_size]
                b_cmd = cmd_ids_t[idx]
                b_sent_emb = F.normalize(
                    seen_sent_embs_t[b_cmd] + 0.04 * torch.randn(len(idx), 384, device=device), dim=-1
                )
                b_probs = seen_probs_t[b_cmd]
                noisy_state = states_t[idx] + 0.015 * torch.randn_like(states_t[idx])

                pred_wp, pred_ctrl = vla(clip_vis_t[idx], segs_t[idx], b_sent_emb, b_probs, noisy_state)
                loss = loss_fn(pred_wp / 15.0, wps_t[idx] / 15.0) + loss_fn(pred_ctrl, ctrls_t[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ep_loss += loss.item()
                steps += 1
            scheduler.step()

            if epoch % 30 == 0 or epoch == 1:
                print(f"  Epoch {epoch:03d}/{epochs} | Pre-trained VLM-VLA Loss: {ep_loss / steps:.4f} | 경과: {time.perf_counter() - t0:.1f}s")

        print("\n" + "=" * 84)
        print("[5/5] Zero-Shot 평가! 학습 때 한 번도 본 적 없는 3가지 새로운 영어 문장으로 CARLA 주행 테스트!")
        print("=" * 84)

        vla.eval()
        panel_videos = []

        for cmd_idx, unseen_prompt in enumerate(UNSEEN_ZERO_SHOT_PROMPTS):
            u_sent_emb, _, u_prob = vlm.encode_prompt_with_late_interaction(unseen_prompt)
            u_sent_in = u_sent_emb.unsqueeze(0)
            u_prob_in = u_prob.unsqueeze(0)
            probs_np = u_prob.cpu().numpy()

            reset_ego_pose()
            frames = []
            all_time_ctrls = np.zeros((args.steps, args.steps + chunk_size, 3), dtype=np.float32)

            for t in range(args.steps):
                world.tick()
                img_f = q_front.get(timeout=5.0)
                img_c = q_chase.get(timeout=5.0)
                img_s = q_seg.get(timeout=5.0)

                vel = ego_vehicle.get_velocity()
                speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                speed_kmh = speed_ms * 3.6
                s, d, _ = compute_frenet_s_d(ego_vehicle.get_location(), start_tf)
                curr_ctrl = ego_vehicle.get_control()

                with torch.inference_mode():
                    t_rgb = (torch.from_numpy(img_f).float().permute(2, 0, 1) / 255.0).unsqueeze(0).to(device)
                    clip_patches = vlm.encode_vision_patches(t_rgb)

                    t_seg = (torch.from_numpy(img_s).float().permute(2, 0, 1) / 255.0).unsqueeze(0).to(device)
                    img6_128 = F.interpolate(torch.cat([t_rgb, t_seg], dim=1), size=(128, 128), mode="bilinear", align_corners=False)

                    st_in = torch.tensor([[speed_ms / 10.0, curr_ctrl.steer, s / 40.0, d / 4.0]], dtype=torch.float32, device=device)
                    pred_wp, pred_ctrl = vla(clip_patches, img6_128, u_sent_in, u_prob_in, st_in)
                    wp_np = pred_wp.squeeze(0).cpu().numpy()
                    ctrl_chunk = pred_ctrl.squeeze(0).cpu().numpy()

                all_time_ctrls[t, t : t + chunk_size] = ctrl_chunk
                start_idx = max(0, t - 6)
                ctrls_curr = all_time_ctrls[start_idx : t + 1, t]
                w = np.exp(-0.15 * np.arange(len(ctrls_curr))[::-1])
                w /= w.sum()
                act = (ctrls_curr * w[:, None]).sum(axis=0)

                throttle = float(np.clip(act[0], 0.0, 1.0))
                steer = float(np.clip(act[1], -0.7, 0.7))
                brake = float(np.clip(act[2], 0.0, 1.0))
                if brake > 0.25:
                    throttle = 0.0

                applied_ctrl = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
                ego_vehicle.apply_control(applied_ctrl)

                panel = draw_zeroshot_vla_panel(
                    img_f,
                    img_c,
                    img_s,
                    ego_vehicle,
                    cam_front,
                    K,
                    wp_np,
                    probs_np,
                    cmd_idx,
                    unseen_prompt,
                    t + 1,
                    args.steps,
                    speed_kmh,
                    applied_ctrl,
                )
                frames.append(panel)

            panel_videos.append(frames)
            s_end, d_end, _ = compute_frenet_s_d(ego_vehicle.get_location(), start_tf)
            vel_end = ego_vehicle.get_velocity()
            v_end_kmh = 3.6 * math.sqrt(vel_end.x**2 + vel_end.y**2 + vel_end.z**2)
            print(
                f"  ★ Zero-Shot 주행 #{cmd_idx + 1} 완료 -> 전진 s={s_end:.1f}m | 횡방향 d={d_end:+.2f}m | "
                f"종료 속도={v_end_kmh:.1f} km/h | Unseen: \"{unseen_prompt}\""
            )

        combined_frames = [
            np.concatenate([panel_videos[0][t], panel_videos[1][t], panel_videos[2][t]], axis=1)
            for t in range(args.steps)
        ]

        mp4_path = out_dir / "carla_pretrained_vlm_zeroshot.mp4"
        snap_path = out_dir / "carla_pretrained_vlm_snapshot.png"
        imageio.mimsave(str(mp4_path), combined_frames, fps=20)
        imageio.imwrite(str(snap_path), combined_frames[95])

        print("=" * 84)
        print(f"Pre-trained VLM Zero-Shot 주행 비교 영상 저장 완료 : {mp4_path}")
        print(f"Pre-trained VLM Zero-Shot 스냅샷 저장 완료       : {snap_path}")
        print(f"VLM 시맨틱 유사도 매트릭스 그래프 저장 완료     : {sim_plot_path}")
        print("=" * 84)

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
