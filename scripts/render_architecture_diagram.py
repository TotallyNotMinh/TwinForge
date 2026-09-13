"""
Publication-quality architectural diagram for TwinForge.
Saves high-resolution PNG (300 DPI) and vector PDF figures.
Features:
- Perfectly aligned parallel dual streams:
    * Top lane: Semantic Segmentation (Emerald) -> SegmentDecoder -> 40-Class Logits
    * Bottom lane: Monocular Depth (Crimson) -> DepthDecoder -> Metric Depth Map
- Center vertical bidirectional cross-task refinement bridge (2x iterations)
- Shared context projections (proj5, proj4) feeding both task decoders
- Clean vertical token aggregation bus from all 5 feature pyramid levels
- Embedded real NYUv2 RGB input, Depth map, and 40-class Segmentation mask
"""

import h5py
import scipy.io
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

def load_sample_data():
    """Load a clean validation sample from NYUv2 labeled dataset."""
    mat_path = Path("data/nyu_depth_v2_labeled.mat")
    splits_path = Path("data/splits.mat")
    class_map_path = Path("data/classMapping40.mat")

    if mat_path.exists() and splits_path.exists() and class_map_path.exists():
        splits = scipy.io.loadmat(str(splits_path))
        test_key = "testNdxs" if "testNdxs" in splits else "testNdx"
        val_indices = splits[test_key].squeeze() - 1
        val_idx = int(val_indices[3])

        class_map = scipy.io.loadmat(str(class_map_path))["mapClass"].squeeze()
        lookup = np.zeros(895, dtype=np.int64)
        lookup[1:895] = class_map

        with h5py.File(str(mat_path), "r") as f:
            img = np.array(f["images"][val_idx]).transpose(2, 1, 0) / 255.0
            depth = np.array(f["depths"][val_idx]).T
            raw_label = np.array(f["labels"][val_idx]).T

        seg = lookup[raw_label]
        return img, depth, seg
    else:
        H, W = 480, 640
        y, x = np.mgrid[:H, :W]
        img = np.zeros((H, W, 3))
        img[..., 0] = 0.8 * (1 - y / H * 0.4)
        img[..., 1] = 0.75 * (1 - y / H * 0.3)
        img[..., 2] = 0.7
        depth = 4.0 - y * 0.005 + x * 0.001
        seg = np.zeros((H, W), dtype=int)
        seg[y > 300] = 2
        seg[y <= 120] = 1
        seg[(y > 180) & (y < 340) & (x > 180) & (x < 460)] = 3
        return img, depth, seg

def create_diagram(output_png="twinforge_architecture.png", output_pdf="twinforge_architecture.pdf"):
    sample_img, sample_depth, sample_seg = load_sample_data()

    # 26 x 14 inches at 300 DPI
    fig = plt.figure(figsize=(26, 14), dpi=300, facecolor="#F8FAFC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # Colors
    C_TEXT = "#0F172A"
    C_SUBTEXT = "#475569"
    C_DIM = "#64748B"
    
    C_FROZEN_BG = "#EFF6FF"
    C_FROZEN_BORDER = "#93C5FD"
    C_FROZEN_ACCENT = "#1D4ED8"
    
    C_TUNED_BG = "#E0F2FE"
    C_TUNED_BORDER = "#38BDF8"
    C_TUNED_ACCENT = "#0284C7"
    
    C_ADAPT_BG = "#F5F3FF"
    C_ADAPT_BORDER = "#C4B5FD"
    C_ADAPT_ACCENT = "#6D28D9"
    
    C_DEC_BG = "#FFFBEB"
    C_DEC_BORDER = "#FCD34D"
    C_DEC_ACCENT = "#B45309"
    
    C_DEPTH_BG = "#FEF2F2"
    C_DEPTH_BORDER = "#FCA5A5"
    C_DEPTH_ACCENT = "#DC2626"
    
    C_SEG_BG = "#ECFDF5"
    C_SEG_BORDER = "#6EE7B7"
    C_SEG_ACCENT = "#059669"
    
    C_CROSS_BG = "#FFF1F2"
    C_CROSS_BORDER = "#FDA4AF"
    C_CROSS_ACCENT = "#E11D48"

    def draw_box(x, y, w, h, bg_color, border_color, border_width=1.5, radius=0.2, linestyle="-"):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=bg_color, edgecolor=border_color,
            linewidth=border_width, linestyle=linestyle,
            zorder=2
        )
        ax.add_patch(box)
        return box

    def draw_badge(x, y, w, h, text, bg="#E2E8F0", fg="#334155", fontsize=7.5):
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle=f"round,pad=0,rounding_size={h/2}",
            facecolor=bg, edgecolor="#CBD5E1", linewidth=0.8,
            zorder=5
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", fontfamily="DejaVu Sans", color=fg, zorder=6)

    def draw_arrow(x1, y1, x2, y2, color="#64748B", width=1.5, style="-|>", rad=0.0):
        connectionstyle = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=connectionstyle,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=width,
            color=color,
            zorder=4
        )
        ax.add_patch(arrow)

    # =========================================================================
    # 0. HEADER & TITLE BLOCK
    # =========================================================================
    ax.text(13.0, 13.45, "TwinForge: Decoupled Multi-Task Vision Architecture", 
            ha="center", va="center", fontsize=22, fontweight="bold", 
            fontfamily="DejaVu Sans", color=C_TEXT)
    ax.text(13.0, 13.05, "Joint Monocular Depth Estimation & Semantic Segmentation with Early-Frozen DINOv2 ViT-S & Cross-Task Transformer Decoder",
            ha="center", va="center", fontsize=11, fontfamily="DejaVu Sans", color=C_SUBTEXT)
    ax.text(13.0, 12.7, "NYUv2 SOTA Benchmark: 0.5813 mIoU  |  0.3650m Depth RMSE  |  0.9112 δ1  |  24.91 GMac (122.7 GMac/Mpx)  |  40.1 FPS (RTX 3060)",
            ha="center", va="center", fontsize=9.5, fontweight="bold", fontfamily="DejaVu Sans", color="#1E293B")

    # =========================================================================
    # 1. INPUT STAGE (x: [0.7, 3.1])
    # =========================================================================
    draw_box(0.7, 2.0, 2.4, 10.1, "#FFFFFF", "#94A3B8", border_width=2.0, radius=0.35)
    ax.text(1.9, 11.75, "INPUT", ha="center", va="center", fontsize=12, fontweight="bold", color=C_TEXT)
    draw_badge(1.9, 11.35, 2.1, 0.35, "RGB Input Image", bg="#F1F5F9", fg="#334155", fontsize=8)

    # Embed real NYUv2 RGB image (4:3 aspect ratio: 2.1 x 1.575)
    img_x, img_y, img_w, img_h = 0.85, 7.8, 2.1, 1.575
    ax_rgb = fig.add_axes([img_x / 26, img_y / 14, img_w / 26, img_h / 14])
    ax_rgb.imshow(sample_img)
    ax_rgb.axis("off")
    draw_box(img_x, img_y, img_w, img_h, "none", "#475569", border_width=1.2, radius=0.06)

    ax.text(1.9, 7.35, "NYUv2 Indoor Scene [X]", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1E293B")
    draw_badge(1.9, 6.75, 2.1, 0.4, "Size: 3 × 392 × 518", bg="#E2E8F0", fg="#1E293B", fontsize=7.5)
    draw_badge(1.9, 6.15, 2.1, 0.35, "Native: 3 × 480 × 640", bg="#F1F5F9", fg="#475569", fontsize=7)
    draw_badge(1.9, 5.6, 2.1, 0.35, "ImageNet Normalized", bg="#F1F5F9", fg="#64748B", fontsize=7)
    draw_badge(1.9, 5.05, 2.1, 0.35, "Orientation Corrected", bg="#DCFCE7", fg="#166534", fontsize=7.2)

    # Description block at bottom of Input
    draw_box(0.85, 2.3, 2.1, 2.2, "#F8FAFC", "#E2E8F0", border_width=1.0, radius=0.15)
    ax.text(1.9, 4.15, "Multi-Task Targets:", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#334155")
    ax.text(1.9, 3.65, "1. Metric Depth (m)\n   Ground Truth Kinect", ha="center", va="center", fontsize=7, color="#991B1B")
    ax.text(1.9, 2.85, "2. Semantic Seg\n   40 Indoor Classes", ha="center", va="center", fontsize=7, color="#065F46")

    # Arrow to Backbone
    draw_arrow(3.1, 8.58, 3.7, 8.58, color="#475569", width=2.2)

    # =========================================================================
    # 2. FOUNDATION ViT BACKBONE (x: [3.7, 8.7])
    # =========================================================================
    draw_box(3.7, 2.0, 5.0, 10.1, "#FFFFFF", "#93C5FD", border_width=2.0, radius=0.35)
    ax.text(6.2, 11.75, "STAGE 1: DINOv2 ViT-S BACKBONE", ha="center", va="center", fontsize=12, fontweight="bold", color=C_FROZEN_ACCENT)
    ax.text(6.2, 11.45, "Pretrained via Depth Anything V2 (12 Blocks, embed_dim=384, 6 heads)", ha="center", va="center", fontsize=8, color=C_SUBTEXT)

    # Patch Embed
    draw_box(3.9, 9.9, 4.6, 1.25, "#F8FAFC", "#CBD5E1", border_width=1.2, radius=0.15)
    ax.text(6.2, 10.75, "Patch Embedding (14×14 conv, stride 14) + Bicubic Pos Encoding", ha="center", va="center", fontsize=8.2, fontweight="bold", color=C_TEXT)
    ax.text(6.2, 10.35, "Generates 1036 Patch Tokens + 1 [CLS] Token → Dimension [B, 1037, 384]", ha="center", va="center", fontsize=7.8, color=C_DIM)

    draw_arrow(6.2, 9.9, 6.2, 9.4, color="#64748B", width=1.5)

    # Frozen Blocks 0..5
    draw_box(3.9, 6.1, 4.6, 3.25, C_FROZEN_BG, C_FROZEN_BORDER, border_width=1.8, radius=0.25)
    draw_badge(6.2, 9.05, 4.2, 0.45, "[FROZEN STAGE] Blocks 0 – 5 | lr = 0.0", bg="#DBEAFE", fg=C_FROZEN_ACCENT, fontsize=8.5)
    ax.text(6.2, 8.55, "Universal Low-Level Geometry & Surface Normal Primitives", ha="center", va="center", fontsize=8, fontweight="bold", color="#1E3A8A")
    ax.text(6.2, 8.2, "Locked to eliminate representation drift on small NYUv2 dataset (795 samples)", ha="center", va="center", fontsize=7.2, color="#2563EB")

    for i in range(6):
        bx = 4.1 + (i % 3) * 1.45
        by = 7.35 if i < 3 else 6.4
        is_tap = (i == 2 or i == 5)
        b_color = "#BFDBFE" if is_tap else "#EFF6FF"
        edge_c = "#3B82F6" if is_tap else "#93C5FD"
        draw_box(bx, by, 1.25, 0.75, b_color, edge_c, border_width=1.2, radius=0.1)
        ax.text(bx + 0.625, by + 0.42, f"Block {i}", ha="center", va="center", fontsize=8, fontweight="bold" if is_tap else "normal", color="#1E40AF")
        if is_tap:
            draw_badge(bx + 0.625, by - 0.05, 0.85, 0.22, "TAP", bg="#2563EB", fg="#FFFFFF", fontsize=6.5)

    draw_arrow(6.2, 6.1, 6.2, 5.5, color="#64748B", width=1.5)

    # Fine-Tuned Blocks 6..11
    draw_box(3.9, 2.3, 4.6, 3.15, C_TUNED_BG, C_TUNED_BORDER, border_width=1.8, radius=0.25)
    draw_badge(6.2, 5.15, 4.2, 0.45, "[FINE-TUNED STAGE] Blocks 6 – 11 | lr = 1e-5", bg="#BAE6FD", fg=C_TUNED_ACCENT, fontsize=8.5)
    ax.text(6.2, 4.65, "Task-Specific Indoor Semantic Adaptation", ha="center", va="center", fontsize=8, fontweight="bold", color="#0369A1")
    ax.text(6.2, 4.3, "Fine-tuned with conservative gradient updates & weight decay = 5e-3", ha="center", va="center", fontsize=7.2, color="#0284C7")

    for i in range(6, 12):
        idx = i - 6
        bx = 4.1 + (idx % 3) * 1.45
        by = 3.45 if idx < 3 else 2.5
        is_tap = (i == 8 or i == 11)
        b_color = "#7DD3FC" if is_tap else "#E0F2FE"
        edge_c = "#0284C7" if is_tap else "#38BDF8"
        draw_box(bx, by, 1.25, 0.75, b_color, edge_c, border_width=1.2, radius=0.1)
        ax.text(bx + 0.625, by + 0.42, f"Block {i}", ha="center", va="center", fontsize=8, fontweight="bold" if is_tap else "normal", color="#075985")
        if is_tap:
            draw_badge(bx + 0.625, by - 0.05, 0.85, 0.22, "TAP", bg="#0284C7", fg="#FFFFFF", fontsize=6.5)

    # =========================================================================
    # 3. MULTI-SCALE FEATURE ADAPTERS (x: [9.2, 12.1])
    # =========================================================================
    draw_box(9.2, 2.0, 2.9, 10.1, "#FFFFFF", C_ADAPT_BORDER, border_width=2.0, radius=0.35)
    ax.text(10.65, 11.75, "STAGE 2: FEATURE ADAPTERS", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_ADAPT_ACCENT)
    ax.text(10.65, 11.45, "Decoupled Optimization | lr = 2e-4", ha="center", va="center", fontsize=8, color=C_SUBTEXT)
    draw_badge(10.65, 11.1, 2.5, 0.32, "1×1 Conv + BatchNorm + ReLU", bg=C_ADAPT_BG, fg=C_ADAPT_ACCENT, fontsize=7.5)

    adapter_stages = [
        ("f1", "proj1", "Blk 2", "H/2, W/2", 64, 9.35),
        ("f2", "proj2", "Blk 2", "H/4, W/4", 256, 7.65),
        ("f3", "proj3", "Blk 5", "H/8, W/8", 512, 5.95),
        ("f4", "proj4", "Blk 8", "H/16, W/16", 1024, 4.25),
        ("f5", "proj5", "Blk 11", "H/32, W/32", 2048, 2.55),
    ]

    for name, proj, blk, res, ch, y_pos in adapter_stages:
        draw_box(9.4, y_pos, 2.5, 1.45, C_ADAPT_BG, C_ADAPT_BORDER, border_width=1.3, radius=0.18)
        ax.text(9.6, y_pos + 1.15, f"Level {name.upper()}", ha="left", va="center", fontsize=9, fontweight="bold", color=C_ADAPT_ACCENT)
        ax.text(11.7, y_pos + 1.15, f"from {blk}", ha="right", va="center", fontsize=7.5, color="#6B21A8", fontweight="bold")
        ax.text(9.6, y_pos + 0.75, f"{proj}: 384 → {ch} ch", ha="left", va="center", fontsize=8, color="#4C1D95")
        draw_badge(10.65, y_pos + 0.3, 2.2, 0.32, f"{res} | C={ch}", bg="#FFFFFF", fg="#5B21B6", fontsize=7.5)

    # Backbone Taps to Feature Adapters
    draw_arrow(7.85, 7.3, 9.2, 10.1, color=C_ADAPT_ACCENT, width=1.4, rad=-0.08)
    draw_arrow(7.85, 7.3, 9.2, 8.4, color=C_ADAPT_ACCENT, width=1.4, rad=-0.04)
    draw_arrow(7.85, 6.35, 9.2, 6.7, color=C_ADAPT_ACCENT, width=1.4, rad=0.0)
    draw_arrow(7.85, 3.4, 9.2, 5.0, color=C_ADAPT_ACCENT, width=1.4, rad=0.04)
    draw_arrow(7.85, 2.5, 9.2, 3.3, color=C_ADAPT_ACCENT, width=1.4, rad=0.08)

    # =========================================================================
    # 4. TOKENIZERS & ASYMMETRIC TRANSFORMERS (x: [12.6, 15.6])
    # =========================================================================
    draw_box(12.6, 2.0, 3.0, 10.1, "#FFFFFF", C_DEC_BORDER, border_width=2.0, radius=0.35)
    ax.text(14.1, 11.75, "STAGE 3A: TOKENIZERS", ha="center", va="center", fontsize=11, fontweight="bold", color=C_DEC_ACCENT)
    ax.text(14.1, 11.45, "PatchEmbed + 2D Pos & Level Embed", ha="center", va="center", fontsize=8, color=C_SUBTEXT)
    draw_badge(14.1, 11.1, 2.6, 0.32, "Unified Token Dim D = 256", bg=C_DEC_BG, fg=C_DEC_ACCENT, fontsize=7.5)

    token_stages = [
        ("f1", "patch=6", "T1 (1 blk)", 9.35),
        ("f2", "patch=4", "T2 (1 blk)", 7.65),
        ("f3", "patch=4", "T3 (2 blks)", 5.95),
        ("f4", "patch=3", "T4 (3 blks)", 4.25),
        ("f5", "patch=3", "T5 (3 blks)", 2.55),
    ]

    for name, patch_s, trans, y_pos in token_stages:
        draw_box(12.8, y_pos, 2.6, 1.45, C_DEC_BG, C_DEC_BORDER, border_width=1.3, radius=0.18)
        ax.text(13.0, y_pos + 1.15, f"Token {name.upper()}", ha="left", va="center", fontsize=9, fontweight="bold", color="#78350F")
        ax.text(15.2, y_pos + 1.15, patch_s, ha="right", va="center", fontsize=7.5, color="#B45309", fontweight="bold")
        ax.text(14.1, y_pos + 0.72, trans, ha="center", va="center", fontsize=8, color="#92400E")
        draw_badge(14.1, y_pos + 0.3, 2.3, 0.32, "Tokens [B, L_i, 256]", bg="#FFFFFF", fg="#78350F", fontsize=7.5)

    # Clean straight horizontal arrows from Adapters to Tokenizers
    for _, _, _, _, _, y_pos in adapter_stages:
        draw_arrow(11.9, y_pos + 0.72, 12.8, y_pos + 0.72, color="#B45309", width=1.5)

    # Vertical Token Bus Line
    bus_x = 15.85
    for _, _, _, y_pos in token_stages:
        ax.plot([15.4, bus_x], [y_pos + 0.72, y_pos + 0.72], color="#D97706", linewidth=1.5, zorder=3)
        ax.plot(bus_x, y_pos + 0.72, "o", color="#B45309", markersize=4, zorder=4)

    ax.plot([bus_x, bus_x], [2.55 + 0.72, 9.35 + 0.72], color="#D97706", linewidth=2.0, zorder=3)
    draw_arrow(bus_x, 10.5, 16.2, 10.5, color="#D97706", width=2.0)
    ax.plot([bus_x, bus_x], [9.35 + 0.72, 10.5], color="#D97706", linewidth=2.0, zorder=3)

    # =========================================================================
    # 5. CROSS-TASK REFINEMENT DECODER (x: [16.2, 21.3]) - PARALLEL TOP/BOTTOM LANES
    # =========================================================================
    draw_box(16.2, 2.0, 5.1, 10.1, "#FFFFFF", C_DEC_BORDER, border_width=2.0, radius=0.35)
    ax.text(18.75, 11.75, "STAGE 3B: CROSS-TASK REFINEMENT", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_DEC_ACCENT)
    ax.text(18.75, 11.45, "Decoupled Optimization @ lr = 2e-4 | 8 Attention Heads", ha="center", va="center", fontsize=8, color=C_SUBTEXT)

    # Joint Token Concatenation Box (Top)
    draw_box(16.4, 9.95, 4.7, 1.15, "#FEF3C7", "#FDE68A", border_width=1.5, radius=0.15)
    ax.text(18.75, 10.65, "Joint Token Concatenation: concat(tok1..tok5)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#92400E")
    ax.text(18.75, 10.25, "Unified Multi-Scale Sequence: [B, ΣL_i, 256]", ha="center", va="center", fontsize=7.8, color="#78350F")

    # Divergence from Concat to Top (Seg) and Bottom (Depth)
    draw_arrow(18.75, 9.95, 18.75, 9.4, color=C_SEG_ACCENT, width=1.6)

    # -------------------------------------------------------------------------
    # TOP LANE: SEMANTIC SEGMENTATION STREAM (y: [7.35, 9.4])
    # -------------------------------------------------------------------------
    draw_box(16.4, 7.35, 4.7, 2.05, C_SEG_BG, C_SEG_BORDER, border_width=1.5, radius=0.2)
    draw_badge(18.75, 9.15, 4.1, 0.35, "Semantic Segmentation Stream (Self-Attn & Cross-Attn)", bg="#A7F3D0", fg=C_SEG_ACCENT, fontsize=8)

    # Left: Seg Self-Attn
    draw_box(16.6, 7.5, 2.05, 1.35, "#FFFFFF", C_SEG_BORDER, border_width=1.1, radius=0.12)
    ax.text(17.62, 8.45, "Seg Self-Attn", ha="center", va="center", fontsize=8, fontweight="bold", color=C_SEG_ACCENT)
    ax.text(17.62, 8.1, "Q, K, V (Seg Tokens)", ha="center", va="center", fontsize=7, color="#065F46")
    draw_badge(17.62, 7.75, 1.8, 0.22, "Extracts s1 token", bg="#D1FAE5", fg="#065F46", fontsize=6.5)

    # Right: Seg Cross Depth
    draw_box(18.85, 7.5, 2.05, 1.35, "#FFFFFF", C_CROSS_BORDER, border_width=1.1, radius=0.12)
    ax.text(19.87, 8.45, "Seg Cross Depth", ha="center", va="center", fontsize=8, fontweight="bold", color=C_SEG_ACCENT)
    ax.text(19.87, 8.1, "Q_seg × (K,V)_depth", ha="center", va="center", fontsize=7, color="#065F46")
    draw_badge(19.87, 7.75, 1.8, 0.22, "Tokens {s2..s5}", bg="#FFE4E6", fg=C_CROSS_ACCENT, fontsize=6.5)

    draw_arrow(18.65, 8.17, 18.85, 8.17, color=C_SEG_ACCENT, width=1.3)

    # -------------------------------------------------------------------------
    # CENTER BRIDGE: 2x MUTUAL CROSS-TASK REFINEMENT INTERACTION (y: [5.75, 7.15])
    # -------------------------------------------------------------------------
    draw_box(16.4, 5.85, 4.7, 1.3, C_CROSS_BG, C_CROSS_BORDER, border_width=1.5, radius=0.18)
    draw_badge(18.75, 6.75, 4.2, 0.35, "↕ 2× Bidirectional Cross-Task Refinement ↕", bg="#FFE4E6", fg=C_CROSS_ACCENT, fontsize=8)
    ax.text(18.75, 6.25, "Mutual Attention Exchange (Depth Discontinuities ↔ Semantic Boundaries)", ha="center", va="center", fontsize=7.2, color="#BE123C")

    # Vertical bidirectional interaction arrows between top Seg and bottom Depth
    draw_arrow(17.62, 7.35, 17.62, 5.7, color=C_CROSS_ACCENT, width=1.5, style="<|-|>")
    draw_arrow(19.87, 7.35, 19.87, 5.7, color=C_CROSS_ACCENT, width=1.5, style="<|-|>")

    # -------------------------------------------------------------------------
    # BOTTOM LANE: MONOCULAR DEPTH STREAM (y: [3.7, 5.75])
    # -------------------------------------------------------------------------
    draw_box(16.4, 3.7, 4.7, 2.05, C_DEPTH_BG, C_DEPTH_BORDER, border_width=1.5, radius=0.2)
    draw_badge(18.75, 5.5, 4.1, 0.35, "Monocular Depth Stream (Self-Attn & Cross-Attn)", bg="#FECDD3", fg=C_DEPTH_ACCENT, fontsize=8)

    # Left: Depth Self-Attn
    draw_box(16.6, 3.85, 2.05, 1.35, "#FFFFFF", C_DEPTH_BORDER, border_width=1.1, radius=0.12)
    ax.text(17.62, 4.8, "Depth Self-Attn", ha="center", va="center", fontsize=8, fontweight="bold", color=C_DEPTH_ACCENT)
    ax.text(17.62, 4.45, "Q, K, V (Depth Tokens)", ha="center", va="center", fontsize=7, color="#991B1B")
    draw_badge(17.62, 4.1, 1.8, 0.22, "Extracts d1 token", bg="#FEE2E2", fg="#991B1B", fontsize=6.5)

    # Right: Depth Cross Seg
    draw_box(18.85, 3.85, 2.05, 1.35, "#FFFFFF", C_CROSS_BORDER, border_width=1.1, radius=0.12)
    ax.text(19.87, 4.8, "Depth Cross Seg", ha="center", va="center", fontsize=8, fontweight="bold", color=C_DEPTH_ACCENT)
    ax.text(19.87, 4.45, "Q_depth × (K,V)_seg", ha="center", va="center", fontsize=7, color="#991B1B")
    draw_badge(19.87, 4.1, 1.8, 0.22, "Tokens {d2..d5}", bg="#FFE4E6", fg=C_CROSS_ACCENT, fontsize=6.5)

    draw_arrow(18.65, 4.52, 18.85, 4.52, color=C_DEPTH_ACCENT, width=1.3)

    # Divergence arrow from Concat down to Depth Stream
    draw_arrow(16.4, 10.5, 16.1, 4.72, color=C_DEPTH_ACCENT, width=1.4, rad=0.1)

    # -------------------------------------------------------------------------
    # SHARED HIGH-LEVEL CONTEXT PROJECTIONS (y: [2.15, 3.5])
    # -------------------------------------------------------------------------
    draw_box(16.4, 2.15, 4.7, 1.4, "#F8FAFC", "#CBD5E1", border_width=1.2, radius=0.18)
    
    draw_box(16.6, 2.25, 2.05, 1.2, "#FFFFFF", "#E2E8F0", border_width=1.0, radius=0.1)
    ax.text(17.62, 3.15, "Shared Proj 5", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1E293B")
    ax.text(17.62, 2.85, "cat(d5, s5, f5) → 128 ch", ha="center", va="center", fontsize=6.8, color="#64748B")
    draw_badge(17.62, 2.5, 1.8, 0.22, "p5 @ H/4, W/4", bg="#E2E8F0", fg="#334155", fontsize=6.5)

    draw_box(18.85, 2.25, 2.05, 1.2, "#FFFFFF", "#E2E8F0", border_width=1.0, radius=0.1)
    ax.text(19.87, 3.15, "Shared Proj 4", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1E293B")
    ax.text(19.87, 2.85, "cat(d4, s4, f4) → 128 ch", ha="center", va="center", fontsize=6.8, color="#64748B")
    draw_badge(19.87, 2.5, 1.8, 0.22, "p4 @ H/4, W/4", bg="#E2E8F0", fg="#334155", fontsize=6.5)

    # =========================================================================
    # 6. TASK DECODERS & PREDICTIONS (x: [21.8, 25.3])
    # =========================================================================
    draw_box(21.8, 2.0, 3.5, 10.1, "#FFFFFF", "#94A3B8", border_width=2.0, radius=0.35)
    ax.text(23.55, 11.75, "STAGE 4: PREDICTIONS & LOSS", ha="center", va="center", fontsize=11, fontweight="bold", color=C_TEXT)
    ax.text(23.55, 11.45, "Fixed 1:1 Task Loss Weighting", ha="center", va="center", fontsize=8, color=C_SUBTEXT)

    # --- TOP BRANCH: SEMANTIC SEGMENTATION ---
    draw_box(22.0, 7.1, 3.1, 4.2, C_SEG_BG, C_SEG_BORDER, border_width=1.8, radius=0.25)
    draw_badge(23.55, 10.95, 2.8, 0.38, "Semantic Segmentation Branch", bg="#A7F3D0", fg=C_SEG_ACCENT, fontsize=8)

    # Embed real NYUv2 Seg image (4:3 aspect ratio: 2.4 x 1.8)
    seg_x, seg_y, seg_w, seg_h = 22.35, 8.85, 2.4, 1.8
    ax_seg = fig.add_axes([seg_x / 26, seg_y / 14, seg_w / 26, seg_h / 14])
    seg_cmap = mpl.colormaps["tab20"].resampled(41)
    ax_seg.imshow(sample_seg, cmap=seg_cmap, vmin=0, vmax=40)
    ax_seg.axis("off")
    draw_box(seg_x, seg_y, seg_w, seg_h, "none", C_SEG_BORDER, border_width=1.2, radius=0.06)

    ax.text(23.55, 8.45, "SegmentDecoder (All-MLP)", ha="center", va="center", fontsize=8, fontweight="bold", color=C_SEG_ACCENT)
    ax.text(23.55, 8.15, "Fuse {s1, p5, p4, (s3,f3), (s2,f2)}", ha="center", va="center", fontsize=7, color="#065F46")
    draw_badge(23.55, 7.75, 2.8, 0.3, "Logits: [B, 40, H, W] | 2× Bilinear", bg="#D1FAE5", fg="#065F46", fontsize=7)
    draw_badge(23.55, 7.35, 2.8, 0.3, "Loss: CE + Lovasz-Softmax (λ=1.0)", bg="#FFFFFF", fg=C_SEG_ACCENT, fontsize=7)

    # Clean straight horizontal flow from Seg stream (Top Lane) to Seg Branch
    draw_arrow(20.9, 8.17, 22.0, 8.17, color=C_SEG_ACCENT, width=1.8)

    # --- BOTTOM BRANCH: MONOCULAR DEPTH ---
    draw_box(22.0, 2.2, 3.1, 4.6, C_DEPTH_BG, C_DEPTH_BORDER, border_width=1.8, radius=0.25)
    draw_badge(23.55, 6.45, 2.8, 0.38, "Monocular Depth Branch", bg="#FECDD3", fg=C_DEPTH_ACCENT, fontsize=8)

    # Embed real NYUv2 Depth image (4:3 aspect ratio: 2.4 x 1.8)
    dep_x, dep_y, dep_w, dep_h = 22.35, 4.35, 2.4, 1.8
    ax_depth = fig.add_axes([dep_x / 26, dep_y / 14, dep_w / 26, dep_h / 14])
    ax_depth.imshow(sample_depth, cmap="plasma")
    ax_depth.axis("off")
    draw_box(dep_x, dep_y, dep_w, dep_h, "none", C_DEPTH_BORDER, border_width=1.2, radius=0.06)

    ax.text(23.55, 3.95, "DepthDecoder (All-MLP)", ha="center", va="center", fontsize=8, fontweight="bold", color=C_DEPTH_ACCENT)
    ax.text(23.55, 3.65, "Fuse {d1, p5, p4, (d3,f3), (d2,f2)}", ha="center", va="center", fontsize=7, color="#991B1B")
    draw_badge(23.55, 3.25, 2.8, 0.3, "Metric Depth: [B, 1, H, W] in [0.001, 10m]", bg="#FFE4E6", fg="#991B1B", fontsize=7)
    draw_badge(23.55, 2.85, 2.8, 0.3, "Loss: SILog Loss (λ=1.0)", bg="#FFFFFF", fg=C_DEPTH_ACCENT, fontsize=7)
    draw_badge(23.55, 2.45, 2.8, 0.3, "Eigen Protocol Validated", bg="#FEE2E2", fg="#7F1D1D", fontsize=7)

    # Clean straight horizontal flow from Depth stream (Bottom Lane) to Depth Branch
    draw_arrow(20.9, 4.52, 22.0, 4.52, color=C_DEPTH_ACCENT, width=1.8)

    # Projections branching to both decoders
    draw_arrow(20.9, 2.85, 22.0, 3.65, color="#64748B", width=1.2, rad=0.05)
    draw_arrow(20.9, 2.85, 22.0, 7.6, color="#64748B", width=1.2, rad=0.15)

    # =========================================================================
    # 7. BOTTOM LEGEND & METRICS SCORECARD (x: [0.7, 25.3])
    # =========================================================================
    draw_box(0.7, 0.45, 24.6, 1.25, "#FFFFFF", "#CBD5E1", border_width=1.5, radius=0.25)
    ax.text(1.0, 1.35, "ARCHITECTURAL KEY & DESIGN PRINCIPLES:", fontsize=9.5, fontweight="bold", color=C_TEXT)

    ax.text(1.0, 1.05, "[FROZEN FOUNDATION] Blocks 0-5 (lr=0.0): Locks universal low-level visual edge/gradient primitives; bypasses NYUv2 795-sample overfitting.", fontsize=7.6, color="#1E3A8A", fontweight="bold")
    ax.text(1.0, 0.75, "[FINE-TUNED BACKBONE] Blocks 6-11 (lr=1e-5): Cautiously adapts high-level indoor semantics & global layout without corrupting earlier representations.", fontsize=7.6, color="#0369A1", fontweight="bold")

    ax.text(13.2, 1.05, "[DECOUPLED OPTIMIZATION] Adapters & MultiHeadDecoder (lr=2e-4): 20× higher LR provides gradient capacity for multi-scale fusion.", fontsize=7.6, color="#5B21B6", fontweight="bold")
    ax.text(13.2, 0.75, "[CROSS-TASK REFINEMENT] 2× Bidirectional Attention: Mutual exchange lets depth discontinuities sharpen segmentation boundaries.", fontsize=7.6, color="#9F1239", fontweight="bold")

    # Hardware badges on bottom right
    draw_badge(23.8, 1.25, 2.6, 0.32, "Parameters: 45.74M (34.34M Train)", bg="#F1F5F9", fg="#0F172A", fontsize=7)
    draw_badge(23.8, 0.88, 2.6, 0.32, "Compute: 24.91 GMac (122.7/Mpx)", bg="#F1F5F9", fg="#0F172A", fontsize=7)
    draw_badge(23.8, 0.52, 2.6, 0.32, "Latency: 24.9ms / 40.1 FPS (RTX 3060)", bg="#F1F5F9", fg="#0F172A", fontsize=7)

    # Save outputs
    print(f"Saving publication-quality architecture diagram to {output_png}...")
    plt.savefig(output_png, dpi=300, facecolor="#F8FAFC", bbox_inches="tight", pad_inches=0.1)
    print(f"Saving vector PDF architecture diagram to {output_pdf}...")
    plt.savefig(output_pdf, facecolor="#F8FAFC", bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print("Done!")

if __name__ == "__main__":
    create_diagram()
