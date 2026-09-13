"""
Generate multi-task qualitative inference figures for 100 NYUv2 validation images.
Each figure displays:
  [Original RGB] | [Predicted Metric Depth] | [Predicted Semantic Segmentation]
Includes per-image metrics (RMSE, delta1, mIoU, Pixel Acc) and an HTML interactive gallery.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import h5py
import scipy.io
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torchvision import transforms
from tqdm import tqdm

from data import NYUv2Dataset
from models import TwinForge
from scripts.evaluate import load_model_checkpoint


def get_nyuv2_colormap():
    """Build high-contrast 41-class colormap for NYUv2 segmentation."""
    np.random.seed(42)
    # Class 0: unlabeled (dark slate)
    colors = [[0.12, 0.15, 0.20]]
    # 40 distinct colors derived from tab20 + tab20b
    cmap1 = mpl.colormaps["tab20"].colors
    cmap2 = mpl.colormaps["tab20b"].colors
    combined = list(cmap1) + list(cmap2)
    for i in range(40):
        colors.append(combined[i % len(combined)])
    return ListedColormap(colors)


def compute_sample_metrics(pred_depth, gt_depth, pred_seg, gt_seg, num_classes=41):
    """Compute per-image evaluation metrics."""
    # 1. Depth metrics (valid where gt > 0.001 and gt <= 10.0)
    mask = (gt_depth > 0.001) & (gt_depth <= 10.0) & (~np.isnan(gt_depth)) & (~np.isnan(pred_depth))
    if np.sum(mask) > 0:
        d_gt = gt_depth[mask]
        d_pr = np.clip(pred_depth[mask], 0.001, 10.0)
        
        rmse = np.sqrt(np.mean((d_pr - d_gt) ** 2))
        ratio = np.maximum(d_pr / d_gt, d_gt / d_pr)
        delta1 = np.mean(ratio < 1.25)
    else:
        rmse, delta1 = 0.0, 0.0

    # 2. Segmentation metrics (ignore class 0)
    valid_seg = (gt_seg > 0) & (gt_seg < num_classes)
    if np.sum(valid_seg) > 0:
        s_gt = gt_seg[valid_seg]
        s_pr = pred_seg[valid_seg]
        pixel_acc = np.mean(s_gt == s_pr)

        ious = []
        for c in range(1, num_classes):
            gt_c = (s_gt == c)
            pr_c = (s_pr == c)
            intersection = np.sum(gt_c & pr_c)
            union = np.sum(gt_c | pr_c)
            if union > 0:
                ious.append(intersection / union)
        miou = np.mean(ious) if len(ious) > 0 else 0.0
    else:
        miou, pixel_acc = 0.0, 0.0

    return {
        "rmse": float(rmse),
        "delta1": float(delta1),
        "miou": float(miou),
        "pixel_acc": float(pixel_acc)
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 100 validation inference figures for TwinForge")
    parser.add_argument("--checkpoint-path", type=str, 
                        default="checkpoints/dino-v2-backbone-early-freeze-fixed-projection-layers/best_depth.zip",
                        help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="inference_val_100",
                        help="Directory to save generated inference figures")
    parser.add_argument("--num-images", type=int, default=100,
                        help="Number of validation images to evaluate")
    parser.add_argument("--input-h", type=int, default=392, help="Input height")
    parser.add_argument("--input-w", type=int, default=518, help="Input width")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("       TwinForge Multi-Task Validation Inference Generator")
    print("=" * 70)
    print(f"  • Checkpoint:     {args.checkpoint_path}")
    print(f"  • Output Dir:     {output_dir}")
    print(f"  • Num Images:     {args.num_images}")
    print(f"  • Resolution:     {args.input_h} × {args.input_w}")
    print(f"  • Device:         {device}")
    print("=" * 70)

    # 1. Load Dataset
    resize = (args.input_h, args.input_w)
    dataset = NYUv2Dataset(
        data_path="data/nyu_depth_v2_labeled.mat",
        class_map_path="data/classMapping40.mat",
        split="val",
        resize=resize,
        augment=False,
        return_raw_depth=False
    )

    # 2. Load Model
    model = TwinForge(
        num_labels=41,
        num_heads=8,
        tok_dim=256,
        size=resize,
        freeze=False
    ).to(device)

    load_model_checkpoint(model, args.checkpoint_path, device)
    model.eval()

    # Inverse normalize transform to reconstruct RGB
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225],
    )

    seg_cmap = get_nyuv2_colormap()

    # 3. Inference Loop
    records = []
    num_to_eval = min(args.num_images, len(dataset))

    print(f"\nGenerating inference for {num_to_eval} validation images...")

    for idx in tqdm(range(num_to_eval), desc="Rendering inference"):
        norm_img, gt_depth_tensor, gt_label_tensor = dataset[idx]
        
        # Forward pass
        with torch.no_grad():
            img_tensor = norm_img.unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred_seg_logits, pred_depth_tensor = model(img_tensor)
            
            # Post-process predictions
            pred_seg = pred_seg_logits.argmax(dim=1).squeeze(0).cpu().numpy()
            pred_depth = pred_depth_tensor.squeeze().cpu().numpy()

        # Recover RGB
        rgb_img = inv_normalize(norm_img).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        gt_depth = gt_depth_tensor.squeeze().cpu().numpy()
        gt_label = gt_label_tensor.squeeze().cpu().numpy()

        # Metrics for this sample
        metrics = compute_sample_metrics(pred_depth, gt_depth, pred_seg, gt_label)
        
        filename = f"val_{idx+1:03d}_sample_{idx:04d}.png"
        filepath = images_dir / filename

        # 4. Render 3-Panel Side-by-Side Figure
        # Width: 16.5 inches, Height: 5.2 inches
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=args.dpi, facecolor="#F8FAFC")
        
        # Super title with sample info & metrics
        fig.suptitle(
            f"NYUv2 Val #{idx+1:03d} (Index {idx})  |  "
            f"Depth RMSE: {metrics['rmse']:.3f}m,  δ1: {metrics['delta1']:.3f}  |  "
            f"Seg mIoU: {metrics['miou']:.3f},  Pixel Acc: {metrics['pixel_acc']*100:.1f}%",
            fontsize=12, fontweight="bold", y=0.97, color="#0F172A", fontfamily="DejaVu Sans"
        )

        # Panel 1: Original RGB Image
        axes[0].imshow(rgb_img)
        axes[0].set_title("Original RGB Input", fontsize=11, fontweight="bold", color="#1E293B", pad=8)
        axes[0].axis("off")

        # Panel 2: Predicted Metric Depth Map
        # Common indoor range: 0.5m to 5.0m for clear visualization
        vmax = max(4.0, min(8.0, np.percentile(pred_depth, 98)))
        im_depth = axes[1].imshow(pred_depth, cmap="plasma", vmin=0.5, vmax=vmax)
        axes[1].set_title(f"Predicted Metric Depth (0.5 – {vmax:.1f}m)", fontsize=11, fontweight="bold", color="#991B1B", pad=8)
        axes[1].axis("off")
        cbar = fig.colorbar(im_depth, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Depth [meters]", fontsize=9, color="#475569")
        cbar.ax.tick_params(labelsize=8)

        # Panel 3: Predicted Semantic Segmentation Mask
        im_seg = axes[2].imshow(pred_seg, cmap=seg_cmap, vmin=0, vmax=40)
        axes[2].set_title("Predicted Semantic Segmentation (40 Classes)", fontsize=11, fontweight="bold", color="#065F46", pad=8)
        axes[2].axis("off")

        plt.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.05, wspace=0.15)
        plt.savefig(filepath, dpi=args.dpi, bbox_inches="tight", facecolor="#F8FAFC")
        plt.close(fig)

        records.append({
            "idx": idx + 1,
            "sample_idx": idx,
            "filename": filename,
            "rel_path": f"images/{filename}",
            "rmse": metrics["rmse"],
            "delta1": metrics["delta1"],
            "miou": metrics["miou"],
            "pixel_acc": metrics["pixel_acc"],
        })

    # 4. Generate Interactive HTML Gallery
    generate_html_gallery(records, output_dir, num_to_eval)
    
    # 5. Generate Markdown Gallery Summary
    generate_markdown_summary(records, output_dir, num_to_eval)

    # Summary Stats
    avg_rmse = np.mean([r["rmse"] for r in records])
    avg_delta1 = np.mean([r["delta1"] for r in records])
    avg_miou = np.mean([r["miou"] for r in records])
    avg_acc = np.mean([r["pixel_acc"] for r in records])

    print("\n" + "=" * 70)
    print("                     INFERENCE GENERATION COMPLETE")
    print("=" * 70)
    print(f"  • Generated Images:  {num_to_eval} saved to '{output_dir}/images/'")
    print(f"  • HTML Gallery:      {output_dir}/index.html")
    print(f"  • Markdown Summary:  {output_dir}/GALLERY.md")
    print("  • Average Metrics across 100 Samples:")
    print(f"      - Depth RMSE:       {avg_rmse:.4f} m")
    print(f"      - Depth δ1:         {avg_delta1:.4f}")
    print(f"      - Segmentation mIoU: {avg_miou:.4f}")
    print(f"      - Pixel Accuracy:   {avg_acc*100:.2f} %")
    print("=" * 70 + "\n")


def generate_html_gallery(records, output_dir, num_samples):
    avg_rmse = np.mean([r["rmse"] for r in records])
    avg_delta1 = np.mean([r["delta1"] for r in records])
    avg_miou = np.mean([r["miou"] for r in records])
    avg_acc = np.mean([r["pixel_acc"] for r in records])

    cards_html = ""
    for r in records:
        cards_html += f"""
        <div class="card">
            <div class="card-header">
                <span class="badge badge-id">#{r['idx']:03d} (Sample {r['sample_idx']})</span>
                <span class="metric">RMSE: <b>{r['rmse']:.3f}m</b></span>
                <span class="metric">&delta;<sub>1</sub>: <b>{r['delta1']:.3f}</b></span>
                <span class="metric">mIoU: <b>{r['miou']:.3f}</b></span>
                <span class="metric">Acc: <b>{r['pixel_acc']*100:.1f}%</b></span>
            </div>
            <a href="{r['rel_path']}" target="_blank">
                <img src="{r['rel_path']}" alt="Sample {r['idx']}" loading="lazy"/>
            </a>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TwinForge: 100 Validation Inference Results</title>
    <style>
        :root {{
            --bg: #0F172A;
            --surface: #1E293B;
            --border: #334155;
            --text: #F8FAFC;
            --subtext: #94A3B8;
            --accent: #38BDF8;
            --green: #34D399;
            --red: #F87171;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
        }}
        .header {{
            max-width: 1400px;
            margin: 0 auto 24px auto;
            background: var(--surface);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border);
        }}
        h1 {{
            margin: 0 0 8px 0;
            font-size: 24px;
            color: var(--accent);
        }}
        .subtitle {{
            color: var(--subtext);
            font-size: 14px;
            margin-bottom: 16px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
        }}
        .stat-box {{
            background: #0B1120;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        .stat-label {{
            font-size: 11px;
            text-transform: uppercase;
            color: var(--subtext);
            letter-spacing: 0.5px;
        }}
        .stat-value {{
            font-size: 20px;
            font-weight: bold;
            color: var(--text);
            margin-top: 4px;
        }}
        .grid {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .card {{
            background: var(--surface);
            border-radius: 10px;
            border: 1px solid var(--border);
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}
        .card-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 10px 16px;
            background: #182234;
            border-bottom: 1px solid var(--border);
            font-size: 13px;
        }}
        .badge-id {{
            background: var(--accent);
            color: #0F172A;
            font-weight: bold;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
        }}
        .metric {{
            color: var(--subtext);
        }}
        .metric b {{
            color: var(--text);
        }}
        .card img {{
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
            transition: opacity 0.15s;
        }}
        .card img:hover {{
            opacity: 0.95;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>TwinForge: Multi-Task Inference Results (100 Validation Samples)</h1>
        <div class="subtitle">Tri-panel inference: [Original RGB] | [Predicted Metric Depth] | [Predicted 40-Class Segmentation Mask]</div>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">Model Checkpoint</div>
                <div class="stat-value" style="font-size: 14px; color: var(--accent);">DINOv2 Early-Freeze (Ep 123)</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Avg Depth RMSE</div>
                <div class="stat-value" style="color: var(--red);">{avg_rmse:.4f} m</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Avg Depth &delta;<sub>1</sub></div>
                <div class="stat-value" style="color: var(--green);">{avg_delta1:.4f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Avg Segmentation mIoU</div>
                <div class="stat-value" style="color: var(--green);">{avg_miou:.4f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Avg Pixel Accuracy</div>
                <div class="stat-value">{avg_acc*100:.2f} %</div>
            </div>
        </div>
    </div>

    <div class="grid">
        {cards_html}
    </div>
</body>
</html>
"""
    with open(output_dir / "index.html", "w") as f:
        f.write(html)


def generate_markdown_summary(records, output_dir, num_samples):
    avg_rmse = np.mean([r["rmse"] for r in records])
    avg_delta1 = np.mean([r["delta1"] for r in records])
    avg_miou = np.mean([r["miou"] for r in records])
    avg_acc = np.mean([r["pixel_acc"] for r in records])

    # Top 5 best depth and top 5 best seg
    best_depth = sorted(records, key=lambda x: x["rmse"])[:5]
    best_seg = sorted(records, key=lambda x: x["miou"], reverse=True)[:5]

    md = f"""# TwinForge: 100 Validation Inference Gallery Summary

* **Model Checkpoint:** `checkpoints/dino-v2-backbone-early-freeze-fixed-projection-layers/best_depth.zip` (Epoch 123)
* **Dataset:** NYU Depth V2 Validation Split (100 samples)
* **Image Dimensions:** $392 \\times 518$ (3-Panel Figure: $16.5 \\times 5.2$ inches, 150 DPI)
* **Interactive HTML Gallery:** [`index.html`](file://{output_dir.resolve()}/index.html)
* **Images Directory:** [`images/`](file://{output_dir.resolve()}/images/)

---

## Aggregate 100-Sample Metrics
| Metric | 100-Sample Subset | Full 654-Val Set (Eigen Protocol) |
| :--- | :---: | :---: |
| **Depth RMSE [m]** (↓) | **{avg_rmse:.4f}m** | 0.3650m |
| **Depth $\\delta_1$** (↑) | **{avg_delta1:.4f}** | 0.9112 |
| **Segmentation mIoU** (↑) | **{avg_miou:.4f}** | 0.5770 |
| **Pixel Accuracy** (↑) | **{avg_acc*100:.2f}%** | 80.19% |

---

## Top 5 Qualitative Depth Predictions (Lowest RMSE)
| Rank | Sample | Filename | RMSE (m) | $\\delta_1$ | mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for rank, r in enumerate(best_depth, 1):
        md += f"| {rank} | #{r['idx']} | [{r['filename']}](file://{output_dir.resolve()}/images/{r['filename']}) | **{r['rmse']:.3f}m** | {r['delta1']:.3f} | {r['miou']:.3f} | {r['pixel_acc']*100:.1f}% |\n"

    md += """
---

## Top 5 Qualitative Segmentation Predictions (Highest mIoU)
| Rank | Sample | Filename | mIoU | Pixel Acc | RMSE (m) | $\\delta_1$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for rank, r in enumerate(best_seg, 1):
        md += f"| {rank} | #{r['idx']} | [{r['filename']}](file://{output_dir.resolve()}/images/{r['filename']}) | **{r['miou']:.3f}** | {r['pixel_acc']*100:.1f}% | {r['rmse']:.3f}m | {r['delta1']:.3f} |\n"

    with open(output_dir / "GALLERY.md", "w") as f:
        f.write(md)


if __name__ == "__main__":
    main()
