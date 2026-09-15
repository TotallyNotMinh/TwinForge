import os
import sys
from pathlib import Path
import argparse
import time

# Cap CPU threads before importing math/tensor libraries
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

# Set PyTorch thread limit
torch.set_num_threads(4)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import TwinForge


def get_nyuv2_palette(num_classes: int = 41) -> np.ndarray:
    """Build a distinct (num_classes, 3) BGR color lookup table for NYUv2 classes."""
    colors1 = plt.colormaps["tab20"].colors
    colors2 = plt.colormaps["tab20b"].colors
    # Background (class 0) as black, remaining 40 classes from palettes
    color_list = [(0.0, 0.0, 0.0)] + list(colors1) + list(colors2)[:20]
    
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(min(num_classes, len(color_list))):
        r, g, b = color_list[i]
        palette[i] = [int(b * 255), int(g * 255), int(r * 255)]  # BGR order for OpenCV
    return palette


def add_banner(img: np.ndarray, text: str, banner_height: int = 36) -> np.ndarray:
    """Draw a clean semi-transparent header bar with title text."""
    h, w = img.shape[:2]
    out = img.copy()
    
    # Dark banner overlay at the top
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
    
    # White label text
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.65
    thickness = 1
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = (w - text_size[0]) // 2
    ty = (banner_height + text_size[1]) // 2 - 2
    cv2.putText(out, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def load_model(checkpoint_path: str, device: str, model_size=(392, 518)) -> TwinForge:
    """Load TwinForge model with weights from checkpoint."""
    model = TwinForge(
        num_labels=41,
        num_heads=8,
        tok_dim=256,
        size=model_size,
        pretrained=False,
        freeze=True
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model successfully loaded and set to eval mode on {device}.")
    return model


COLORMAP_MAP = {
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
}


def run_video_inference(
    video_path: str,
    checkpoint_path: str,
    output_path: str = None,
    layout: str = "side-by-side",
    panel_width: int = 640,
    panel_height: int = 480,
    batch_size: int = 8,
    fps: float = None,
    stride: int = 1,
    max_frames: int = None,
    depth_colormap: str = "inferno",
    alpha: float = 0.45,
    add_labels: bool = True,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Determine output path if not specified
    if output_path is None:
        in_p = Path(video_path)
        output_path = str(in_p.parent / f"{in_p.stem}_pred_{layout}.mp4")

    os.makedirs(Path(output_path).parent, exist_ok=True)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = (src_fps / stride) if fps is None else float(fps)

    frames_to_process = total_frames if max_frames is None else min(total_frames, max_frames)
    effective_frames = (frames_to_process + stride - 1) // stride

    print("=" * 60)
    print("TwinForge Video Inference")
    print(f"Input Video:      {video_path}")
    print(f"Resolution:       {src_w}x{src_h} @ {src_fps:.2f} fps ({total_frames} total frames)")
    print(f"Output Video:     {output_path} @ {out_fps:.2f} fps")
    print(f"Processing:       {effective_frames} frames (stride={stride}, max_frames={max_frames})")
    print(f"Layout:           {layout} (panel: {panel_width}x{panel_height})")
    print(f"Batch Size:       {batch_size} | Device: {device}")
    print("=" * 60)

    # Model input resolution for ViT patch divisibility (must be multiple of 14)
    model_h, model_w = 392, 518
    model = load_model(checkpoint_path, device, model_size=(model_h, model_w))

    # Color palette and colormap
    palette = get_nyuv2_palette(num_classes=41)
    cv_cmap = COLORMAP_MAP.get(depth_colormap.lower(), cv2.COLORMAP_INFERNO)

    # Output dimensions based on layout
    pw, ph = panel_width, panel_height
    if layout == "side-by-side":
        out_w, out_h = pw * 3, ph
    elif layout == "grid":
        out_w, out_h = pw * 2, ph * 2
    else:  # 'depth', 'seg', 'overlay'
        out_w, out_h = pw, ph

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    # Normalization transform
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Running depth range bounds for flicker-free temporal smoothing
    smooth_dmin = None
    smooth_dmax = None
    smooth_alpha = 0.15

    frame_idx = 0
    pbar = tqdm(total=effective_frames, desc="Processing Video", unit="frame")
    start_time = time.time()

    try:
        while frame_idx < frames_to_process:
            batch_raw_frames = []
            batch_model_inputs = []

            # Collect a batch of frames respecting stride
            while len(batch_raw_frames) < batch_size and frame_idx < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % stride == 0:
                    batch_raw_frames.append(frame)
                    # Prepare model input: BGR -> RGB -> resize -> float tensor [0, 1]
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    in_img = cv2.resize(rgb_frame, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
                    t = torch.from_numpy(in_img).permute(2, 0, 1).float() / 255.0
                    batch_model_inputs.append(t)

                frame_idx += 1

            if not batch_raw_frames:
                break

            # Batch tensor: (B, 3, H, W)
            x = torch.stack(batch_model_inputs).to(device, non_blocking=True)
            x = (x - mean) / std

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu"):
                    pred_seg, pred_depth = model(x)

            # Process batch outputs
            pred_labels = torch.argmax(pred_seg, dim=1).cpu().numpy().astype(np.uint8)  # (B, H, W)
            pred_depths = pred_depth.squeeze(1).cpu().numpy()  # (B, H, W)

            for b in range(len(batch_raw_frames)):
                orig_frame = batch_raw_frames[b]
                rgb_panel = cv2.resize(orig_frame, (pw, ph), interpolation=cv2.INTER_LINEAR)

                # Depth processing with smoothed normalization
                cur_d = pred_depths[b]
                p_min = float(np.percentile(cur_d, 2))
                p_max = float(np.percentile(cur_d, 98))

                if smooth_dmin is None:
                    smooth_dmin = p_min
                    smooth_dmax = p_max
                else:
                    smooth_dmin = (1 - smooth_alpha) * smooth_dmin + smooth_alpha * p_min
                    smooth_dmax = (1 - smooth_alpha) * smooth_dmax + smooth_alpha * p_max

                d_norm = np.clip((cur_d - smooth_dmin) / (smooth_dmax - smooth_dmin + 1e-8), 0.0, 1.0)
                # Invert: closer is brighter/warmer
                depth_u8 = (255.0 * (1.0 - d_norm)).astype(np.uint8)
                depth_u8 = cv2.resize(depth_u8, (pw, ph), interpolation=cv2.INTER_LINEAR)
                depth_colored = cv2.applyColorMap(depth_u8, cv_cmap)

                # Segmentation processing
                seg_label = pred_labels[b]
                seg_label_resized = cv2.resize(seg_label, (pw, ph), interpolation=cv2.INTER_NEAREST)
                seg_colored = palette[seg_label_resized]

                # Compose layout
                if layout == "side-by-side":
                    p1 = add_banner(rgb_panel, "Input Video") if add_labels else rgb_panel
                    p2 = add_banner(depth_colored, "Predicted Depth") if add_labels else depth_colored
                    p3 = add_banner(seg_colored, "Semantic Segmentation") if add_labels else seg_colored
                    out_frame = np.hstack([p1, p2, p3])

                elif layout == "grid":
                    overlay = cv2.addWeighted(seg_colored, alpha, rgb_panel, 1.0 - alpha, 0)
                    p1 = add_banner(rgb_panel, "Input Video") if add_labels else rgb_panel
                    p2 = add_banner(depth_colored, "Predicted Depth") if add_labels else depth_colored
                    p3 = add_banner(seg_colored, "Semantic Segmentation") if add_labels else seg_colored
                    p4 = add_banner(overlay, "RGB + Seg Overlay") if add_labels else overlay
                    top_row = np.hstack([p1, p2])
                    bot_row = np.hstack([p3, p4])
                    out_frame = np.vstack([top_row, bot_row])

                elif layout == "overlay":
                    overlay = cv2.addWeighted(seg_colored, alpha, rgb_panel, 1.0 - alpha, 0)
                    out_frame = add_banner(overlay, "RGB + Seg Overlay") if add_labels else overlay

                elif layout == "depth":
                    out_frame = add_banner(depth_colored, "Predicted Depth") if add_labels else depth_colored

                elif layout == "seg":
                    out_frame = add_banner(seg_colored, "Semantic Segmentation") if add_labels else seg_colored

                writer.write(out_frame)
                pbar.update(1)

    finally:
        cap.release()
        writer.release()
        pbar.close()

    total_time = time.time() - start_time
    proc_fps = effective_frames / max(total_time, 1e-6)
    out_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0

    print("-" * 60)
    print(f"Inference Completed!")
    print(f"Output saved to:  {output_path} ({out_size_mb:.2f} MB)")
    print(f"Processed frames: {effective_frames} in {total_time:.2f}s ({proc_fps:.2f} FPS)")
    print("-" * 60)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="TwinForge Video Inference Pipeline")
    parser.add_argument(
        "--video-path",
        type=str,
        default="data/40753679/40753679.mov",
        help="Path to input video (.mov, .mp4, etc.)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/dino-v2-backbone-early-freeze-fixed-projection-layers/best_depth.zip",
        help="Path to trained TwinForge model checkpoint",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path for output video file (defaults to <video_name>_pred_<layout>.mp4)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["side-by-side", "grid", "overlay", "depth", "seg"],
        default="side-by-side",
        help="Output composition layout (default: side-by-side)",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=640,
        help="Width of each sub-panel (default: 640)",
    )
    parser.add_argument(
        "--panel-height",
        type=int,
        default=480,
        help="Height of each sub-panel (default: 480)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size (default: 8)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video FPS (default: matches input video)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (default: all)",
    )
    parser.add_argument(
        "--depth-colormap",
        type=str,
        choices=["inferno", "magma", "turbo", "viridis", "plasma"],
        default="inferno",
        help="Colormap for depth visualization (default: inferno)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Segmentation transparency for overlay layout (default: 0.45)",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Disable title labels on panels",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (default: cuda if available else cpu)",
    )

    args = parser.parse_args()

    run_video_inference(
        video_path=args.video_path,
        checkpoint_path=args.checkpoint,
        output_path=args.output_path,
        layout=args.layout,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        batch_size=args.batch_size,
        fps=args.fps,
        stride=args.stride,
        max_frames=args.max_frames,
        depth_colormap=args.depth_colormap,
        alpha=args.alpha,
        add_labels=not args.no_labels,
        device=args.device,
    )


if __name__ == "__main__":
    main()
