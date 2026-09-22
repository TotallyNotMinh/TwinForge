import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import NYUv2Dataset
from data.dataset import NYU40_TO_SCANNET20
from models import TwinForge
from metrics import MultiTaskMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TwinForge checkpoints using the Eigen protocol on NYUv2")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/vit-orientation-fixed/best_depth.zip", help="Path to checkpoint (.pth or .zip)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"], help="Dataset split to evaluate")
    parser.add_argument("--classes", type=int, choices=[20, 40], default=40, help="Number of segmentation classes to evaluate: 40 (NYU40) or 20 (ScanNet20) (default: 40)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--raw-depth", action="store_true", default=True, help="Evaluate against raw 480x640 depth without interpolation (Eigen protocol)")
    parser.add_argument("--no-raw-depth", action="store_false", dest="raw_depth", help="Evaluate against downsampled depth")
    parser.add_argument("--input-h", type=int, default=378, help="Model input height")
    parser.add_argument("--input-w", type=int, default=504, help="Model input width")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    return parser.parse_args()


def load_model_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Adapt positional embeddings if resolution/patch sizes changed across runs
    model_state = model.state_dict()
    for key in ["decoder.pos_embed1", "decoder.pos_embed2", "decoder.pos_embed3", "decoder.pos_embed4", "decoder.pos_embed5"]:
        if key in state_dict and key in model_state:
            if state_dict[key].shape != model_state[key].shape:
                target_hw = model_state[key].shape[-2:]
                state_dict[key] = F.interpolate(
                    state_dict[key],
                    size=target_hw,
                    mode="bicubic",
                    align_corners=False
                )

    model.load_state_dict(state_dict)
    epoch = checkpoint.get("epoch", "N/A")
    print(f"Loaded checkpoint '{checkpoint_path}' (Epoch {epoch})")
    return checkpoint


def evaluate():
    args = parse_args()
    device = torch.device(args.device)
    resize = (args.input_h, args.input_w)

    print("=" * 70)
    print("       NYU Depth V2 - Standard Eigen Protocol Evaluation")
    print("=" * 70)
    print(f"  • Checkpoint:     {args.checkpoint_path}")
    print(f"  • Split:          {args.split} (654 images)")
    print(f"  • Classes:        {args.classes} ({'ScanNet 20 benchmark' if args.classes == 20 else 'NYU 40'})")
    print(f"  • Input Size:     {resize}")
    print(f"  • Raw 480x640 GT: {args.raw_depth} (Eigen Protocol compliant)")
    print(f"  • Device:         {device}")
    print("=" * 70)

    # 1. Dataset & DataLoader
    dataset = NYUv2Dataset(
        data_path="data/nyu_depth_v2_labeled.mat",
        class_map_path="data/classMapping40.mat",
        split=args.split,
        resize=resize,
        augment=False,
        return_raw_depth=args.raw_depth
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )

    # 2. Model (detect num_labels from checkpoint if available)
    model_num_labels = 41
    if os.path.exists(args.checkpoint_path):
        ckpt_head = torch.load(args.checkpoint_path, map_location="cpu")
        sd = ckpt_head.get("model_state_dict", ckpt_head)
        if "decoder.segment_dec.out.3.weight" in sd:
            model_num_labels = sd["decoder.segment_dec.out.3.weight"].shape[0]

    model = TwinForge(
        num_labels=model_num_labels,
        num_heads=8,
        tok_dim=256,
        size=resize,
        freeze=False
    ).to(device)

    load_model_checkpoint(model, args.checkpoint_path, device)
    model.eval()

    # 3. Metrics (21 classes for ScanNet20 benchmark, 41 for NYU40)
    eval_num_classes = 21 if args.classes == 20 else 41
    metrics = MultiTaskMetrics(num_classes=eval_num_classes)
    metrics.reset()

    mapping = NYU40_TO_SCANNET20.to(device) if args.classes == 20 else None

    # 4. Evaluation Loop
    desc_str = f"Evaluating (Eigen Protocol - {args.classes} classes)"
    with torch.no_grad():
        for images, depths, labels in tqdm(data_loader, desc=desc_str):
            images = images.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred_seg, pred_depth = model(images)

            if args.classes == 20:
                pred_label = torch.argmax(pred_seg, dim=1)
                pred_seg_eval = mapping[pred_label] if model_num_labels == 41 else pred_label
                labels_eval = mapping[labels]
            else:
                pred_seg_eval = pred_seg
                labels_eval = labels

            # Pred depth is automatically interpolated by DepthMetrics if target has different shape (480, 640)
            metrics.update(pred_seg_eval, pred_depth, labels_eval, depths)

    results = metrics.compute()
    depth_res = results["depth"]
    seg_res = results["segmentation"]

    # 5. Format and Print Results
    print("\n" + "=" * 70)
    print("                EIGEN PROTOCOL DEPTH METRICS")
    print("=" * 70)
    print(f"  Threshold δ < 1.25  (↑):  {depth_res['delta1']:.4f}")
    print(f"  Threshold δ < 1.25² (↑):  {depth_res['delta2']:.4f}")
    print(f"  Threshold δ < 1.25³ (↑):  {depth_res['delta3']:.4f}")
    print(f"  Absolute Relative   (↓):  {depth_res['abs_rel']:.4f}")
    print(f"  Squared Relative    (↓):  {depth_res['sq_rel']:.4f}")
    print(f"  RMSE [m]            (↓):  {depth_res['rmse']:.4f}")
    print(f"  RMSE (log)          (↓):  {depth_res['rmse_log']:.4f}")
    print(f"  log10               (↓):  {depth_res['log10']:.4f}")
    print("=" * 70)

    seg_header = "SEMANTIC SEGMENTATION METRICS (ScanNet 20)" if args.classes == 20 else "SEMANTIC SEGMENTATION METRICS (NYU 40)"
    print(f"               {seg_header}")
    print("=" * 70)
    print(f"  Mean IoU            (↑):  {seg_res['miou']:.4f}")
    print(f"  Dice Coefficient    (↑):  {seg_res['dice']:.4f}")
    print(f"  Pixel Accuracy      (↑):  {seg_res['pixel_acc']:.4f}")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    evaluate()
