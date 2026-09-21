import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import numpy as np

import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import NYUv2Dataset
from models import TwinForge


# ============================================================
# CLI Arguments
# ============================================================

parser = argparse.ArgumentParser(description="TwinForge Single Image Inference")
parser.add_argument("--checkpoint", type=str, default="checkpoints/best_depth.pth", help="Path to model checkpoint")
parser.add_argument("--classes", type=int, choices=[20, 40], default=40, help="Number of segmentation classes to visualize: 40 (NYU40) or 20 (ScanNet20) (default: 40)")
parser.add_argument("--idx", type=int, default=7, help="Sample index in NYUv2 validation split (default: 7)")
parser.add_argument("--output", type=str, default="inference_result.png", help="Output visualization path (default: inference_result.png)")
args, unknown = parser.parse_known_args()

# Backwards compatibility: allow positional checkpoint path
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    checkpoint_path = sys.argv[1]
else:
    checkpoint_path = args.checkpoint


# ============================================================
# Class Mapping & Colormap Configuration
# ============================================================

# Canonical ScanNet 20 benchmark subset mapping from NYU40 IDs
NYU40_TO_SCANNET20 = torch.tensor([
    0,   # 0: unlabeled
    1,   # 1: wall
    2,   # 2: floor
    3,   # 3: cabinet
    4,   # 4: bed
    5,   # 5: chair
    6,   # 6: sofa
    7,   # 7: table
    8,   # 8: door
    9,   # 9: window
    10,  # 10: bookshelf
    11,  # 11: picture
    12,  # 12: counter
    0,   # 13: blinds -> ignore
    13,  # 14: desk
    0,   # 15: shelves -> ignore
    14,  # 16: curtain
    0,   # 17: dresser -> ignore
    0,   # 18: pillow -> ignore
    0,   # 19: mirror -> ignore
    0,   # 20: floor mat -> ignore
    0,   # 21: clothes -> ignore
    0,   # 22: ceiling -> ignore
    0,   # 23: books -> ignore
    15,  # 24: refrigerator
    0,   # 25: television -> ignore
    0,   # 26: paper -> ignore
    0,   # 27: towel -> ignore
    16,  # 28: shower curtain
    0,   # 29: box -> ignore
    0,   # 30: whiteboard -> ignore
    0,   # 31: person -> ignore
    0,   # 32: nightstand -> ignore
    17,  # 33: toilet
    18,  # 34: sink
    0,   # 35: lamp -> ignore
    19,  # 36: bathtub
    0,   # 37: bag -> ignore
    0,   # 38: otherstructure -> ignore
    20,  # 39: otherfurniture
    0    # 40: otherprop -> ignore
], dtype=torch.int64)

if args.classes == 20:
    # 21 distinct colors: class 0 (unlabeled) as black, 1-20 from tab20
    colors = [(0.0, 0.0, 0.0)] + list(plt.colormaps["tab20"].colors)
    cmap = mcolors.ListedColormap(colors[:21])
    norm = mcolors.Normalize(vmin=0, vmax=20)
else:
    colors1 = plt.colormaps["tab20"].colors
    colors2 = plt.colormaps["tab20b"].colors
    colors = [(0.0, 0.0, 0.0)] + list(colors1) + list(colors2)[:20]
    cmap = mcolors.ListedColormap(colors[:41])
    norm = mcolors.Normalize(vmin=0, vmax=40)


# ============================================================
# Load dataset
# ============================================================

dataset = NYUv2Dataset(
    "data/nyu_depth_v2_labeled.mat",
    "data/classMapping40.mat",
    split="val"
)

image, depth, label = dataset[args.idx]

print("Image shape:", image.shape)
print("Depth shape:", depth.shape)
print("Label shape:", label.shape)
print(f"Classes present (NYU40): {label.unique().tolist()}")


# ============================================================
# Load model
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TwinForge(num_labels=41, num_heads=8, tok_dim=256, freeze=False).to(device)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {checkpoint_path}")
else:
    print(f"Checkpoint not found at '{checkpoint_path}'. Running with initialized weights.")

model.eval()


# ============================================================
# Run inference
# ============================================================

image_input = image.unsqueeze(0).to(device)

with torch.no_grad():
    pred_seg, pred_depth = model(image_input)


# ============================================================
# Process predictions
# ============================================================

# Segmentation: [1, 41, H, W] -> [H, W]
pred_label = torch.argmax(pred_seg, dim=1).squeeze(0).cpu()

if args.classes == 20:
    pred_label = NYU40_TO_SCANNET20[pred_label]
    label = NYU40_TO_SCANNET20[label]

# Depth: [1, 1, H, W] -> [H, W]
pred_depth = pred_depth.squeeze().cpu()


# ============================================================
# Denormalize RGB
# ============================================================

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


# ============================================================
# Convert tensors for visualization
# ============================================================

# RGB [3, H, W] -> [H, W, 3]
rgb_img = denormalize(image).permute(1, 2, 0).cpu().numpy()

# Ground-truth depth
depth_map = depth.squeeze().cpu().numpy()

# Predicted depth
pred_depth_map = pred_depth.numpy()

# Ground-truth segmentation
label_map = label.squeeze().cpu().numpy()

# Predicted segmentation
pred_label_map = pred_label.numpy()


# ============================================================
# Plot (2 rows, 3 columns)
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. RGB
axes[0, 0].imshow(rgb_img)
axes[0, 0].set_title("RGB Image")
axes[0, 0].axis("off")

# 2. GT Depth
axes[0, 1].imshow(depth_map, cmap="inferno")
axes[0, 1].set_title("GT Depth")
axes[0, 1].axis("off")

# 3. Predicted Depth
axes[0, 2].imshow(pred_depth_map, cmap="inferno")
axes[0, 2].set_title("Predicted Depth")
axes[0, 2].axis("off")

# 4. GT Segmentation
seg_suffix = " (ScanNet 20)" if args.classes == 20 else " (NYU 40)"
axes[1, 0].imshow(label_map, cmap=cmap, norm=norm, interpolation="nearest")
axes[1, 0].set_title("GT Segmentation" + seg_suffix)
axes[1, 0].axis("off")

# 5. Predicted Segmentation
axes[1, 1].imshow(pred_label_map, cmap=cmap, norm=norm, interpolation="nearest")
axes[1, 1].set_title("Predicted Segmentation" + seg_suffix)
axes[1, 1].axis("off")

# 6. Blank / Info
axes[1, 2].axis("off")

plt.tight_layout()
output_path = args.output
plt.savefig(output_path, dpi=150)
print(f"Inference visualization saved to {output_path}")
if os.environ.get("DISPLAY"):
    plt.show()