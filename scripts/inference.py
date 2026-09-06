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
# Configuration
# ============================================================

num_classes = 41

# 40 distinct categorical colors
colors1 = plt.colormaps["tab20"].colors
colors2 = plt.colormaps["tab20b"].colors

colors = list(colors1) + list(colors2)

cmap = mcolors.ListedColormap(colors[:num_classes])

norm = mcolors.Normalize(
    vmin=0,
    vmax=num_classes - 1
)

# ============================================================
# Load dataset
# ============================================================

dataset = NYUv2Dataset(
    "data/nyu_depth_v2_labeled.mat",
    "data/classMapping40.mat",
    split="val"
)

image, depth, label = dataset[7]

print("Image shape:", image.shape)
print("Depth shape:", depth.shape)
print("Label shape:", label.shape)
print("Classes present:", label.unique())


# ============================================================
# Load model
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TwinForge(num_labels=num_classes, num_heads=8, tok_dim=256, freeze=False).to(device)

import os
checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_depth.pth"

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
axes[1, 0].imshow(label_map, cmap=cmap, norm=norm, interpolation="nearest")
axes[1, 0].set_title("GT Segmentation")
axes[1, 0].axis("off")

# 5. Predicted Segmentation
axes[1, 1].imshow(pred_label_map, cmap=cmap, norm=norm, interpolation="nearest")
axes[1, 1].set_title("Predicted Segmentation")
axes[1, 1].axis("off")

# 6. Blank / Info
axes[1, 2].axis("off")

plt.tight_layout()
output_path = "inference_result.png"
plt.savefig(output_path, dpi=150)
print(f"Inference visualization saved to {output_path}")
if os.environ.get("DISPLAY"):
    plt.show()