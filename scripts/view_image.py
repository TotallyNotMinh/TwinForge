import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import numpy as np

import sys
from pathlib import Path

# Add project root (one level up from scripts/) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.dataset import NYUv2Dataset

# Number of NYUv2 classes
num_classes = 40

# Get 40 distinct categorical colors
colors1 = plt.colormaps["tab20"].colors
colors2 = plt.colormaps["tab20b"].colors

colors = list(colors1) + list(colors2)

cmap = mcolors.ListedColormap(colors[:num_classes])

# Make sure:
# class 0 -> color 0
# class 1 -> color 1
# ...
# class 39 -> color 39
norm = mcolors.BoundaryNorm(
    np.arange(-0.5, num_classes + 0.5, 1),
    cmap.N
)


def denormalize(tensor):
    # Invert ImageNet normalization:
    # x_original = x_normalized * std + mean

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=tensor.device
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=tensor.device
    ).view(3, 1, 1)

    return torch.clamp(
        tensor * std + mean,
        0.0,
        1.0
    )


# Load dataset
dataset = NYUv2Dataset(
    "data/nyu_depth_v2_labeled.mat",
    "data/classMapping40.mat",
    split="train"
)

# Get sample
image, depth, label, boundary = dataset[126]

print("Image shape:", image.shape)
print("Depth shape:", depth.shape)
print("Label shape:", label.shape)
print("Boundary shape:", boundary.shape)
print("Classes present:", label.unique())


# --------------------------------
# Convert tensors for visualization
# --------------------------------

# [3, H, W] -> [H, W, 3]
rgb_img = denormalize(image).permute(1, 2, 0).numpy()

# [1, H, W] -> [H, W]
depth_map = depth.squeeze().numpy()

# [H, W]
label_map = label.numpy()

boundary_map = boundary.squeeze().numpy()

# --------------------------------
# Plot
# --------------------------------

fig, axes = plt.subplots(
    1, 4,
    figsize=(15, 5)
)


# RGB
axes[0].imshow(rgb_img)
axes[0].set_title("RGB Image")
axes[0].axis("off")


# Depth
axes[1].imshow(
    depth_map,
    cmap="inferno"
)
axes[1].set_title("Depth Map")
axes[1].axis("off")


# Segmentation
im = axes[2].imshow(
    label_map,
    cmap=cmap,
    norm=norm,
    interpolation="nearest"
)

axes[2].set_title("Segmentation Label")
axes[2].axis("off")

# Boundary
im = axes[3].imshow(
    boundary_map,
    cmap=cmap,
    norm=norm,
    interpolation="nearest"
)

axes[3].set_title("Boundary Map")
axes[3].axis("off")



plt.tight_layout()
plt.show()