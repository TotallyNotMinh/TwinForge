import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import scipy.io
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from .augment import NYUv2Augmentation
import os
import glob
import random
from PIL import Image

def get_boundary_map(label: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if label.dim() == 2:
        label = label.unsqueeze(0).unsqueeze(0)
    elif label.dim() == 3:
        label = label.unsqueeze(1)

    label_float = label.float()
    padding = kernel_size // 2

    # Find max and min label in each local patch
    max_label = F.max_pool2d(label_float, kernel_size=kernel_size, stride=1, padding=padding)
    min_label = -F.max_pool2d(-label_float, kernel_size=kernel_size, stride=1, padding=padding)

    # Any patch where max != min contains a boundary
    boundary = (max_label != min_label).float()
    return boundary


class NYUv2Dataset(Dataset):

    def __init__(self, data_path, class_map_path, split, splits_path="data/splits.mat", augment=True, resize=(480, 640), return_raw_depth=False):
        self.data_path = data_path
        self.class_map_path = class_map_path
        self.splits_path = splits_path
        self.return_raw_depth = return_raw_depth
        self.data = None

        mat = scipy.io.loadmat(self.class_map_path)
        map_class = mat["mapClass"].squeeze()
        lookup = np.zeros(895, dtype=np.int64)
        lookup[1:895] = map_class
        self.class_map = torch.from_numpy(lookup)

        self.resize = resize
        self.rgb_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.split = split
        # Disable augment on val/test
        if self.split != "train":
            augment = False
        self.augment = augment
        self.augmentation = NYUv2Augmentation()

        with h5py.File(self.data_path, "r") as f:
            self.images = np.array(f["images"])
            self.depths = np.array(f["depths"])
            self.labels = np.array(f["labels"])

        splits_data = scipy.io.loadmat(self.splits_path)
        train_key = "trainNdxs" if "trainNdxs" in splits_data else "trainNdx"
        test_key = "testNdxs" if "testNdxs" in splits_data else "testNdx"

        self.train_indices = torch.from_numpy(splits_data[train_key].squeeze() - 1).long()
        self.val_indices = torch.from_numpy(splits_data[test_key].squeeze() - 1).long()

        if self.split == "train":
            self.indices = self.train_indices
        elif self.split in ["val", "test"]:
            self.indices = self.val_indices
        else:
            raise ValueError(f"Unknown split: {self.split}. Expected 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]

        # Invert HDF5 (W, H) storage order into standard PyTorch (C, H, W) format
        image = torch.from_numpy(
            self.images[idx]
        ).permute(0, 2, 1).float() / 255.0

        depth = torch.from_numpy(
            self.depths[idx]
        ).transpose(1, 0).float()

        raw_label = torch.from_numpy(
            self.labels[idx]
        ).transpose(1, 0).long()

        if self.augment:
            image, depth, raw_label = self.augmentation(
                image,
                depth,
                raw_label
            )

        image = TF.resize(
            image,
            self.resize,
            interpolation=InterpolationMode.BILINEAR
        )

        if not self.return_raw_depth:
            depth = depth.unsqueeze(0).unsqueeze(0)

            depth = F.interpolate(
                depth,
                size=self.resize,
                mode="bilinear",
                align_corners=False
            ).squeeze(0).squeeze(0)

        raw_label = raw_label.unsqueeze(0).unsqueeze(0)

        raw_label = F.interpolate(
            raw_label.float(),
            size=self.resize,
            mode="nearest"
        ).squeeze(0).squeeze(0).long()

        label = self.class_map[raw_label]

        image = self.rgb_transform(image)

        return image, depth, label
    


class ScanNetVideoDataset(Dataset):
    """
    Video sequence dataset for ScanNet25k.
    Samples consecutive video clips of length T (num_frames) with frame stride.
    Returns:
        images: (T, 3, H, W) float32 normalized RGB
        depths: (T, 1, H, W) float32 metric depth in meters
        labels: (T, H, W)    int64 NYU40 class IDs (0-40)
    """
    def __init__(
        self,
        root_dir: str = "data/scannet_frames_25k",
        split: str = "train",
        num_frames: int = 4,
        stride: int = 1,
        resize: tuple = (384, 512),
        augment: bool = True,
        train_ratio: float = 0.85
    ):
        super().__init__()
        self.num_frames = num_frames
        self.stride = stride
        self.resize = resize
        self.augment = augment if split == "train" else False

        self.rgb_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Discover scenes and partition train / val
        all_scenes = sorted(glob.glob(os.path.join(root_dir, "scene*")))
        if len(all_scenes) == 0:
            raise RuntimeError(f"No scene directories found in {root_dir}")

        split_idx = int(train_ratio * len(all_scenes))
        self.scenes = all_scenes[:split_idx] if split == "train" else all_scenes[split_idx:]

        # Index all valid consecutive clips
        self.clips = []
        for s_dir in self.scenes:
            color_dir = os.path.join(s_dir, "color")
            if not os.path.isdir(color_dir):
                continue
            frame_stems = sorted([
                os.path.splitext(f)[0] for f in os.listdir(color_dir) if f.endswith(".jpg")
            ])
            max_start = len(frame_stems) - (num_frames - 1) * stride
            clip_step = num_frames * stride
            for i in range(0, max_start, clip_step):
                clip = [frame_stems[i + j * stride] for j in range(num_frames)]
                self.clips.append((s_dir, clip))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        s_dir, stems = self.clips[idx]
        
        # Synchronized horizontal flip across all frames in this clip
        do_flip = self.augment and (random.random() < 0.5)

        imgs, depths, labels = [], [], []
        for stem in stems:
            # 1. Load Color Frame
            col_path = os.path.join(s_dir, "color", f"{stem}.jpg")
            col = Image.open(col_path).convert("RGB")
            col = col.resize((self.resize[1], self.resize[0]), Image.BILINEAR)
            if do_flip:
                col = TF.hflip(col)
            imgs.append(self.rgb_norm(TF.to_tensor(col)))

            # 2. Load Depth Map (16-bit uint mm -> float32 meters)
            dep_path = os.path.join(s_dir, "depth", f"{stem}.png")
            dep = Image.open(dep_path)
            dep = dep.resize((self.resize[1], self.resize[0]), Image.NEAREST)
            if do_flip:
                dep = TF.hflip(dep)
            dep_m = torch.from_numpy(np.array(dep, dtype=np.float32) / 1000.0).unsqueeze(0)
            depths.append(dep_m)

            # 3. Load Semantic Mask (NYU40 uint8 -> int64)
            lbl_path = os.path.join(s_dir, "label", f"{stem}.png")
            lbl = Image.open(lbl_path)
            lbl = lbl.resize((self.resize[1], self.resize[0]), Image.NEAREST)
            if do_flip:
                lbl = TF.hflip(lbl)
            labels.append(torch.from_numpy(np.array(lbl, dtype=np.int64)))

        return (
            torch.stack(imgs, dim=0),     # Shape: (T, 3, H, W)
            torch.stack(depths, dim=0),   # Shape: (T, 1, H, W)
            torch.stack(labels, dim=0)    # Shape: (T, H, W)
        )


if __name__ == "__main__":
    dataset_path = "data/nyu_depth_v2_labeled.mat"
    class_map_path = "data/classMapping40.mat"
    dataset = NYUv2Dataset(dataset_path, class_map_path, split="train")

    print("images:", dataset.images.shape)
    print("depths:", dataset.depths.shape)
    print("labels:", dataset.labels.shape)

    print("image[0]:", dataset.images[0].shape)
    print("depth[0]:", dataset.depths[0].shape)
    print("label[0]:", dataset.labels[0].shape)


    image, depth, label = dataset[0]

    print("image:", image.shape)
    print("depth:", depth.shape)
    print("label:", label.shape)