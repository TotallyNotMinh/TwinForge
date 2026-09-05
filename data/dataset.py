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

    def __init__(self, data_path, class_map_path, split, splits_path="data/splits.mat", augment=True, resize=(640, 480)):
        self.data_path = data_path
        self.class_map_path = class_map_path
        self.splits_path = splits_path
        self.data = None

        mat = scipy.io.loadmat(self.class_map_path)
        map_class = mat["mapClass"].squeeze()
        lookup = np.zeros(895, dtype=np.int64)
        lookup[1:895] = map_class
        self.class_map = torch.from_numpy(lookup)

        self.resize = resize
        self.rgb_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

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

        image = torch.from_numpy(
            self.images[idx]
        ).float() / 255.0

        depth = torch.from_numpy(
            self.depths[idx]
        ).float()

        raw_label = torch.from_numpy(
            self.labels[idx]
        ).long()

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

        boundary = get_boundary_map(label).float()

        image = self.rgb_transform(image)

        return image, depth, label, boundary
    
if __name__ == "__main__":
    dataset_path = "data/nyu_depth_v2_labeled.mat"
    dataset = NYUv2Dataset(dataset_path, split="train")

    dataset = NYUv2Dataset(dataset_path, split="train")

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