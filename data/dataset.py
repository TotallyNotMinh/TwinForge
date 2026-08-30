import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import scipy.io
import numpy as np

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

    def __init__(self, data_path, class_map_path, split, augment=True, resize=(640, 320)):
        self.data_path = data_path
        self.class_map_path = class_map_path
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

        data = h5py.File(self.data_path, "r")
        self.images = data["images"]
        self.train_size = int(0.8 * len(self.images))
        self.val_size = len(self.images) - self.train_size

        generator = torch.Generator().manual_seed(42)

        indices = torch.randperm(len(self.images), generator=generator)

        self.train_indices = indices[:self.train_size]
        self.val_indices = indices[self.train_size:]
        self.indices = self.train_indices if self.split == "train" else self.val_indices

    def __len__(self):
        return self.train_size if self.split == "train" else self.val_size

    def __getitem__(self, idx):
        if self.data == None: data = h5py.File(self.data_path, "r") 
        idx = self.indices[idx]

        image = torch.from_numpy(data["images"][idx])
        depth = torch.from_numpy(data["depths"][idx])
        raw_label = torch.from_numpy(data["labels"][idx])

        image = image.float() / 255.0
        image = self.rgb_transform(image)

        # F.interpolate() expects image-like tensors in (N, C, H, W) format:
        # Add dimensions for interpolate
        # [H, W] -> [1, 1, H, W]
        raw_label = raw_label.unsqueeze(0).unsqueeze(0)
        depth = depth.unsqueeze(0).unsqueeze(0)

        raw_label = F.interpolate(
            raw_label.float(),
            size=self.resize,
            mode="nearest"
        )

        depth = F.interpolate(
            depth.float(),
            size=self.resize,
            mode="bilinear"
        )

        # [1, 1, H, W] -> [H, W]
        raw_label = raw_label.squeeze(0).squeeze(0).long()
        depth = depth.squeeze(0).squeeze(0)

        label = self.class_map[raw_label] 

        boundary = get_boundary_map(label)
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