import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F


class NYUv2Dataset(Dataset):

    def __init__(self, path, split, augment=True, resize=(640, 320)):
        self.path = path
        self.data = None

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

        data = h5py.File(self.path, "r")
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
        if self.data == None: data = h5py.File(self.path, "r") 
        idx = self.indices[idx]

        image = torch.from_numpy(data["images"][idx])
        depth = torch.from_numpy(data["depths"][idx])
        label = torch.from_numpy(data["labels"][idx])

        image = image.float() / 255.0
        image = self.rgb_transform(image)

        # F.interpolate() expects image-like tensors in (N, C, H, W) format:
        # Add dimensions for interpolate
        # [H, W] -> [1, 1, H, W]
        label = label.unsqueeze(0).unsqueeze(0)
        depth = depth.unsqueeze(0).unsqueeze(0)

        label = F.interpolate(
            label.float(),
            size=self.resize,
            mode="nearest"
        )

        depth = F.interpolate(
            depth.float(),
            size=self.resize,
            mode="bilinear"
        )

        # [1, 1, H, W] -> [H, W]
        label = label.squeeze(0).squeeze(0).long()
        depth = depth.squeeze(0).squeeze(0)

        return image, depth, label
    
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