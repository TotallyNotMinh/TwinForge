import random

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomResizedCrop

class NYUv2Augmentation:

    def __init__(self):
        pass

    def __call__(self, image, depth, label):
        depth_is_2d = (depth.dim() == 2)
        label_is_2d = (label.dim() == 2)
        if depth_is_2d:
            depth = depth.unsqueeze(0)
        if label_is_2d:
            label = label.unsqueeze(0)

        if random.random() < 0.7:
            image = TF.hflip(image)
            depth = TF.hflip(depth)
            label = TF.hflip(label)

        if random.random() < 0.5:
            angle = random.uniform(-5.0, 5.0)
            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR
            )
            depth = TF.rotate(
                depth,
                angle,
                interpolation=InterpolationMode.BILINEAR
            )
            label = TF.rotate(
                label,
                angle,
                interpolation=InterpolationMode.NEAREST
            )

        if random.random() < 0.5:
            image = TF.adjust_brightness(
                image,
                random.uniform(0.8, 1.2)
            )

        if random.random() < 0.5:
            image = TF.adjust_contrast(
                image,
                random.uniform(0.8, 1.2)
            )

        if random.random() < 0.5:
            image = TF.adjust_saturation(
                image,
                random.uniform(0.8, 1.2)
            )

        if random.random() < 0.3:
            image = TF.adjust_hue(
                image,
                random.uniform(-0.05, 0.05)
            )

        if random.random() < 0.2:
            noise = torch.randn_like(image) * 0.02
            image = torch.clamp(image + noise, 0.0, 1.0)

        if random.random() < 0.1:
            image = TF.gaussian_blur(
                image,
                kernel_size=5,
                sigma=random.uniform(0.1, 1.5)
            )

        if random.random() < 0.1:
            noise = torch.randn_like(depth) * 0.005
            depth = torch.clamp(depth + noise, min=0.0)

        if random.random() < 0.6:
            i, j, h, w = RandomResizedCrop.get_params(image, scale=(0.75, 1.25), ratio=[0.9, 1.1])
            output_size = (image.shape[-2], image.shape[-1]) # (H, W)

            image = TF.resized_crop(image, i, j, h, w, size=output_size, interpolation=InterpolationMode.BILINEAR)
            depth = TF.resized_crop(depth, i, j, h, w, size=output_size, interpolation=InterpolationMode.BILINEAR)
            label = TF.resized_crop(label, i, j, h, w, size=output_size, interpolation=InterpolationMode.NEAREST)

        if depth_is_2d:
            depth = depth.squeeze(0)
        if label_is_2d:
            label = label.squeeze(0)

        return image, depth, label