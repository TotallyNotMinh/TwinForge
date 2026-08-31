import random

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class NYUv2Augmentation:
    def __init__(self, output_size=(160, 320)):
        self.output_size = output_size

    def __call__(self, image, depth, label, boundary):

        # ============================================================
        # 1. Random horizontal flip
        # ============================================================

        if random.random() < 0.5:

            image = TF.hflip(image)
            depth = TF.hflip(depth)
            label = TF.hflip(label)
            boundary = TF.hflip(boundary)

        # ============================================================
        # 2. Random resized crop
        #
        # SAME crop for RGB / depth / label / boundary
        # ============================================================

        if random.random() < 0.3:

            _, H, W = image.shape

            # Random crop size
            scale = random.uniform(0.8, 1.0)

            crop_h = int(H * scale)
            crop_w = int(W * scale)

            # Random crop position
            top = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)

            image = TF.resized_crop(
                image,
                top,
                left,
                crop_h,
                crop_w,
                self.output_size,
                interpolation=InterpolationMode.BILINEAR
            )

            depth = TF.resized_crop(
                depth.unsqueeze(0),
                top,
                left,
                crop_h,
                crop_w,
                self.output_size,
                interpolation=InterpolationMode.BILINEAR
            ).squeeze(0)

            label = TF.resized_crop(
                label.unsqueeze(0).float(),
                top,
                left,
                crop_h,
                crop_w,
                self.output_size,
                interpolation=InterpolationMode.NEAREST
            ).squeeze(0).long()

            boundary = TF.resized_crop(
                boundary.unsqueeze(0),
                top,
                left,
                crop_h,
                crop_w,
                self.output_size,
                interpolation=InterpolationMode.NEAREST
            ).squeeze(0)

        # ============================================================
        # 3. Random rotation
        #
        # SAME angle for all modalities
        # ============================================================

        if random.random() < 0.3:

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

            boundary = TF.rotate(
                boundary,
                angle,
                interpolation=InterpolationMode.NEAREST
            )

        # ============================================================
        # 4. RGB brightness
        # ============================================================

        if random.random() < 0.3:

            image = TF.adjust_brightness(
                image,
                brightness_factor=random.uniform(0.8, 1.2)
            )

        # ============================================================
        # 5. RGB contrast
        # ============================================================

        if random.random() < 0.3:

            image = TF.adjust_contrast(
                image,
                contrast_factor=random.uniform(0.8, 1.2)
            )

        # ============================================================
        # 6. RGB saturation
        # ============================================================

        if random.random() < 0.3:

            image = TF.adjust_saturation(
                image,
                saturation_factor=random.uniform(0.8, 1.2)
            )

        # ============================================================
        # 7. RGB hue
        # ============================================================

        if random.random() < 0.1:

            image = TF.adjust_hue(
                image,
                hue_factor=random.uniform(-0.05, 0.05)
            )

        # ============================================================
        # 8. RGB Gaussian noise
        # ============================================================

        if random.random() < 0.2:

            noise = torch.randn_like(image) * 0.02

            image = image + noise
            image = torch.clamp(image, 0.0, 1.0)

        # ============================================================
        # 9. RGB Gaussian blur
        # ============================================================

        if random.random() < 0.1:

            image = TF.gaussian_blur(
                image,
                kernel_size=5,
                sigma=random.uniform(0.1, 1.5)
            )

        # ============================================================
        # 10. Small depth noise
        # ============================================================

        if random.random() < 0.1:

            noise = torch.randn_like(depth) * 0.005

            depth = depth + noise
            depth = torch.clamp(depth, min=0.0)

        return image, depth, label, boundary