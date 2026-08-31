import torch
import torch.nn.functional as F

def multiclass_dice_loss(pred, target, num_classes=41, smooth=1e-6):
    pred = F.softmax(pred, dim=1)

    target = F.one_hot(
        target,
        num_classes=num_classes
    ).permute(0, 3, 1, 2).float()

    intersection = (pred * target).sum(dim=(2, 3))

    denominator = (
        pred.sum(dim=(2, 3)) +
        target.sum(dim=(2, 3))
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return 1.0 - dice[:, 1:].mean()


def binary_dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(1, 2, 3))

    denominator = (
        pred.sum(dim=(1, 2, 3))
        + target.sum(dim=(1, 2, 3))
    )

    dice = (2 * intersection + smooth) / (
        denominator + smooth
    )

    return 1 - dice.mean()