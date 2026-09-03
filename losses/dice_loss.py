import torch
import torch.nn.functional as F

def multiclass_dice_loss(pred, target, num_classes=41, smooth=1e-6, ignore_index=0):
    pred = F.softmax(pred, dim=1)
    
    # Mask out ignore_index
    valid_mask = (target != ignore_index).unsqueeze(1).float()
    pred = pred * valid_mask

    target_one_hot = F.one_hot(target.clamp(min=0), num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = target_one_hot * valid_mask

    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    denominator = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    # Exclude class 0
    dice_fg = dice[:, 1:]
    present = target_one_hot[:, 1:].sum(dim=(2, 3)) > 0         

    return 1.0 - (dice_fg * present).sum() / present.sum().clamp_min(1) # Calculate dice score based on classes present in the image not on all 40 classes


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