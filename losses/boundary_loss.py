from torch import nn
from losses.dice_loss import binary_dice_loss
from losses.focal_loss import BinaryFocalLoss

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = BinaryFocalLoss()

    def forward(self, pred, target):
        return 0.5 * self.criterion(pred, target) + 0.5 * binary_dice_loss(pred, target)
    