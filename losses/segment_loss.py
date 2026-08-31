from torch import nn
from losses.dice_loss import multiclass_dice_loss
from losses.focal_loss import MultiFocalLoss

class SegmentLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super().__init__()

        self.criterion = MultiFocalLoss()

    def forward(self, pred, target):
        return self.criterion(pred, target) + multiclass_dice_loss(pred, target)
    