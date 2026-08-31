from torch import nn
from losses.dice_loss import multiclass_dice_loss

class SegmentLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        return self.criterion(pred, target) + multiclass_dice_loss(pred, target)
    