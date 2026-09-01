from torch import nn
from losses.dice_loss import multiclass_dice_loss
from losses.focal_loss import MultiFocalLoss

class SegmentLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super().__init__()

        self.class_pct = [17.3811, 21.3955,  9.0635,  6.2009,  3.7681,  3.3295,  2.6782,  2.1378,
         2.1623,  2.1201,  1.9332,  2.0844,  1.4385,  1.6721,  1.1108,  1.0195,
         1.1034,  0.9162,  0.8270,  0.9961,  0.7352,  0.7007,  1.3741,  0.5908,
         0.5826,  0.4941,  0.3902,  0.3662,  0.3768,  0.3375,  0.3179,  0.2946,
         0.2790,  0.2654,  0.2669,  0.2770,  0.2544,  0.2303,  1.9691,  1.9088,
         4.6503]
        
        self.criterion = MultiFocalLoss(class_pct=self.class_pct)

    def forward(self, pred, target):
        return self.criterion(pred, target) + multiclass_dice_loss(pred, target)
    