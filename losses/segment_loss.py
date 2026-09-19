import torch
from torch import nn
from losses.lovasz_loss import LovaszSoftmaxLoss

class SegmentLoss(nn.Module):
    def __init__(self, ignore_index=0, lovasz_weight=0.5):
        super().__init__()
        self.ignore_index = ignore_index
        self.lovasz_weight = lovasz_weight

        # ScanNet25k 40-class empirical training frequencies
        fg_pcts = torch.tensor([
            24.51, 19.37,  5.04,  3.60,  6.30,  2.72,  5.11,  4.47,  1.67,  1.31,
                0.45,  0.83,  0.05,  2.29,  1.68,  1.11,  0.69,  0.56,  0.12,  0.07,
                0.56,  0.56,  0.63,  0.86,  0.26,  0.03,  0.38,  0.53,  0.59,  0.60,
                0.04,  0.23,  0.66,  0.48,  0.11,  0.68,  0.13,  2.32,  3.43,  4.94
        ], dtype=torch.float32)

        # Smooth median-frequency weighting bounded to [0.25, 3.5]
        med = torch.median(fg_pcts)
        fg_weights = med / (fg_pcts + 0.1)
        fg_weights = torch.clamp(fg_weights, min=0.25, max=3.5)
        fg_weights = fg_weights / fg_weights.mean()

        weights = torch.zeros(41, dtype=torch.float32)
        weights[1:] = fg_weights
        weights[0] = 0.0

        self.register_buffer("class_weights", weights)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index, label_smoothing=0.05)
        self.lovasz = LovaszSoftmaxLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        lovasz_loss = self.lovasz(pred, target)
        return ce_loss + self.lovasz_weight * lovasz_loss
    