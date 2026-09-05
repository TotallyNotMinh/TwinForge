import torch
from torch import nn
from losses.lovasz_loss import LovaszSoftmaxLoss

class SegmentLoss(nn.Module):
    def __init__(self, ignore_index=0, lovasz_weight=0.5):
        super().__init__()
        self.ignore_index = ignore_index
        self.lovasz_weight = lovasz_weight

        # NYUv2 40-class empirical training frequencies
        fg_pcts = torch.tensor([
            26.17, 11.13,  7.96,  4.26,  4.10,  2.86,  2.45,  2.43,  2.67,  2.33,
             2.42,  1.62,  1.90,  1.20,  1.50,  1.20,  1.11,  0.97,  1.29,  0.78,
             1.02,  1.76,  0.65,  0.64,  0.67,  0.45,  0.40,  0.49,  0.50,  0.32,
             0.37,  0.31,  0.34,  0.29,  0.33,  0.26,  0.25,  2.75,  2.27,  5.60
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
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index)
        self.lovasz = LovaszSoftmaxLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        lovasz_loss = self.lovasz(pred, target)
        return ce_loss + self.lovasz_weight * lovasz_loss
    