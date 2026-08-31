import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=-1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        ce = F.cross_entropy(
            pred,
            target,
            reduction="none",
            ignore_index=self.ignore_index
        )

        pt = torch.exp(-ce)

        loss = self.alpha * (1 - pt) ** self.gamma * ce

        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            loss = loss[mask]

        return loss.mean()


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        target = target.float()

        bce = F.binary_cross_entropy_with_logits(
            pred,
            target,
            reduction="none"
        )

        prob = torch.sigmoid(pred)
        pt = prob * target + (1 - prob) * (1 - target)

        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = alpha_t * (1 - pt).pow(self.gamma) * bce

        return loss.mean()