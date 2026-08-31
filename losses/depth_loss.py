from torch import nn
import torch

def berhu_loss(pred, target):

    diff = torch.abs(pred - target)

    c = 0.2 * diff.max().detach()

    loss = torch.where(
        diff <= c,
        diff,
        (diff ** 2 + c ** 2) / (2 * c)
    )

    return loss.mean()


class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        return 0.5 * self.criterion(pred, target) + 0.5 * berhu_loss(pred, target)
    