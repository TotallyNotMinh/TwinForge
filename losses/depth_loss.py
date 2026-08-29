from torch import nn

class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = nn.L1Loss()

    def forward(self, pred, label):
        return self.criterion(pred, label)
    