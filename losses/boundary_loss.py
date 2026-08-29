from torch import nn

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred, label):
        return self.criterion(pred, label)
    