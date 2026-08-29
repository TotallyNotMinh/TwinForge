from torch import nn

class SegmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, label):
        return self.criterion(pred, label)
    