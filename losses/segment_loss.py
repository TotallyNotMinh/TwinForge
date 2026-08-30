from torch import nn

class SegmentLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, label):
        return self.criterion(pred, label)
    