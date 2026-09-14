import torch
import torch.nn.functional as F

class BoundaryMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, pred, target):
        pred = pred.float()
        target = target.float()

        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        # Upsample pred if spatial dimensions differ
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False
            ).squeeze(1)

        pred_bin = (torch.sigmoid(pred) > self.threshold).bool()
        target_bin = (target > 0.5).bool()

        self.tp += (pred_bin & target_bin).sum().item()
        self.fp += (pred_bin & ~target_bin).sum().item()
        self.fn += (~pred_bin & target_bin).sum().item()

    def get_results(self):
        eps = 1e-6
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)

        return {
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }

    @staticmethod
    def compute(pred, target, threshold=0.5):
        meter = BoundaryMetrics(threshold=threshold)
        meter.update(pred, target)
        return meter.get_results()

