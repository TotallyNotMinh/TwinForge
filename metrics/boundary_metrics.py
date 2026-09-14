import torch

class BoundaryMetrics:

    @staticmethod
    def compute(pred, target, threshold=0.5):

        pred = torch.sigmoid(pred)

        pred = pred > threshold
        target = target > 0.5

        pred = pred.bool()
        target = target.bool()

        true_positive = (pred & target).sum().float()
        false_positive = (pred & ~target).sum().float()
        false_negative = (~pred & target).sum().float()

        eps = 1e-6

        precision = (
            true_positive /
            (true_positive + false_positive + eps)
        )

        recall = (
            true_positive /
            (true_positive + false_negative + eps)
        )

        f1 = (
            2 * precision * recall /
            (precision + recall + eps)
        )

        return {
            "f1": f1.item(),
            "precision": precision.item(),
            "recall": recall.item(),
        }
