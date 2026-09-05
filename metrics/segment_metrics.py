import torch

class SegmentationMetrics:
    def __init__(self, num_classes=41):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64
        )

    def update(self, pred, target):
        # pred: [B, C, H, W] or [B, H, W], target: [B, H, W]
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)

        valid = (target > 0) & (target < self.num_classes)
        p = pred[valid].long().cpu()
        t = target[valid].long().cpu()

        if p.numel() == 0:
            return

        bin_counts = torch.bincount(
            t * self.num_classes + p,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.conf_matrix += bin_counts

    def get_results(self):
        conf = self.conf_matrix.float()
        # Evaluate classes 1 to num_classes-1 (class 0 is unlabeled/void)
        tp = conf.diag()[1:]
        target_counts = conf[1:, :].sum(dim=1)
        pred_counts = conf[:, 1:].sum(dim=0)

        union = target_counts + pred_counts - tp
        present_mask = target_counts > 0

        if not present_mask.any():
            return {"miou": 0.0, "dice": 0.0, "pixel_acc": 0.0}

        iou = tp[present_mask] / union[present_mask].clamp_min(1e-6)
        dice = (2.0 * tp[present_mask]) / (target_counts[present_mask] + pred_counts[present_mask]).clamp_min(1e-6)

        miou = iou.mean().item()
        mean_dice = dice.mean().item()
        pixel_acc = (tp.sum() / target_counts.sum().clamp_min(1.0)).item()

        return {
            "miou": miou,
            "dice": mean_dice,
            "pixel_acc": pixel_acc,
        }

    @staticmethod
    def compute(pred, target, num_classes=41):
        meter = SegmentationMetrics(num_classes=num_classes)
        meter.update(pred, target)
        return meter.get_results()
