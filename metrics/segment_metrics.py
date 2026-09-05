import torch

class SegmentationMetrics:

    @staticmethod
    def compute(pred, target, num_classes=41):

        pred = torch.argmax(pred, dim=1)

        ious = []
        dices = []

        for cls in range(1, num_classes): # Exclude unlabeled class (0)

            pred_cls = pred == cls
            target_cls = target == cls

            intersection = (pred_cls & target_cls).sum().float()

            union = (pred_cls | target_cls).sum().float()

            pred_area = pred_cls.sum().float()
            target_area = target_cls.sum().float()

            # IoU
            if union > 0:
                iou = intersection / union
                ious.append(iou)

            # Dice
            denominator = pred_area + target_area

            if denominator > 0:
                dice = 2 * intersection / denominator
                dices.append(dice)

        # Pixel accuracy
        valid = target > 0
        pixel_accuracy = (
            (pred[valid] == target[valid]).float().mean() if valid.any() else torch.tensor(0.0, device=pred.device)
        )

        miou = torch.stack(ious).mean() if len(ious) > 0 else torch.tensor(0.0, device=pred.device)
        mean_dice = torch.stack(dices).mean() if len(dices) > 0 else torch.tensor(0.0, device=pred.device)

        return {
            "miou": miou.item(),
            "dice": mean_dice.item(),
            "pixel_acc": pixel_accuracy.item(),
        }
