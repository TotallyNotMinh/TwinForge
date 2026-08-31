import torch

class SegmentationMetrics:

    @staticmethod
    def compute(pred, target, num_classes=40):

        pred = torch.argmax(pred, dim=1)

        ious = []
        dices = []

        for cls in range(num_classes):

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
        pixel_accuracy = (
            (pred == target).float().mean()
        )

        miou = torch.stack(ious).mean()
        mean_dice = torch.stack(dices).mean()

        return {
            "miou": miou.item(),
            "dice": mean_dice.item(),
            "pixel_acc": pixel_accuracy.item(),
        }
