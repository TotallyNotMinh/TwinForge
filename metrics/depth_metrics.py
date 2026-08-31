import torch

class DepthMetrics:

    @staticmethod
    def compute(pred, target):

        pred = pred.float()
        target = target.float()

        pred = pred.squeeze(1)
        target = target.squeeze(1)

        eps = 1e-6

        # RMSE
        rmse = torch.sqrt(
            torch.mean((pred - target) ** 2)
        )

        # Absolute Relative Error
        abs_rel = torch.mean(
            torch.abs(pred - target) / (target + eps)
        )

        # Delta accuracy
        ratio = torch.maximum(
            pred / (target + eps),
            target / (pred + eps)
        )

        delta1 = (ratio < 1.25).float().mean()
        delta2 = (ratio < 1.25 ** 2).float().mean()
        delta3 = (ratio < 1.25 ** 3).float().mean()

        return {
            "rmse": rmse.item(),
            "abs_rel": abs_rel.item(),
            "delta1": delta1.item(),
            "delta2": delta2.item(),
            "delta3": delta3.item(),
        }