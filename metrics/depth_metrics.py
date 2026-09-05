import torch

class DepthMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_sq_err = 0.0
        self.total_abs_rel = 0.0
        self.total_delta1 = 0.0
        self.total_delta2 = 0.0
        self.total_delta3 = 0.0
        self.total_pixels = 0

    @staticmethod
    def get_eigen_mask(target):
        # NYUv2 standard Eigen crop: [45:471, 41:601] on 480x640 (Eigen et al., NIPS 2014)
        H, W = target.shape[-2:]
        y1 = int(round(45.0 / 480.0 * H))
        y2 = int(round(471.0 / 480.0 * H))
        x1 = int(round(41.0 / 640.0 * W))
        x2 = int(round(601.0 / 640.0 * W))

        crop_mask = torch.zeros_like(target, dtype=torch.bool)
        crop_mask[..., y1:y2, x1:x2] = True
        valid = (target >= 1e-3) & (target <= 10.0) & crop_mask
        return valid

    def update(self, pred, target):
        pred = pred.float()
        target = target.float()
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        valid = self.get_eigen_mask(target)
        if not valid.any():
            return

        p = pred[valid]
        t = target[valid]

        self.total_sq_err += torch.sum((p - t) ** 2).item()
        self.total_abs_rel += torch.sum(torch.abs(p - t) / t).item()

        ratio = torch.maximum(p / t, t / p)
        self.total_delta1 += (ratio < 1.25).sum().item()
        self.total_delta2 += (ratio < 1.25 ** 2).sum().item()
        self.total_delta3 += (ratio < 1.25 ** 3).sum().item()
        self.total_pixels += p.numel()

    def get_results(self):
        if self.total_pixels == 0:
            return {
                "rmse": 0.0,
                "abs_rel": 0.0,
                "delta1": 0.0,
                "delta2": 0.0,
                "delta3": 0.0,
            }

        rmse = (self.total_sq_err / self.total_pixels) ** 0.5
        abs_rel = self.total_abs_rel / self.total_pixels
        delta1 = self.total_delta1 / self.total_pixels
        delta2 = self.total_delta2 / self.total_pixels
        delta3 = self.total_delta3 / self.total_pixels

        return {
            "rmse": rmse,
            "abs_rel": abs_rel,
            "delta1": delta1,
            "delta2": delta2,
            "delta3": delta3,
        }

    @staticmethod
    def compute(pred, target):
        meter = DepthMetrics()
        meter.update(pred, target)
        return meter.get_results()