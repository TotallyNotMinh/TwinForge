import torch
import torch.nn.functional as F
import numpy as np

class DepthMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rmse_list = []
        self.abs_rel_list = []
        self.sq_rel_list = []
        self.rmse_log_list = []
        self.log10_list = []
        self.delta1_list = []
        self.delta2_list = []
        self.delta3_list = []

    @staticmethod
    def get_eigen_mask(target):
        # NYUv2 standard Eigen crop: [45:471, 41:601] on 480x640 (Eigen et al., NIPS 2014)
        H, W = target.shape[-2:]
        if H == 480 and W == 640:
            y1, y2, x1, x2 = 45, 471, 41, 601
        else:
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

        # If spatial dimensions differ, upsample pred to match target
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False
            ).squeeze(1)

        # Handle both batched (B, H, W) and single image (H, W)
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        batch_size = pred.shape[0]
        for b in range(batch_size):
            p_img = pred[b]
            t_img = target[b]

            valid = self.get_eigen_mask(t_img)
            if not valid.any():
                continue

            p = p_img[valid].clamp(min=1e-4, max=10.0)
            t = t_img[valid]

            diff = p - t
            abs_diff = torch.abs(diff)

            rmse = torch.sqrt(torch.mean(diff ** 2)).item()
            abs_rel = torch.mean(abs_diff / t).item()
            sq_rel = torch.mean((diff ** 2) / t).item()
            rmse_log = torch.sqrt(torch.mean((torch.log(p) - torch.log(t)) ** 2)).item()
            log10 = torch.mean(torch.abs(torch.log10(p) - torch.log10(t))).item()

            ratio = torch.maximum(p / t, t / p)
            delta1 = (ratio < 1.25).float().mean().item()
            delta2 = (ratio < 1.25 ** 2).float().mean().item()
            delta3 = (ratio < 1.25 ** 3).float().mean().item()

            self.rmse_list.append(rmse)
            self.abs_rel_list.append(abs_rel)
            self.sq_rel_list.append(sq_rel)
            self.rmse_log_list.append(rmse_log)
            self.log10_list.append(log10)
            self.delta1_list.append(delta1)
            self.delta2_list.append(delta2)
            self.delta3_list.append(delta3)

    def get_results(self):
        if not self.rmse_list:
            return {
                "rmse": 0.0,
                "abs_rel": 0.0,
                "sq_rel": 0.0,
                "rmse_log": 0.0,
                "log10": 0.0,
                "delta1": 0.0,
                "delta2": 0.0,
                "delta3": 0.0,
            }

        return {
            "rmse": float(np.mean(self.rmse_list)),
            "abs_rel": float(np.mean(self.abs_rel_list)),
            "sq_rel": float(np.mean(self.sq_rel_list)),
            "rmse_log": float(np.mean(self.rmse_log_list)),
            "log10": float(np.mean(self.log10_list)),
            "delta1": float(np.mean(self.delta1_list)),
            "delta2": float(np.mean(self.delta2_list)),
            "delta3": float(np.mean(self.delta3_list)),
        }

    @staticmethod
    def compute(pred, target):
        meter = DepthMetrics()
        meter.update(pred, target)
        return meter.get_results()