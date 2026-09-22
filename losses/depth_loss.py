from torch import nn
import torch
import torch.nn.functional as F

def berhu_loss(pred, target):
    mask = target > 0
    if not mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    diff = torch.abs(pred[mask] - target[mask])

    c = 0.2 * diff.max().detach()

    loss = torch.where(
        diff <= c,
        diff,
        (diff ** 2 + c ** 2) / ((2 * c) + 1e-6)
    )

    return loss.mean()

def gm_loss(pred, gt, mask, boundary=None, scales=4, lambda_boundary=3.0):
    R = (torch.log(pred.clamp_min(1e-3)) - torch.log(gt.clamp_min(1e-3))) * mask
    total = 0.0
    for s in range(scales):
        k = 2 ** s
        r, m = R[..., ::k, ::k], mask[..., ::k, ::k].float()
        mx = m[..., :, 1:] * m[..., :, :-1]
        my = m[..., 1:, :] * m[..., :-1, :]

        # Weight edges with (1 + λ * boundary) instead of suppressing them with (1 - boundary)
        if boundary is not None:
            b_s = boundary[..., ::k, ::k].float()
            wx = 1.0 + lambda_boundary * b_s[..., :, 1:]
            wy = 1.0 + lambda_boundary * b_s[..., 1:, :]
            mx = mx * wx
            my = my * wy

        dx = (r[..., :, 1:] - r[..., :, :-1]).abs() * mx
        dy = (r[..., 1:, :] - r[..., :-1, :]).abs() * my
        total += (dx.sum() + dy.sum()) / (mx.sum() + my.sum() + 1e-8)
    return total / scales


class SILogLoss(nn.Module):
    def __init__(self, alpha=10.0, lambda_param=0.85):
        super().__init__()
        self.alpha = alpha
        self.lambda_param = lambda_param

    def forward(self, pred, target, mask=None):
        if mask is None:
            mask = (target > 0) & (pred > 0)
        else:
            mask = mask & (target > 0) & (pred > 0)

        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_valid = pred[mask].clamp_min(1e-4)
        target_valid = target[mask].clamp_min(1e-4)

        g = torch.log(pred_valid) - torch.log(target_valid)
        n = g.numel()
        dg2 = torch.sum(g ** 2) / n
        dg_sum = torch.sum(g) / n
        variance = dg2 - self.lambda_param * (dg_sum ** 2)
        return self.alpha * torch.sqrt(torch.clamp(variance, min=1e-8))


def temporal_depth_grad_loss(pred, target, B=None, T=None):
    """
    Computes temporal gradient matching loss across consecutive video frames.
    Accepts 4D (B*T, 1, H, W) or 5D (B, T, 1, H, W) tensors.
    """
    if pred.dim() == 4:
        if B is None or T is None or T <= 1:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        pred = pred.view(B, T, 1, pred.shape[-2], pred.shape[-1])
        target = target.view(B, T, 1, target.shape[-2], target.shape[-1])
    elif pred.dim() == 5:
        if pred.shape[1] <= 1:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # 1. Compute frame-to-frame temporal differences
    delta_pred = pred[:, 1:] - pred[:, :-1]        # (B, T-1, 1, H, W)
    delta_target = target[:, 1:] - target[:, :-1]  # (B, T-1, 1, H, W)

    # 2. Both current and next frame must have valid depth (> 0)
    mask_curr = target[:, :-1] > 0
    mask_next = target[:, 1:] > 0
    valid_mask = mask_curr & mask_next

    if not valid_mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # 3. Smooth L1 between predicted and ground truth temporal transitions
    return F.smooth_l1_loss(delta_pred[valid_mask], delta_target[valid_mask])


def get_gpu_boundary_map(label: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if label.dim() == 2:
        label = label.unsqueeze(0).unsqueeze(0)
    elif label.dim() == 3:
        label = label.unsqueeze(1)

    label_float = label.float()
    padding = kernel_size // 2

    # Find max and min label in each local patch on GPU
    max_label = F.max_pool2d(label_float, kernel_size=kernel_size, stride=1, padding=padding)
    min_label = -F.max_pool2d(-label_float, kernel_size=kernel_size, stride=1, padding=padding)
    return (max_label != min_label).float()

class DepthLoss(nn.Module):
    def __init__(self, alpha=10.0, lambda_param=0.85, l1_weight=0.5, grad_weight=1.5, temporal_weight=10.0):
        super().__init__()
        self.silog = SILogLoss(alpha=alpha, lambda_param=lambda_param)
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight
        self.temporal_weight = temporal_weight

    def forward(self, pred, target, label_or_boundary, B=None, T=None):
        mask = target > 0
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        silog = self.silog(pred, target, mask=mask)
    
        if label_or_boundary.dtype in (torch.int32, torch.int64):
            boundary = get_gpu_boundary_map(label_or_boundary)
        else:
            boundary = label_or_boundary

        grad = gm_loss(pred, target, boundary=boundary, mask=mask.float(), scales=4, lambda_boundary=3.0)
        l1 = torch.abs(pred[mask] - target[mask]).mean()
        spatial_loss = silog + self.grad_weight * grad + self.l1_weight * l1

        # Compute temporal loss if video frames are present
        if T is not None and T > 1:
            temp_loss = temporal_depth_grad_loss(pred, target, B=B, T=T)
            return spatial_loss + self.temporal_weight * temp_loss

        return spatial_loss
    