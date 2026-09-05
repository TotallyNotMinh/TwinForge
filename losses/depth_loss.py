from torch import nn
import torch

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

def boundary_guided_depth_grad_loss(pred_depth, gt_depth, boundary_pred, mask=None):
    """
    boundary_pred: [B, 1, H, W] sigmoid output from your BoundaryHead (0=flat region, 1=edge)
    """
    pred_dx = pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1]
    pred_dy = pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :]
    gt_dx = gt_depth[:, :, :, 1:] - gt_depth[:, :, :, :-1]
    gt_dy = gt_depth[:, :, 1:, :] - gt_depth[:, :, :-1, :]

    # Ground truth boundaries are binary [0.0, 1.0] (or predicted logits if passed)
    if boundary_pred.dtype in (torch.float16, torch.float32, torch.float64):
        if (boundary_pred < 0.0).any() or (boundary_pred > 1.0).any():
            b = torch.sigmoid(boundary_pred).detach()
        else:
            b = boundary_pred.detach()
    else:
        b = boundary_pred.float().detach()

    weight_x = (1.0 - b[:, :, :, 1:])
    weight_y = (1.0 - b[:, :, 1:, :])

    loss_x = torch.abs(pred_dx - gt_dx) * weight_x
    loss_y = torch.abs(pred_dy - gt_dy) * weight_y

    if mask is not None:
        mask_x = mask[:, :, :, 1:]
        mask_y = mask[:, :, 1:, :]
        loss_x = loss_x * mask_x
        loss_y = loss_y * mask_y
        return (loss_x.sum() + loss_y.sum()) / (mask_x.sum() + mask_y.sum() + 1e-8)

    return loss_x.mean() + loss_y.mean()


class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic (SILog) loss (Eigen et al., DPT, AdaBins).
    L_silog = alpha * sqrt( 1/N sum(g_i^2) - lambda/N^2 (sum g_i)^2 )
    where g_i = ln(pred_i) - ln(target_i).
    """
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


import torch.nn.functional as F

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
    def __init__(self, alpha=10.0, lambda_param=0.85, l1_weight=0.5):
        super().__init__()
        self.silog = SILogLoss(alpha=alpha, lambda_param=lambda_param)
        self.l1_weight = l1_weight

    def forward(self, pred, target, label_or_boundary):
        mask = target > 0
        silog = self.silog(pred, target, mask=mask)

        # If integer class labels are passed, extract boundaries rapidly in batch on GPU
        if label_or_boundary.dtype in (torch.int32, torch.int64):
            boundary = get_gpu_boundary_map(label_or_boundary)
        else:
            boundary = label_or_boundary

        grad = boundary_guided_depth_grad_loss(pred, target, boundary, mask=mask.float())
        l1 = F.smooth_l1_loss(pred[mask], target[mask])

        return silog + 0.5 * grad + self.l1_weight * l1
    