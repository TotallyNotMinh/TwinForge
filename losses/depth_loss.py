from torch import nn
import torch

def berhu_loss(pred, target):

    diff = torch.abs(pred - target)

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


class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, bound_pred):
        return 0.5 * boundary_guided_depth_grad_loss(pred, target, bound_pred) + 0.5 * berhu_loss(pred, target)
    