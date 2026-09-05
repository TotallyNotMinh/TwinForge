import torch
import torch.nn as nn
import torch.nn.functional as F

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, ignore_index=0):
    """
    Multi-class Lovasz-Softmax loss on flattened inputs.
    probas: [P, C] class probabilities (softmax)
    labels: [P] ground truth labels
    """
    if probas.numel() == 0:
        return torch.tensor(0.0, device=probas.device, requires_grad=True)

    valid = (labels != ignore_index)
    if not valid.any():
        return torch.tensor(0.0, device=probas.device, requires_grad=True)

    probas = probas[valid]
    labels = labels[valid]

    C = probas.size(1)
    losses = []
    # Evaluate foreground classes present in the batch
    present_classes = torch.unique(labels)
    for c in present_classes:
        c_item = c.item()
        if c_item == ignore_index:
            continue
        fg = (labels == c_item).float()
        errors = (fg - probas[:, c_item]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))

    if len(losses) == 0:
        return torch.tensor(0.0, device=probas.device, requires_grad=True)

    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        # pred: [B, C, H, W], target: [B, H, W]
        probas = F.softmax(pred, dim=1)
        B, C, H, W = probas.size()
        probas_flat = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
        target_flat = target.contiguous().view(-1)
        return lovasz_softmax_flat(probas_flat, target_flat, ignore_index=self.ignore_index)
