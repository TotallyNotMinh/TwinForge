import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiFocalLoss(nn.Module):
    def __init__(self, class_pct, beta=0.999, gamma=2.0, ignore_index=0):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

        class_pct = torch.as_tensor(class_pct, dtype=torch.float32)
        effective_num = 1.0 - torch.pow(beta, class_pct)
        alpha = (1.0 - beta) / effective_num
        alpha = alpha / alpha.sum() * len(alpha)

        self.register_buffer("alpha", alpha)  # [C], moves with .to(device)/.cuda()

    def forward(self, pred, target):
        ce = F.cross_entropy(
            pred, target, reduction="none", ignore_index=self.ignore_index
        )  # 

        pt = torch.exp(-ce)

        valid_mask = target != self.ignore_index
        target_safe = target.clone()
        target_safe[~valid_mask] = 0  # dummy index, masked out below

        alpha_t = self.alpha[target_safe]  

        loss = alpha_t * (1 - pt) ** self.gamma * ce
        loss = loss[valid_mask]

        return loss.mean()

