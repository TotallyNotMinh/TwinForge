from torch import nn
import torch

class KendallMultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        total = 0.0

        for i, loss in enumerate(losses):
            total += (
                0.5 * torch.exp(-self.log_vars[i]) * loss
                + 0.5 * self.log_vars[i]
            )

        return total