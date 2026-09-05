from torch import nn
import torch

class KendallMultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=3, min_log_var=-2.0, max_log_var=2.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

    def forward(self, losses):
        center = (self.max_log_var + self.min_log_var) / 2.0
        scale = (self.max_log_var - self.min_log_var) / 2.0
        total = 0.0

        for i, loss in enumerate(losses):
            # Smoothly bounded log_var preserving continuous non-zero gradient
            log_var = center + scale * torch.tanh(self.log_vars[i])
            
            total += (
                0.5 * torch.exp(-log_var) * loss
                + 0.5 * log_var
            )

        return total
