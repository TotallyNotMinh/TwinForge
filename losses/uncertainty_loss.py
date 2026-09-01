from torch import nn
import torch

class KendallMultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=3, min_log_var=-1.0, max_log_var=1.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

    def forward(self, losses):
        total = 0.0

        for i, loss in enumerate(losses):
            # Clamp log_vars
            log_var = torch.clamp(self.log_vars[i], min=self.min_log_var, max=self.max_log_var)
            
            total += (
                0.5 * torch.exp(-log_var) * loss
                + 0.5 * log_var
            )

        return total
