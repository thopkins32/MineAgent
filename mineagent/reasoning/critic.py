import torch.nn as nn

from ..utils import add_forward_hooks


class LinearCritic(nn.Module):
    """
    Feed-forward critic module.

    This module estimates future reward for a given input.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.l1 = nn.Linear(embed_dim, 1)

        # Monitoring
        self.start_monitoring()

    def forward(self, x):
        return self.l1(x)

    def stop_monitoring(self):
        for hook in self.hooks:
            hook.remove()

    def start_monitoring(self):
        self.hooks = add_forward_hooks(self, "Critic")
