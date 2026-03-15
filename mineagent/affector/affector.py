from dataclasses import dataclass

import torch
import torch.nn as nn

from ..client.protocol import NUM_KEYS, MOUSE_DX_RANGE, MOUSE_DY_RANGE, SCROLL_RANGE
from ..utils import add_forward_hooks


@dataclass
class AffectorOutput:
    """All distribution parameters produced by the affector in a single object."""

    key_logits: torch.Tensor
    mouse_dx_mean: torch.Tensor
    mouse_dx_std: torch.Tensor
    mouse_dy_mean: torch.Tensor
    mouse_dy_std: torch.Tensor
    mouse_button_logits: torch.Tensor
    scroll_mean: torch.Tensor
    scroll_std: torch.Tensor
    focus_means: torch.Tensor
    focus_stds: torch.Tensor


class LinearAffector(nn.Module):
    """
    Feed-forward affector (action) module.

    Produces distribution parameters for the raw-input action space:
    - Independent Bernoulli logits for each key in KEY_LIST
    - Gaussian parameters (mean, std) for mouse_dx, mouse_dy, scroll_delta
    - Independent Bernoulli logits for 3 mouse buttons
    - Gaussian parameters for the internal focus/ROI mechanism
    """

    def __init__(self, embed_dim: int, num_keys: int = NUM_KEYS):
        super().__init__()

        self.num_keys = num_keys

        # Binary key logits (one per key)
        self.key_head = nn.Linear(embed_dim, num_keys)

        # Mouse movement (mean + log-std for dx and dy)
        self.mouse_dx_mean = nn.Linear(embed_dim, 1)
        self.mouse_dx_logstd = nn.Linear(embed_dim, 1)
        self.mouse_dy_mean = nn.Linear(embed_dim, 1)
        self.mouse_dy_logstd = nn.Linear(embed_dim, 1)

        # Mouse buttons (3 independent Bernoulli logits)
        self.mouse_button_head = nn.Linear(embed_dim, 3)

        # Scroll (mean + log-std)
        self.scroll_mean = nn.Linear(embed_dim, 1)
        self.scroll_logstd = nn.Linear(embed_dim, 1)

        # Internal focus / region-of-interest
        self.focus_means = nn.Linear(embed_dim, 2)
        self.focus_logstds = nn.Linear(embed_dim, 2)

        self.softplus = nn.Softplus()

        # Monitoring
        self.start_monitoring()

    def forward(self, x: torch.Tensor) -> AffectorOutput:
        return AffectorOutput(
            key_logits=self.key_head(x),
            mouse_dx_mean=self.mouse_dx_mean(x).squeeze(-1),
            mouse_dx_std=self.softplus(self.mouse_dx_logstd(x)).squeeze(-1),
            mouse_dy_mean=self.mouse_dy_mean(x).squeeze(-1),
            mouse_dy_std=self.softplus(self.mouse_dy_logstd(x)).squeeze(-1),
            mouse_button_logits=self.mouse_button_head(x),
            scroll_mean=self.scroll_mean(x).squeeze(-1),
            scroll_std=self.softplus(self.scroll_logstd(x)).squeeze(-1),
            focus_means=self.focus_means(x),
            focus_stds=self.softplus(self.focus_logstds(x)),
        )

    def stop_monitoring(self):
        for hook in self.hooks:
            hook.remove()

    def start_monitoring(self):
        self.hooks = add_forward_hooks(self, "Affector")
