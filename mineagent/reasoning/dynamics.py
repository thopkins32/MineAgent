import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import MultiDiscrete

from ..affector.affector import LinearAffector
from ..utils import add_forward_hooks


class InverseDynamics(nn.Module):
    def __init__(self, embed_dim: int, action_space: MultiDiscrete):
        super().__init__()
        # Multiply by 2 since we are concatenating the current obs and the next obs
        self.affector = LinearAffector(embed_dim * 2, action_space)

        # Monitoring
        self.start_monitoring()

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Inverse dynamics module forward pass. This module takes as input the
        feature representation of the current state and the next state and
        tries to predict the action vector that it took to get there.

        Parameters
        ----------
        x1 : torch.Tensor
            Feature representation of the current state of the environment
            Expected shape (BS, embed_dim)
        x2 : torch.Tensor
            Feature representation of the next state of the environment
            Expected shape (BS, embed_dim)

        Returns
        -------
        tuple
            Action taken in the environment to get from x1 to x2
            The tuple contains 10 tensors representing the distribution
            over each sub-action.
        """
        x = torch.cat((x1, x2), dim=1)
        x = self.affector(x)
        return x  # type: ignore

    def stop_monitoring(self):
        for hook in self.hooks:
            hook.remove()

    def start_monitoring(self):
        self.hooks = add_forward_hooks(self, "InverseDynamics")


class ForwardDynamics(nn.Module):
    def __init__(self, embed_dim: int, action_dim: int):
        super().__init__()
        self.l1 = nn.Linear(embed_dim + action_dim, 512)
        self.l2 = nn.Linear(512, embed_dim)

        # Monitoring
        self.start_monitoring()

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Forward dynamics module forward pass. This module takes as input the
        feature representation of the current state and an action and tries to
        predict the feature representation of the next state.

        Parameters
        ----------
        x : torch.Tensor
            Feature representation of the current state of the environment
            Expected shape (BS, embed_dim)
        a : torch.Tensor
            Action taken in the environment
            Expected shape (BS, action_dim)

        Returns
        -------
        torch.Tensor
            Feature representation of the next state of the environment
        """
        x = torch.cat((x, a), dim=-1)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

    def stop_monitoring(self):
        for hook in self.hooks:
            hook.remove()

    def start_monitoring(self):
        self.hooks = add_forward_hooks(self, "ForwardDynamics")
