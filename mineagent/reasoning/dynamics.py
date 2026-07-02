import torch
import torch.nn as nn
import torch.nn.functional as F

from ..affector.affector import AffectorOutput, LinearAffector


class InverseDynamics(nn.Module):
    def __init__(self, embed_dim: int, num_keys: int | None = None):
        super().__init__()
        kwargs = {} if num_keys is None else {"num_keys": num_keys}
        self.affector = LinearAffector(embed_dim * 2, **kwargs)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> AffectorOutput:
        """
        Predict the action distribution that transitions from state x1 to x2.

        Parameters
        ----------
        x1 : torch.Tensor  (BS, embed_dim)
        x2 : torch.Tensor  (BS, embed_dim)

        Returns
        -------
        AffectorOutput
        """
        x = torch.cat((x1, x2), dim=1)
        return self.affector(x)


class ForwardDynamics(nn.Module):
    def __init__(self, embed_dim: int, action_dim: int | None = None):
        super().__init__()
        from ..client.protocol import NUM_KEYS

        if action_dim is None:
            action_dim = NUM_KEYS + 3 + 3
        self.l1 = nn.Linear(embed_dim + action_dim, 512)
        self.l2 = nn.Linear(512, embed_dim)

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
