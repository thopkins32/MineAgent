from typing import Generator, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mineagent.memory.trajectory import TrajectoryBuffer
from mineagent.config import ICMConfig


@dataclass
class ICMSample:
    """
    A sample of items from a trajectory.

    Attributes
    ----------
    features : torch.Tensor
        Visual features computed from raw observations
    next_features : torch.Tensor
        Visual features representing the next observation from the environment
    actions : torch.Tensor
        Actions taken using features
    """

    features: torch.Tensor
    next_features: torch.Tensor
    actions: torch.Tensor

    def __len__(self):
        return self.features.shape[0]

    def get(
        self, shuffle: bool = False, batch_size: Union[int, None] = None
    ) -> Generator["ICMSample", None, None]:
        """
        Generates samples from the current sample.

        Parameters
        ----------
        shuffle : bool, optional
            Randomize the order of the trajectory buffer
        batch_size : int, optional
            Yield the data in batches instead of all at once

        Yields
        ------
        ICMSample
            Subset of information about the trajectory, possibly shuffled
        """
        size = len(self.features)
        if shuffle:
            indices = np.random.permutation(size)
        else:
            indices = np.arange(size)

        if batch_size is None:
            batch_size = size

        start_idx = 0
        while start_idx < size:
            batch_ind = indices[start_idx : start_idx + batch_size]
            yield ICMSample(
                features=self.features[batch_ind],
                next_features=self.next_features[batch_ind],
                actions=self.actions[batch_ind],
            )

            start_idx += batch_size


class ICM:
    """
    Intrinsic Curiosity Module (ICM). This enhances the reward from the environment by adding a
    component that favors situations where next-state prediction error is high.
    """

    def __init__(
        self,
        forward_dynamics: nn.Module,
        inverse_dynamics: nn.Module,
        config: ICMConfig,
    ):
        self.config = config

        self.forward_dynamics = forward_dynamics
        self.inverse_dynamics = inverse_dynamics

        self.forward_dynamics_optimizer = optim.Adam(
            self.forward_dynamics.parameters(), lr=config.forward_dynamics_lr
        )
        self.inverse_dynamics_optimizer = optim.Adam(
            self.inverse_dynamics.parameters(), lr=config.inverse_dynamics_lr
        )

    def intrinsic_reward(
        self,
        embedding: torch.Tensor,
        action: torch.Tensor,
        next_embedding: torch.Tensor,
    ) -> float:
        """
        Computes the intrinsic reward of the agent. This is used as a proxy for curiosity based on how well
        the agent is able to predict the `next_embedding` given the `embedding` and `action` vectors.
        The idea is that if the prediction error is low, then we have probably seen similar observations many times before
        and we should explore elsewhere.

        Parameters
        ----------
        embedding : torch.Tensor
            Features extracted from the current observation
            Expected shape (1, self.embed_dim)
        action : torch.Tensor
            Action taken in the environment to go from the current observation -> next observation
            Expected shape (1, self.action_dim)
        next_embedding : torch.Tensor
            Features extracted from the next observation
            Expected shape (1, self.embed_dim)

        Returns
        -------
        float
            Weighted mean-squared error between the prediction from the forward dynamics model
            and the `next_embedding`
        """
        with torch.no_grad():
            predicted_next_state = self.forward_dynamics(embedding, action)
        return (
            self.config.scaling_factor
            * F.mse_loss(predicted_next_state, next_embedding).item()
        )

    def _compute_inverse_dynamics_loss(self, sample: ICMSample) -> torch.Tensor:
        feat, next_feat, actions = (
            sample.features,
            sample.next_features,
            sample.actions,
        )
        actions_pred = self.inverse_dynamics(feat, next_feat)
        return (
            F.nll_loss(torch.log(actions_pred[0]), actions[:, 0])
            + F.nll_loss(torch.log(actions_pred[1]), actions[:, 1])
            + F.nll_loss(torch.log(actions_pred[2]), actions[:, 2])
            + F.nll_loss(torch.log(actions_pred[3]), actions[:, 3])
            + F.nll_loss(torch.log(actions_pred[4]), actions[:, 4])
            + F.nll_loss(torch.log(actions_pred[5]), actions[:, 5])
            + F.nll_loss(torch.log(actions_pred[6]), actions[:, 6])
            + F.nll_loss(torch.log(actions_pred[7]), actions[:, 7])
            + F.gaussian_nll_loss(
                actions_pred[8][:, 0], actions[:, 8], actions_pred[9][:, 0] ** 2
            )
            + F.gaussian_nll_loss(
                actions_pred[8][:, 1], actions[:, 9], actions_pred[9][:, 1] ** 2
            )
        )

    def _update_inverse_dynamics(self, data: ICMSample) -> None:
        self.inverse_dynamics.train()
        buffer_size = len(data)
        for sample in data.get(
            shuffle=True,
            batch_size=buffer_size // self.config.train_inverse_dynamics_iters,
        ):
            self.inverse_dynamics_optimizer.zero_grad()
            loss = self._compute_inverse_dynamics_loss(sample)
            loss.backward()
            self.inverse_dynamics_optimizer.step()
        self.inverse_dynamics.eval()

    def _compute_forward_dynamics_loss(self, sample: ICMSample) -> torch.Tensor:
        feat, next_feat, actions = (
            sample.features,
            sample.next_features,
            sample.actions,
        )
        next_feat_pred = self.forward_dynamics(feat, actions)
        return F.mse_loss(next_feat_pred, next_feat)

    def _update_forward_dynamics(self, data: ICMSample) -> None:
        self.forward_dynamics.train()
        buffer_size = len(data)
        for sample in data.get(
            shuffle=True,
            batch_size=buffer_size // self.config.train_forward_dynamics_iters,
        ):
            self.forward_dynamics_optimizer.zero_grad()
            loss = self._compute_forward_dynamics_loss(sample)
            loss.backward()
            self.forward_dynamics_optimizer.step()
        self.forward_dynamics.eval()

    def _finalize_trajectory(self, data: TrajectoryBuffer) -> ICMSample:
        features_buffer = list(data.features_buffer)
        features = torch.stack(features_buffer[:-1])
        next_features = torch.stack(features_buffer[1:])
        actions = torch.stack(list(data.actions_buffer)[:-1])
        return ICMSample(features, next_features, actions)

    def update(self, data: TrajectoryBuffer) -> None:
        sample = self._finalize_trajectory(data)
        self._update_inverse_dynamics(sample)
        self._update_forward_dynamics(sample)
