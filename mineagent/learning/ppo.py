from typing import Union, Generator
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mineagent.utils import joint_logp_action, discount_cumsum
from mineagent.config import PPOConfig
from mineagent.memory.trajectory import TrajectoryBuffer


@dataclass
class PPOSample:
    """
    A sample of items from a trajectory.

    Attributes
    ----------
    features : torch.Tensor
        Visual features computed from raw observations
    actions : torch.Tensor
        Environment actions taken (no focus)
    returns : torch.Tensor
        The return from the full trajectory computed so far
    advantages : torch.Tensor
        Advantages of taking each action over the alternative, e.g. `Q(s, a) - V(s)`
    log_probabilities: torch.Tensor
        Joint log probability of the environment action
    focus_actions : torch.Tensor
        Focus/ROI coordinates (stored separately from env actions)
    focus_log_probabilities : torch.Tensor
        Joint log probability of the focus action
    """

    features: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    log_probabilities: torch.Tensor
    focus_actions: torch.Tensor
    focus_log_probabilities: torch.Tensor

    def __len__(self):
        return self.features.shape[0]

    def get(
        self, shuffle: bool = False, batch_size: Union[int, None] = None
    ) -> Generator["PPOSample", None, None]:
        """
        Parameters
        ----------
        shuffle : bool, optional
            Randomize the order of the trajectory buffer
        batch_size : int, optional
            Yield the data in batches instead of all at once

        Yields
        ------
        PPOSample
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
            yield PPOSample(
                features=self.features[batch_ind],
                actions=self.actions[batch_ind],
                returns=self.returns[batch_ind],
                advantages=self.advantages[batch_ind],
                log_probabilities=self.log_probabilities[batch_ind],
                focus_actions=self.focus_actions[batch_ind],
                focus_log_probabilities=self.focus_log_probabilities[batch_ind],
            )

            start_idx += batch_size


class PPO:
    """Inspired by https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py"""

    def __init__(self, actor: nn.Module, critic: nn.Module, config: PPOConfig):
        """
        Parameters
        ----------
        actor : nn.Module
            Neural network for the actor
        critic : nn.Module
            Neural network for the critic
        config : PPOConfig
            Configuration for the PPO algorithm
        """
        # Actor & critic
        self.actor = actor
        self.critic = critic

        # Training duration
        self.train_actor_iters = config.train_actor_iters
        self.train_critic_iters = config.train_critic_iters

        # Learning hyperparameters
        self.discount_factor = config.discount_factor
        self.gae_discount_factor = config.gae_discount_factor
        self.clip_ratio = config.clip_ratio
        self.target_kl = config.target_kl
        self.focus_loss_coeff = config.focus_loss_coeff
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=config.actor_lr,
        )
        self.critic_optim = optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
        )

    def _compute_actor_loss(self, data: PPOSample) -> tuple[torch.Tensor, torch.Tensor]:
        feat, act, adv, logp_old = (
            data.features,
            data.actions,
            data.advantages,
            data.log_probabilities,
        )

        action_dist = self.actor(feat)

        # Standard clipped surrogate loss (env actions only)
        logp = joint_logp_action(action_dist, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        env_loss = -(torch.min(ratio * adv, clip_adv)).mean()

        # Separate REINFORCE loss for focus head (not clipped, not in KL)
        focus_dist = torch.distributions.Normal(
            action_dist.focus_means, action_dist.focus_stds
        )
        focus_logp_new = focus_dist.log_prob(data.focus_actions).sum(dim=-1)
        focus_loss = -(focus_logp_new * adv.detach()).mean()

        loss = env_loss + self.focus_loss_coeff * focus_loss

        # KL is based on env actions only
        kl = (logp_old - logp).mean()

        return loss, kl

    def _update_actor(self, data: PPOSample) -> None:
        self.actor.train()
        buffer_size = len(data)
        for sample in data.get(
            shuffle=True, batch_size=buffer_size // self.train_actor_iters
        ):
            self.actor_optim.zero_grad()
            loss, kl = self._compute_actor_loss(sample)
            if kl > 1.5 * self.target_kl:
                # early stopping
                break
            loss.backward()
            self.actor_optim.step()
        self.actor.eval()

    def _compute_critic_loss(self, data: PPOSample) -> torch.Tensor:
        feat, ret = (
            data.features,
            data.returns,
        )
        v = self.critic(feat)
        return F.mse_loss(v, ret)

    def _update_critic(self, data: PPOSample) -> None:
        self.critic.train()
        buffer_size = len(data)
        for sample in data.get(
            shuffle=True, batch_size=buffer_size // self.train_critic_iters
        ):
            self.critic_optim.zero_grad()
            loss = self._compute_critic_loss(sample)
            loss.backward()
            self.critic_optim.step()
        self.critic.eval()

    def _finalize_trajectory(self, data: TrajectoryBuffer) -> PPOSample:
        """
        This aggregates the features, actions, log probabilities, returns, and advantages
        from a fixed trajectory.

        This method expects the following at time `t`:
        - data.features[t]: visual features of the environment computed at time `t`
        - data.actions[t]: the env action taken in the environment at time `t`
        - data.log_probabilities[t]: the log probability of selecting `data.actions[t]`
        - data.values[t]: the value assigned to `data.features[t]`
        - data.rewards[t]: the reward given by the environment about the action take at time `t - 1`
        - data.focus[t]: the focus/ROI coordinates at time `t`
        - data.focus_logp[t]: the log probability of the focus coordinates
        """

        # Cannot use the last values here since we don't have the associated reward yet
        features = torch.stack(list(data.features_buffer)[:-1])
        actions = torch.stack(list(data.actions_buffer)[:-1])
        # Sum per-component log probs to get joint log prob per timestep
        raw_logp = torch.stack(list(data.log_probs_buffer)[:-1])
        log_probabilities = raw_logp.sum(dim=-1) if raw_logp.dim() > 1 else raw_logp

        # Focus data (stored separately)
        focus_actions = torch.stack(list(data.focus_buffer)[:-1])
        raw_focus_logp = torch.stack(list(data.focus_logp_buffer)[:-1])
        focus_log_probabilities = (
            raw_focus_logp.sum(dim=-1) if raw_focus_logp.dim() > 1 else raw_focus_logp
        )

        # Cannot use the first reward value since we no longer have the associated feature
        # The reward for a_t is at r_{t+1}
        env_rewards = torch.tensor(list(data.rewards_buffer)[1:])
        intrinsic_rewards = torch.tensor(list(data.intrinsic_rewards_buffer)[1:])
        rewards = env_rewards + intrinsic_rewards
        # Need all values since the final one is used to estimate future reward
        values = torch.tensor(list(data.values_buffer))

        deltas = rewards + (self.discount_factor * values[1:]) - values[:-1]
        advantages = torch.tensor(
            discount_cumsum(
                deltas.numpy(), self.discount_factor * self.gae_discount_factor
            ).copy(),
            dtype=torch.float,
        )
        returns = torch.tensor(
            discount_cumsum(rewards.numpy(), self.discount_factor).copy(),
            dtype=torch.float,
        ).unsqueeze(1)

        return PPOSample(
            features=features,
            actions=actions,
            returns=returns,
            advantages=advantages,
            log_probabilities=log_probabilities,
            focus_actions=focus_actions,
            focus_log_probabilities=focus_log_probabilities,
        )

    def update(self, trajectory: TrajectoryBuffer) -> None:
        """Updates the actor and critic models given the a trajectory"""
        data = self._finalize_trajectory(trajectory)
        self._update_actor(data)
        self._update_critic(data)
