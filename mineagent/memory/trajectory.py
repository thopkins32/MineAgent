from collections import deque

import torch


class TrajectoryBuffer:
    """
    Used to store a single trajectory.

    A single item in a trajectory contains the following information:
    - The visual features computed from the observation of the environment
    - The action taken given the features computed
    - The reward given by the environment for the PREVIOUS action
    - The value assigned to the features computed
    - The log probability of sampling the action taken

    Note: a single item is not complete enough information to learn from.
    At least two consecutive items in the trajectory are necessary for learning.
    This is because the next observation and reward are not available until the next step is stored.
    """

    def __init__(self, max_buffer_size: int):
        self.max_buffer_size = max_buffer_size

        self.features_buffer: deque[torch.Tensor] = deque([], maxlen=max_buffer_size)
        self.actions_buffer: deque[torch.Tensor] = deque([], maxlen=max_buffer_size)
        self.rewards_buffer: deque[float] = deque([], maxlen=max_buffer_size)
        self.intrinsic_rewards_buffer: deque[float] = deque([], maxlen=max_buffer_size)
        self.values_buffer: deque[float] = deque([], maxlen=max_buffer_size)
        self.log_probs_buffer: deque[torch.Tensor] = deque([], maxlen=max_buffer_size)
        self.focus_buffer: deque[torch.Tensor] = deque([], maxlen=max_buffer_size)
        self.focus_logp_buffer: deque[torch.Tensor] = deque([], maxlen=max_buffer_size)

    def __len__(self):
        return len(self.features_buffer)

    def store(
        self,
        visual_features: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        intrinsic_reward: float,
        value: float,
        log_prob: torch.Tensor,
        focus: torch.Tensor | None = None,
        focus_logp: torch.Tensor | None = None,
    ) -> None:
        """
        Append a single time-step to the trajectory.

        Parameters
        ----------
        visual_features : torch.Tensor
            Features computed by visual perception from the observation of the environment
        action : torch.Tensor
            Environment action tensor (keys, mouse, buttons, scroll -- no focus)
        reward : float
            Reward value from the environment for the previous action
        intrinsic_reward : float
            Reward value from the Intrinsic Curiosity Module (ICM)
        value : float
            Value assigned to the observation by the agent
        log_prob : torch.Tensor
            Log probability of selecting each environment sub-action
        focus : torch.Tensor | None
            Focus/ROI coordinates (2-dim), stored separately from env action
        focus_logp : torch.Tensor | None
            Log probability of the focus coordinates
        """
        self.features_buffer.append(visual_features)
        self.actions_buffer.append(action)
        self.rewards_buffer.append(reward)
        self.intrinsic_rewards_buffer.append(intrinsic_reward)
        self.values_buffer.append(value)
        self.log_probs_buffer.append(log_prob)
        if focus is not None:
            self.focus_buffer.append(focus)
        if focus_logp is not None:
            self.focus_logp_buffer.append(focus_logp)
