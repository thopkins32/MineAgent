from datetime import datetime

import numpy as np
import torch
from torchvision.transforms.functional import center_crop, crop  # type: ignore

from ..perception.visual import VisualPerception
from ..affector.affector import LinearAffector
from ..reasoning.critic import LinearCritic
from ..reasoning.dynamics import InverseDynamics, ForwardDynamics
from ..memory.trajectory import TrajectoryBuffer
from ..learning.icm import ICM
from ..learning.ppo import PPO
from ..config import AgentConfig
from ..monitoring.event import Action
from ..utils import sample_action
from ..monitoring.event_bus import get_event_bus
from ..client.protocol import (
    NUM_KEYS,
    MOUSE_DX_RANGE,
    MOUSE_DY_RANGE,
    SCROLL_RANGE,
)


class AgentV1:
    """
    Version 1 of this agent will stand in place and observe the environment. It will be allowed to move its head only to look around.
    So, it will have a visual perception module to start with and a simple learning algorithm to guide where its looking.
    This is to test the following:
    - Can the agent learn to focus its attention on new information?
    - How fast is the visual perception module? Does it need to be faster?
    """

    EMBED_DIM = 32 + 32

    def __init__(
        self,
        config: AgentConfig,
    ) -> None:
        self.vision = VisualPerception(out_channels=32)
        self.affector = LinearAffector(self.EMBED_DIM)
        self.critic = LinearCritic(self.EMBED_DIM)
        self.memory = TrajectoryBuffer(config.max_buffer_size)
        self.inverse_dynamics = InverseDynamics(self.EMBED_DIM)
        self.forward_dynamics = ForwardDynamics(self.EMBED_DIM)
        self.ppo = PPO(self.affector, self.critic, config.ppo)
        self.icm = ICM(self.forward_dynamics, self.inverse_dynamics, config.icm)
        self.config = config
        self.monitor_actions = False

        # region of interest initialization
        self.roi_action: torch.Tensor | None = None
        self.prev_visual_features: torch.Tensor = torch.zeros(
            (1, self.EMBED_DIM), dtype=torch.float
        )
        self.event_bus = get_event_bus()

    def _transform_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if self.roi_action is None:
            roi_obs = center_crop(obs, list(self.config.roi_shape))
        else:
            roi_obs = crop(
                obs,
                int(self.roi_action[:, 0]),
                int(self.roi_action[:, 1]),
                self.config.roi_shape[0],
                self.config.roi_shape[1],
            )
        return roi_obs

    def start_monitoring(self) -> None:
        self.vision.start_monitoring()
        self.affector.start_monitoring()
        self.critic.start_monitoring()
        self.inverse_dynamics.start_monitoring()
        self.forward_dynamics.start_monitoring()
        self.monitor_actions = True

    def stop_monitoring(self) -> None:
        self.vision.stop_monitoring()
        self.affector.stop_monitoring()
        self.critic.stop_monitoring()
        self.inverse_dynamics.stop_monitoring()
        self.forward_dynamics.stop_monitoring()
        self.monitor_actions = False

    @staticmethod
    def action_tensor_to_env(action: torch.Tensor) -> dict[str, np.ndarray]:
        """
        Convert the flat action tensor (without the trailing 2 focus dims)
        into the Dict-space action expected by MinecraftEnv.
        """
        a = action.squeeze(0)
        keys = (a[:NUM_KEYS] > 0.5).to(torch.int8).numpy()
        col = NUM_KEYS
        mouse_dx = np.float32(np.clip(a[col].item(), *MOUSE_DX_RANGE))
        col += 1
        mouse_dy = np.float32(np.clip(a[col].item(), *MOUSE_DY_RANGE))
        col += 1
        scroll = np.float32(np.clip(a[col].item(), *SCROLL_RANGE))
        col += 1
        mouse_buttons = (a[col : col + 3] > 0.5).to(torch.int8).numpy()
        return {
            "keys": keys,
            "mouse_dx": mouse_dx,
            "mouse_dy": mouse_dy,
            "mouse_buttons": mouse_buttons,
            "scroll_delta": scroll,
        }

    def act(self, obs: torch.Tensor, reward: float = 0.0) -> dict[str, np.ndarray]:
        roi_obs = self._transform_observation(obs)
        with torch.no_grad():
            visual_features = self.vision(obs, roi_obs)
            affector_output = self.affector(visual_features)
            value = self.critic(visual_features)
        action, logp_action = sample_action(affector_output)

        with torch.no_grad():
            intrinsic_reward = self.icm.intrinsic_reward(
                self.prev_visual_features, action, visual_features
            )

        self.memory.store(
            visual_features, action, reward, intrinsic_reward, value, logp_action
        )

        if len(self.memory) == self.config.max_buffer_size:
            self.ppo.update(self.memory)
            self.icm.update(self.memory)

        self.prev_visual_features = visual_features
        self.roi_action = action[:, -2:].round().long()

        env_action = self.action_tensor_to_env(action[:, :-2])

        if self.monitor_actions:
            self.event_bus.publish(
                Action(
                    timestamp=datetime.now(),
                    visual_features=visual_features,
                    action_distribution=affector_output,
                    action=action,
                    logp_action=logp_action,
                    value=value,
                    region_of_interest=roi_obs,
                    intrinsic_reward=intrinsic_reward,
                )
            )
        return env_action
