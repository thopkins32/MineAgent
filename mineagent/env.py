import asyncio
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .client import (
    AsyncMinecraftClient,
    ConnectionConfig,
    RawInput,
    action_to_raw_input,
    make_action_space,
)


@dataclass
class MinecraftEnvConfig:
    """Configuration for the Minecraft environment."""

    frame_width: int = 320
    frame_height: int = 240
    max_steps: int = 10_000


class MinecraftEnv(gym.Env):
    """
    Gymnasium environment for Minecraft using the Forge mod.

    This environment provides a Gymnasium-compatible interface for interacting
    with Minecraft through the custom Forge mod.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_config: MinecraftEnvConfig | None = None,
        connection_config: ConnectionConfig | None = None,
    ):
        super().__init__()

        self.env_config = env_config or MinecraftEnvConfig()
        self.connection_config = connection_config or ConnectionConfig()

        self.connection_config.frame_width = self.env_config.frame_width
        self.connection_config.frame_height = self.env_config.frame_height

        self._client = AsyncMinecraftClient(self.connection_config)
        self._loop: asyncio.AbstractEventLoop | None = None

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env_config.frame_height, self.env_config.frame_width, 3),
            dtype=np.uint8,
        )

        self.action_space = make_action_space()

        self._step_count = 0
        self._last_reward: float = 0.0

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run_async(self, coro):
        loop = self._ensure_loop()
        return loop.run_until_complete(coro)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        if not self._client.connected:
            if not self._run_async(self._client.connect()):
                raise RuntimeError("Failed to connect to Minecraft Forge mod")

        self._run_async(self._client.send_action(RawInput.release_all()))

        obs = self._run_async(self._client.receive_observation())
        frame = obs.frame
        self._last_reward = obs.reward

        self._step_count = 0

        return frame, {"step_count": self._step_count, "reward": self._last_reward}

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw_input = action_to_raw_input(action)
        self._run_async(self._client.send_action(raw_input))

        obs = self._run_async(self._client.receive_observation())
        frame = obs.frame
        reward = obs.reward

        self._last_reward = reward
        self._step_count += 1

        terminated = False
        truncated = self._step_count >= self.env_config.max_steps

        info = {
            "step_count": self._step_count,
            "reward": reward,
        }

        return frame, reward, terminated, truncated, info

    def render(self, mode: str = "rgb_array") -> np.ndarray | None:
        raise NotImplementedError("Rendering is not supported.")

    def close(self):
        if self._client.connected:
            self._run_async(self._client.disconnect())
        if self._loop and not self._loop.is_closed():
            self._loop.close()
            self._loop = None


def create_minecraft_env(
    env_config: MinecraftEnvConfig | None = None,
    connection_config: ConnectionConfig | None = None,
) -> MinecraftEnv:
    """
    Factory function to create a Minecraft environment.

    Parameters
    ----------
    env_config : MinecraftEnvConfig | None
        Environment configuration
    connection_config : ConnectionConfig | None
        Connection configuration for the Minecraft mod

    Returns
    -------
    MinecraftEnv
        Configured Minecraft environment
    """
    return MinecraftEnv(env_config=env_config, connection_config=connection_config)
