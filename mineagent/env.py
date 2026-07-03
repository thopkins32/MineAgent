import asyncio
from dataclasses import dataclass
from typing import Any
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .client import (
    AsyncMinecraftClient,
    ConnectionConfig,
    ActionMessage,
    NUM_KEYS,
    held_state_diff,
    make_action_space,
)


@dataclass
class MinecraftEnvConfig:
    """Configuration for the Minecraft environment."""

    frame_width: int = 320
    frame_height: int = 240


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

        # Held-state register: the agent emits absolute desired state each
        # step; the env diffs this against the register to produce the
        # PRESS/RELEASE edges that go on the wire. Java mirrors this state.
        self._held_keys = np.zeros(NUM_KEYS, dtype=np.int8)
        self._held_buttons = np.zeros(3, dtype=np.int8)

        self._step_count = 0
        self._last_reward: float = 0.0
        self._minecraft_process: asyncio.subprocess.Process | None = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run_async(self, coro):
        loop = self._ensure_loop()
        return loop.run_until_complete(coro)

    async def _launch_minecraft_process(self) -> None:
        """Launch minecraft in a Python subprocess.

        Stdout/stderr are discarded because the Forge mod already writes full
        Log4j logs to ``forge/run/logs/latest.log`` (and ``debug.log``);
        mirroring them here would just duplicate that.
        """
        if (
            self._minecraft_process is not None
            and self._minecraft_process.returncode is None
        ):
            return  # already running

        self._minecraft_process = await asyncio.create_subprocess_exec(
            "gradle",
            "runClient",
            cwd=Path.cwd() / "forge",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self._run_async(self._launch_minecraft_process())

        if not self._client.connected:
            self._run_async(self._client.connect())

        # Reset Java's held state and zero the local register in lockstep.
        self._held_keys = np.zeros(NUM_KEYS, dtype=np.int8)
        self._held_buttons = np.zeros(3, dtype=np.int8)
        self._run_async(self._client.send_action(ActionMessage.reset()))

        obs = self._run_async(self._client.receive_observation())
        frame = obs.frame
        self._last_reward = obs.reward

        self._step_count = 0

        return frame, {"step_count": self._step_count, "reward": self._last_reward}

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        message = self._action_to_message(action)
        self._run_async(self._client.send_action(message))
        obs = self._run_async(self._client.receive_observation())
        frame = obs.frame
        reward = obs.reward

        self._last_reward = reward
        self._step_count += 1

        terminated = False
        info = {
            "step_count": self._step_count,
            "reward": reward,
        }

        return frame, reward, terminated, False, info

    def _action_to_message(self, action: dict[str, np.ndarray]) -> ActionMessage:
        """Translate an absolute-state action dict into an event-based wire
        message by diffing against the held-state register, then advance the
        register to the new desired state."""
        new_keys = np.asarray(action["keys"], dtype=np.int8).ravel()
        new_buttons = np.asarray(action["mouse_buttons"], dtype=np.int8).ravel()

        key_press, key_release, button_press, button_release = held_state_diff(
            self._held_keys, new_keys, self._held_buttons, new_buttons
        )

        mouse_dx = float(action["mouse_dx"])
        mouse_dy = float(action["mouse_dy"])
        scroll = float(action["scroll_delta"])

        message = ActionMessage(
            key_press=list(key_press),
            key_release=list(key_release),
            has_mouse=(mouse_dx != 0.0 or mouse_dy != 0.0),
            mouse_dx=mouse_dx,
            mouse_dy=mouse_dy,
            has_buttons=(button_press != 0 or button_release != 0),
            button_press=button_press,
            button_release=button_release,
            has_scroll=(scroll != 0.0),
            scroll=scroll,
        )

        # Advance the register to the agent's desired held state.
        self._held_keys = new_keys
        self._held_buttons = new_buttons

        return message

    def render(self, mode: str = "rgb_array") -> np.ndarray | None:
        raise NotImplementedError("Rendering is not supported.")

    async def _stop_minecraft_process(self) -> None:
        if (
            self._minecraft_process is None
            or self._minecraft_process.returncode is not None
        ):
            return  # already stopped

        self._minecraft_process.terminate()
        try:
            await asyncio.wait_for(self._minecraft_process.wait(), timeout=60)
        except asyncio.TimeoutError:
            self._minecraft_process.kill()
            await self._minecraft_process.wait()

    def close(self):
        if self._client.connected:
            self._run_async(self._client.disconnect())
        self._run_async(self._stop_minecraft_process())
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
