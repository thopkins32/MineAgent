import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from .protocol import Observation, RawInput, parse_observation


@dataclass
class ConnectionConfig:
    """Configuration for Minecraft Forge mod connection."""

    observation_socket: str = "/tmp/mineagent_observation.sock"
    action_socket: str = "/tmp/mineagent_action.sock"
    frame_width: int = 320
    frame_height: int = 240
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


class AsyncMinecraftClient:
    """
    Async client for communicating with the Minecraft Forge mod via Unix domain sockets.
    """

    def __init__(self, config: ConnectionConfig | None = None):
        self.config = config or ConnectionConfig()
        self._observation_reader: asyncio.StreamReader | None = None
        self._action_writer: asyncio.StreamWriter | None = None
        self._connected: bool = False
        self._logger = logging.getLogger(__name__)

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Establish connection to the Minecraft Forge mod."""
        for attempt in range(self.config.max_retries):
            try:
                self._observation_reader, _ = await asyncio.open_unix_connection(
                    self.config.observation_socket
                )
                _, self._action_writer = await asyncio.open_unix_connection(
                    self.config.action_socket
                )
                self._connected = True
                self._logger.info(
                    "Connected to Minecraft Forge mod - Observation: %s, Action: %s",
                    self.config.observation_socket,
                    self.config.action_socket,
                )
                return True
            except OSError as e:
                self._logger.warning("Connection attempt %d failed: %s", attempt + 1, e)
                await self._cleanup()
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    self._logger.error("Failed to connect after %d attempts", self.config.max_retries)
                    return False
        return False

    async def disconnect(self) -> None:
        """Disconnect from the Minecraft Forge mod."""
        await self._cleanup()
        self._connected = False
        self._logger.info("Disconnected from Minecraft Forge mod")

    async def _cleanup(self) -> None:
        """Clean up sockets."""
        if self._action_writer:
            self._action_writer.close()
            await self._action_writer.wait_closed()
            self._action_writer = None
        self._observation_reader = None

    async def send_action(self, raw_input: RawInput) -> bool:
        """
        Send a raw input action to the Minecraft Forge mod.

        Parameters
        ----------
        raw_input : RawInput
            The input to send

        Returns
        -------
        bool
            True if sent successfully, False otherwise
        """
        if not self._connected or not self._action_writer:
            self._logger.error("Not connected to Minecraft Forge mod")
            return False

        try:
            data = raw_input.to_bytes()
            self._action_writer.write(data)
            await self._action_writer.drain()
            return True
        except OSError as e:
            self._logger.error("Failed to send action: %s", e)
            self._connected = False
            return False

    async def receive_observation(self) -> Observation | None:
        """
        Receive an observation from the Minecraft Forge mod.

        Returns
        -------
        Observation | None
            The observation if received successfully, None otherwise
        """
        if not self._connected or not self._observation_reader:
            self._logger.error("Not connected to Minecraft Forge mod")
            return None

        try:
            header = await self._observation_reader.readexactly(12)
            reward = struct.unpack(">d", header[0:8])[0]
            frame_length = struct.unpack(">I", header[8:12])[0]

            if frame_length == 0:
                return Observation(reward=reward, frame=np.zeros((self.config.frame_height, self.config.frame_width, 3), dtype=np.uint8))

            frame_data = await self._observation_reader.readexactly(frame_length)

            return parse_observation(
                header,
                frame_data,
                (self.config.frame_height, self.config.frame_width),
            )
        except asyncio.IncompleteReadError:
            self._logger.warning("Incomplete observation received")
            return None
        except OSError as e:
            self._logger.error("Failed to receive observation: %s", e)
            self._connected = False
            return None


import struct
import numpy as np
