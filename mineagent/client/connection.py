import struct
import asyncio
import logging
from dataclasses import dataclass

import numpy as np

from .protocol import Observation, RawInput, parse_observation


@dataclass
class ConnectionConfig:
    """Configuration for Minecraft Forge mod connection."""

    observation_socket: str = "/tmp/mineagent_observation.sock"
    action_socket: str = "/tmp/mineagent_action.sock"
    frame_width: int = 320
    frame_height: int = 240
    timeout: float = 150.0


class AsyncMinecraftClient:
    """
    Async client for communicating with the Minecraft Forge mod via Unix domain sockets.
    """

    def __init__(self, config: ConnectionConfig | None = None):
        self.config = config or ConnectionConfig()
        self._observation_reader: asyncio.StreamReader | None = None
        self._observation_writer: asyncio.StreamWriter | None = None
        self._action_reader: asyncio.StreamReader | None = None
        self._action_writer: asyncio.StreamWriter | None = None
        self._connected: bool = False
        self._logger = logging.getLogger(__name__)

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Open both Unix sockets, waiting for the Forge mod to start them.

        The mod creates the socket files asynchronously with this process, so
        the first attempts fail with FileNotFoundError (no socket yet) or
        ConnectionRefusedError (file exists but not listening yet). Retry
        until self.config.timeout elapses.

        Keep both the reader and writer for each socket. open_unix_connection
        returns a (reader, writer) pair sharing one transport; discarding
        either half can let the shared transport be reaped, tearing down the
        socket while the other half is still in use.
        """
        (
            self._observation_reader,
            self._observation_writer,
        ) = await self._open_unix_connection(self.config.observation_socket)
        self._action_reader, self._action_writer = await self._open_unix_connection(
            self.config.action_socket
        )
        self._connected = True

    async def _open_unix_connection(
        self, path: str
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        loop = asyncio.get_running_loop()
        end = loop.time() + self.config.timeout
        while True:
            try:
                return await asyncio.open_unix_connection(path)
            except (FileNotFoundError, ConnectionRefusedError):
                if loop.time() >= end:
                    raise TimeoutError(f"Timed out waiting for socket {path}")
                await asyncio.sleep(0.1)

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
        if self._observation_writer:
            self._observation_writer.close()
            await self._observation_writer.wait_closed()
            self._observation_writer = None
        self._observation_reader = None
        self._action_reader = None

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

    async def receive_observation(self) -> Observation:
        """
        Receive an observation from the Minecraft Forge mod.

        Returns
        -------
        Observation
            Parsed observation with reward and frame.

        Raises
        ------
        ConnectionError
            If not connected, the connection drops, or the mod sends a
            frame whose size does not match the configured dimensions.
        """
        if not self._connected or not self._observation_reader:
            raise ConnectionError("Not connected to Minecraft Forge mod")

        try:
            header = await self._observation_reader.readexactly(12)
        except asyncio.IncompleteReadError:
            self._connected = False
            raise

        reward = struct.unpack(">d", header[0:8])[0]
        frame_length = struct.unpack(">I", header[8:12])[0]

        expected_length = self.config.frame_height * self.config.frame_width * 3

        if frame_length == 0:
            return Observation(
                reward=reward,
                frame=np.zeros(
                    (self.config.frame_height, self.config.frame_width, 3),
                    dtype=np.uint8,
                ),
            )

        if frame_length != expected_length:
            self._connected = False
            raise ConnectionError(
                f"Frame size mismatch: mod sent {frame_length} bytes "
                f"but expected {expected_length} "
                f"(configured {self.config.frame_width}x"
                f"{self.config.frame_height}x3). "
                f"Check that the Forge mod window size matches "
                f"ConnectionConfig.frame_width/frame_height."
            )

        try:
            frame_data = await self._observation_reader.readexactly(frame_length)
        except asyncio.IncompleteReadError as e:
            self._connected = False
            raise ConnectionError(
                f"Connection lost while reading frame data: "
                f"got {len(e.partial)} of {frame_length} bytes"
            ) from e

        try:
            return parse_observation(
                header,
                frame_data,
                (self.config.frame_height, self.config.frame_width),
            )
        except OSError as e:
            self._connected = False
            raise ConnectionError(f"Failed to receive observation: {e}") from e
