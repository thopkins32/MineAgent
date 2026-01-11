"""
Minecraft Forge Client for Gymnasium API

This module provides a client implementation that connects to the Minecraft Forge mod
and exposes a Gymnasium-compatible interface for reinforcement learning experiments.
The current implementation focuses on reading frame data only.
"""

import socket
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any
import time
import logging
from dataclasses import dataclass
from PIL import Image

from .config import Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@dataclass
class ConnectionConfig:
    """Configuration for Minecraft Forge mod connection"""

    command_port: str = "/tmp/mineagent_receive.sock"
    data_port: str = "/tmp/mineagent_send.sock"
    width: int = 320
    height: int = 240
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


class MinecraftForgeClient:
    """
    Low-level client for communicating with the Minecraft Forge mod
    Uses TCP for commands and UDP for frame data (asynchronous)
    """

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.command_socket: socket.socket | None = None
        self.data_socket: socket.socket | None = None
        self.connected: bool = False
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Delta encoding state
        self.current_frame: np.ndarray | None = None

    def connect(self) -> bool:
        """
        Establish connection to the Minecraft Forge mod

        Returns
        -------
        bool
            True if connection successful, False otherwise
        """
        for attempt in range(self.config.max_retries):
            try:
                # Connect Unix domain socket for commands
                self.command_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.command_socket.settimeout(self.config.timeout)
                self.command_socket.connect(self.config.command_port)

                # Create Unix domain socket for frame data with performance optimizations
                self.data_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.data_socket.settimeout(self.config.timeout)

                # Performance optimizations for large data transfers
                # Set large receive buffer (1MB) for better throughput
                self.data_socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024
                )
                self.data_socket.connect(self.config.data_port)

                self.connected = True
                self.logger.info(
                    f"Connected to Minecraft Forge mod - Command: {self.config.command_port}, Data: {self.config.data_port}"
                )
                return True
            except (socket.error, ConnectionRefusedError) as e:
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                self._cleanup_sockets()
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    self.logger.error("Failed to connect after all retries")
                    return False

        return False

    def disconnect(self):
        """Disconnect from the Minecraft Forge mod"""
        self._cleanup_sockets()
        self.connected = False
        self.logger.info("Disconnected from Minecraft Forge mod")

    def _cleanup_sockets(self):
        """Clean up both sockets"""
        if self.command_socket:
            try:
                self.command_socket.close()
            finally:
                self.command_socket = None

        if self.data_socket:
            try:
                self.data_socket.close()
            finally:
                self.data_socket = None

    def send_command(self, command: str) -> bool:
        """
        Send a command to the Minecraft Forge mod via TCP

        Parameters
        ----------
        command : str
            Command to send to the mod

        Returns
        -------
        bool
            True if command sent successfully, False otherwise
        """
        if not self.connected or not self.command_socket:
            self.logger.error("Not connected to Minecraft Forge mod")
            return False

        try:
            self.command_socket.send((command + "\n").encode())
            return True
        except socket.error as e:
            self.logger.error(f"Failed to send command: {e}")
            self.connected = False
            return False

    def receive_frame_data(self) -> np.ndarray | None:
        """
        Receive frame data from the Minecraft Forge mod via Unix domain socket

        Returns
        -------
        np.ndarray | None
            Frame data as numpy array if successful, None otherwise
        """
        if not self.connected or not self.data_socket:
            self.logger.error("Not connected to Minecraft Forge mod")
            return None

        try:
            # Fastest approach: receive header + data in one shot if possible
            # First get just the header to know the total size
            header_data = self.data_socket.recv(8, socket.MSG_WAITALL)
            if len(header_data) != 8:
                self.logger.error(
                    "Failed to receive complete protocol header, got %d bytes",
                    len(header_data),
                )
                return None
            else:
                self.logger.info("Received header: %s", len(header_data))

            # Parse header
            reward = int.from_bytes(header_data[0:4], byteorder="big", signed=True)
            data_length = int.from_bytes(
                header_data[4:8], byteorder="big", signed=False
            )
            self.logger.info(
                "Received reward: %d, data length: %d", reward, data_length
            )

            if data_length == 0:
                return None

            # Receive all frame data in single call with MSG_WAITALL
            frame_data = self.data_socket.recv(data_length, socket.MSG_WAITALL)
            if len(frame_data) != data_length:
                self.logger.error(
                    "Failed to receive complete frame data, got %d bytes, expected %d bytes",
                    len(frame_data),
                    data_length,
                )
                return None

            # Parse frame data using delta encoding protocol
            return self._parse_frame_data(frame_data)

        except socket.timeout:
            # Timeout is expected if no frames are being sent
            self.logger.warning("Timeout waiting for frame data")
            return None
        except socket.error as e:
            self.logger.error(f"Failed to receive frame data: {e}")
            self.connected = False
            return None

    def _parse_frame_data(self, frame_data: bytes) -> np.ndarray | None:
        """
        Parse frame data using delta encoding protocol

        Parameters
        ----------
        frame_data : bytes
            Raw frame data from socket

        Returns
        -------
        np.ndarray | None
            Parsed frame as numpy array if successful, None otherwise
        """
        return np.flipud(
            np.frombuffer(frame_data, dtype=np.uint8).reshape(
                self.config.height, self.config.width, 3
            )
        )


class MinecraftEnv(gym.Env[np.ndarray, np.int64]):
    """
    Gymnasium environment for Minecraft using the Forge mod

    This environment provides a Gymnasium-compatible interface for interacting
    with Minecraft through the custom Forge mod. Currently supports reading
    frame data only.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: Config | None = None,
        connection_config: ConnectionConfig | None = None,
    ):
        super().__init__()

        self.config = config or Config()
        self.connection_config = connection_config or ConnectionConfig()
        self.client = MinecraftForgeClient(self.connection_config)

        # Set up observation space (RGB image)
        height, width = self.config.engine.image_size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )

        # For now, we don't support actions, but we need to define the space
        # This will be expanded when action support is added
        self.action_space = spaces.Discrete(1)  # No-op action

        self.current_frame = None
        self.step_count = 0
        self.logger = logging.getLogger(__name__)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment

        Parameters
        ----------
        seed : int | None
            Random seed for reproducibility
        options : dict[str, Any] | None
            Additional options for reset

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Initial observation and info dictionary
        """
        super().reset(seed=seed, options=options)

        # Connect to Minecraft if not already connected
        if not self.client.connected:
            if not self.client.connect():
                raise RuntimeError("Failed to connect to Minecraft Forge mod")

        # Send reset command to mod (for future use)
        self.client.send_command("RESET")

        # Get initial frame
        self.current_frame = self.client.receive_frame_data()
        if self.current_frame is None:
            self.logger.warning("No frame data received")
            # Return a black frame if we can't get data
            height, width = self.connection_config.width, self.connection_config.height
            self.current_frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            self.logger.info(f"Received frame data: {self.current_frame.shape}")

        self.step_count = 0

        return self.current_frame, {"step_count": self.step_count}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment

        Parameters
        ----------
        action
            Action to take (currently ignored)

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
            Observation, reward, terminated, truncated, info
        """
        # For now, we ignore the action since we're only reading frame data

        # Get new frame data
        new_frame = self.client.receive_frame_data()
        if new_frame is not None:
            self.current_frame = self._resize_frame(new_frame)

        # Ensure current_frame is never None
        if self.current_frame is None:
            height, width = self.config.engine.image_size
            self.current_frame = np.zeros((height, width, 3), dtype=np.uint8)

        self.step_count += 1

        # Placeholder values for reward and termination
        reward = 0.0
        terminated = False
        truncated = self.step_count >= self.config.engine.max_steps

        info = {"step_count": self.step_count, "frame_received": new_frame is not None}

        return self.current_frame, reward, terminated, truncated, info

    def render(self, mode: str = "rgb_array") -> np.ndarray | None:
        """
        Render the environment

        Parameters
        ----------
        mode : str
            Render mode (only "rgb_array" supported)

        Returns
        -------
        np.ndarray | None
            Rendered frame as numpy array
        """
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")

        return self.current_frame

    def close(self):
        """Close the environment and disconnect from Minecraft"""
        self.client.disconnect()

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to match expected dimensions

        Parameters
        ----------
        frame : np.ndarray
            Input frame

        Returns
        -------
        np.ndarray
            Resized frame
        """
        target_height, target_width = self.config.engine.image_size

        if frame.shape[:2] != (target_height, target_width):
            # Use PIL for high-quality resizing
            pil_image = Image.fromarray(frame)
            try:
                # Try new PIL API first
                from PIL.Image import Resampling

                pil_image = pil_image.resize(
                    (target_width, target_height), Resampling.LANCZOS
                )
            except (ImportError, AttributeError):
                # Fall back to old PIL API using numeric constant (1 = LANCZOS)
                pil_image = pil_image.resize((target_width, target_height), 1)
            frame = np.array(pil_image)

        return frame


def create_minecraft_env(
    config: Config | None = None, connection_config: ConnectionConfig | None = None
) -> MinecraftEnv:
    """
    Factory function to create a Minecraft environment

    Parameters
    ----------
    config : Config | None
        MineAgent configuration object
    connection_config : ConnectionConfig | None
        Connection configuration for the Minecraft mod

    Returns
    -------
    MinecraftEnv
        Configured Minecraft environment
    """
    return MinecraftEnv(config=config, connection_config=connection_config)
