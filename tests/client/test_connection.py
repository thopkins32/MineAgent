import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mineagent.client.connection import AsyncMinecraftClient, ConnectionConfig
from mineagent.client.protocol import RawInput

FRAME_WIDTH = 2
FRAME_HEIGHT = 2
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3


@pytest.fixture
def config():
    return ConnectionConfig(
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        max_retries=3,
        retry_delay=0.0,
    )


@pytest.fixture
def client(config):
    return AsyncMinecraftClient(config)


def _make_mock_reader():
    reader = AsyncMock(spec=asyncio.StreamReader)
    return reader


def _make_mock_writer():
    writer = MagicMock(spec=asyncio.StreamWriter)
    writer.drain = AsyncMock()
    writer.wait_closed = AsyncMock()
    return writer


def _build_observation_bytes(reward: float, frame: bytes) -> tuple[bytes, bytes]:
    """Build the header and frame bytes matching the wire protocol."""
    header = struct.pack(">d", reward) + struct.pack(">I", len(frame))
    return header, frame


async def _connect_client(client: AsyncMinecraftClient):
    """Patch open_unix_connection and connect, returning the mock reader/writer."""
    reader = _make_mock_reader()
    writer = _make_mock_writer()

    async def fake_open(path):
        if path == client.config.observation_socket:
            return (reader, MagicMock())
        return (MagicMock(), writer)

    with patch("mineagent.client.connection.asyncio.open_unix_connection", side_effect=fake_open):
        result = await client.connect()

    assert result is True
    return reader, writer


@pytest.mark.asyncio
async def test_connect_success(client):
    reader = _make_mock_reader()
    writer = _make_mock_writer()

    async def fake_open(path):
        if path == client.config.observation_socket:
            return (reader, MagicMock())
        return (MagicMock(), writer)

    with patch("mineagent.client.connection.asyncio.open_unix_connection", side_effect=fake_open):
        result = await client.connect()

    assert result is True
    assert client.connected is True


@pytest.mark.asyncio
async def test_connect_failure_retries(client):
    call_count = 0

    async def fake_open(path):
        nonlocal call_count
        call_count += 1
        raise OSError("Connection refused")

    with patch("mineagent.client.connection.asyncio.open_unix_connection", side_effect=fake_open):
        result = await client.connect()

    assert result is False
    assert client.connected is False
    assert call_count == client.config.max_retries


@pytest.mark.asyncio
async def test_disconnect(client):
    reader, writer = await _connect_client(client)
    assert client.connected is True

    await client.disconnect()

    assert client.connected is False
    writer.close.assert_called_once()
    writer.wait_closed.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_action(client):
    raw_input = RawInput(key_codes=[87], mouse_dx=1.0, mouse_dy=-1.0)
    expected_bytes = raw_input.to_bytes()

    _, writer = await _connect_client(client)

    result = await client.send_action(raw_input)

    assert result is True
    writer.write.assert_called_once_with(expected_bytes)
    writer.drain.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_action_not_connected(client):
    result = await client.send_action(RawInput())
    assert result is False


@pytest.mark.asyncio
async def test_receive_observation(client):
    reward = 1.5
    frame_bytes = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8).tobytes()
    header, frame_data = _build_observation_bytes(reward, frame_bytes)

    reader, _ = await _connect_client(client)
    reader.readexactly = AsyncMock(side_effect=[header, frame_data])

    obs = await client.receive_observation()

    assert obs is not None
    assert obs.reward == reward
    assert obs.frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)


@pytest.mark.asyncio
async def test_receive_observation_incomplete(client):
    reader, _ = await _connect_client(client)
    reader.readexactly = AsyncMock(
        side_effect=asyncio.IncompleteReadError(partial=b"", expected=12)
    )

    obs = await client.receive_observation()

    assert obs is None


@pytest.mark.asyncio
async def test_receive_observation_not_connected(client):
    obs = await client.receive_observation()
    assert obs is None
