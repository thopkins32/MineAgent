import struct

import numpy as np
import pytest

from mineagent.client.protocol import GLFW, RawInput, parse_observation


# --- to_bytes serialization ---


def test_empty_raw_input_to_bytes():
    data = RawInput().to_bytes()

    # 1 (key count) + 4+4 (mouse dx/dy) + 1 (buttons) + 4 (scroll) + 2 (text len) = 16
    assert len(data) == 16

    assert data[0] == 0
    assert struct.unpack(">f", data[1:5])[0] == 0.0
    assert struct.unpack(">f", data[5:9])[0] == 0.0
    assert data[9] == 0
    assert struct.unpack(">f", data[10:14])[0] == 0.0
    assert struct.unpack(">H", data[14:16])[0] == 0


def test_to_bytes_with_key_codes():
    raw = RawInput(key_codes=[GLFW.KEY_W, GLFW.KEY_SPACE])
    data = raw.to_bytes()

    assert data[0] == 2
    assert struct.unpack(">h", data[1:3])[0] == GLFW.KEY_W
    assert struct.unpack(">h", data[3:5])[0] == GLFW.KEY_SPACE


def test_to_bytes_with_mouse_deltas():
    raw = RawInput(mouse_dx=10.5, mouse_dy=-3.25)
    data = raw.to_bytes()

    offset = 1  # skip key count (0 keys)
    assert struct.unpack(">f", data[offset : offset + 4])[0] == pytest.approx(10.5)
    assert struct.unpack(">f", data[offset + 4 : offset + 8])[0] == pytest.approx(-3.25)


def test_to_bytes_with_text():
    raw = RawInput(text="hello")
    data = raw.to_bytes()

    text_len_offset = 14  # 1 + 4 + 4 + 1 + 4
    text_len = struct.unpack(">H", data[text_len_offset : text_len_offset + 2])[0]
    assert text_len == 5
    assert data[text_len_offset + 2 :] == b"hello"


def test_to_bytes_with_unicode_text():
    raw = RawInput(text="héllo")
    data = raw.to_bytes()

    text_len_offset = 14
    text_len = struct.unpack(">H", data[text_len_offset : text_len_offset + 2])[0]
    encoded = "héllo".encode("utf-8")
    assert text_len == len(encoded)  # 6 bytes, not 5 characters
    assert data[text_len_offset + 2 :] == encoded


def test_to_bytes_round_trip():
    raw = RawInput(
        key_codes=[GLFW.KEY_W, GLFW.KEY_A],
        mouse_dx=5.0,
        mouse_dy=-2.0,
        mouse_buttons=0b101,  # left + middle
        scroll_delta=1.5,
        text="test",
    )
    data = raw.to_bytes()

    offset = 0
    num_keys = data[offset]
    offset += 1
    assert num_keys == 2

    keys = []
    for _ in range(num_keys):
        keys.append(struct.unpack(">h", data[offset : offset + 2])[0])
        offset += 2
    assert keys == [GLFW.KEY_W, GLFW.KEY_A]

    mouse_dx = struct.unpack(">f", data[offset : offset + 4])[0]
    offset += 4
    assert mouse_dx == pytest.approx(5.0)

    mouse_dy = struct.unpack(">f", data[offset : offset + 4])[0]
    offset += 4
    assert mouse_dy == pytest.approx(-2.0)

    mouse_buttons = data[offset]
    offset += 1
    assert mouse_buttons == 0b101

    scroll_delta = struct.unpack(">f", data[offset : offset + 4])[0]
    offset += 4
    assert scroll_delta == pytest.approx(1.5)

    text_len = struct.unpack(">H", data[offset : offset + 2])[0]
    offset += 2
    assert text_len == 4

    text = data[offset : offset + text_len].decode("utf-8")
    assert text == "test"

    assert offset + text_len == len(data)


# --- to_bytes edge cases ---


def test_to_bytes_too_many_keys():
    raw = RawInput(key_codes=list(range(256)))
    with pytest.raises(ValueError, match="Too many keys"):
        raw.to_bytes()


def test_to_bytes_max_valid_keys():
    raw = RawInput(key_codes=list(range(255)))
    data = raw.to_bytes()
    assert data[0] == 255


def test_to_bytes_text_too_long():
    raw = RawInput(text="x" * 65536)
    with pytest.raises(ValueError, match="Text too long"):
        raw.to_bytes()


# --- Mouse button helpers ---


def test_set_left_mouse():
    raw = RawInput()
    raw.set_left_mouse(True)
    assert raw.mouse_buttons & 1

    raw.set_left_mouse(False)
    assert raw.mouse_buttons == 0


def test_set_right_mouse():
    raw = RawInput()
    raw.set_right_mouse(True)
    assert raw.mouse_buttons & 2

    raw.set_right_mouse(False)
    assert raw.mouse_buttons == 0


def test_set_middle_mouse():
    raw = RawInput()
    raw.set_middle_mouse(True)
    assert raw.mouse_buttons & 4

    raw.set_middle_mouse(False)
    assert raw.mouse_buttons == 0


def test_set_multiple_mouse_buttons():
    raw = RawInput()
    raw.set_left_mouse(True)
    raw.set_right_mouse(True)
    raw.set_middle_mouse(True)
    assert raw.mouse_buttons == 0b111

    raw.set_right_mouse(False)
    assert raw.mouse_buttons == 0b101


# --- release_all ---


def test_release_all():
    raw = RawInput.release_all()
    assert raw.key_codes == []
    assert raw.mouse_dx == 0.0
    assert raw.mouse_dy == 0.0
    assert raw.mouse_buttons == 0
    assert raw.scroll_delta == 0.0
    assert raw.text == ""


# --- parse_observation ---

FRAME_HEIGHT = 4
FRAME_WIDTH = 4
FRAME_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH)
FRAME_NUM_BYTES = FRAME_HEIGHT * FRAME_WIDTH * 3


def _build_header(reward: float, frame_length: int) -> bytes:
    return struct.pack(">d", reward) + struct.pack(">I", frame_length)


def test_parse_observation():
    reward = 2.5
    frame_data = bytes(range(FRAME_NUM_BYTES % 256)) * (
        FRAME_NUM_BYTES // (FRAME_NUM_BYTES % 256) + 1
    )
    frame_data = frame_data[:FRAME_NUM_BYTES]
    header = _build_header(reward, len(frame_data))

    obs = parse_observation(header, frame_data, FRAME_SHAPE)

    assert obs.reward == reward
    assert obs.frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    assert obs.frame.dtype == np.uint8


def test_parse_observation_zero_reward():
    frame_data = b"\x00" * FRAME_NUM_BYTES
    header = _build_header(0.0, FRAME_NUM_BYTES)

    obs = parse_observation(header, frame_data, FRAME_SHAPE)

    assert obs.reward == 0.0
    assert obs.frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    assert np.all(obs.frame == 0)


def test_parse_observation_negative_reward():
    frame_data = b"\xff" * FRAME_NUM_BYTES
    header = _build_header(-100.0, FRAME_NUM_BYTES)

    obs = parse_observation(header, frame_data, FRAME_SHAPE)

    assert obs.reward == -100.0
    assert np.all(obs.frame == 255)


def test_parse_observation_frame_values():
    frame_array = np.arange(FRAME_NUM_BYTES, dtype=np.uint8)
    frame_data = frame_array.tobytes()
    header = _build_header(1.0, FRAME_NUM_BYTES)

    obs = parse_observation(header, frame_data, FRAME_SHAPE)

    expected = frame_array.reshape(FRAME_HEIGHT, FRAME_WIDTH, 3)
    np.testing.assert_array_equal(obs.frame, expected)


def test_parse_observation_invalid_header_length():
    with pytest.raises(ValueError, match="Header must be 12 bytes"):
        parse_observation(b"\x00" * 8, b"\x00" * FRAME_NUM_BYTES, FRAME_SHAPE)


def test_parse_observation_frame_length_mismatch():
    wrong_length = FRAME_NUM_BYTES + 1
    header = _build_header(0.0, wrong_length)

    with pytest.raises(ValueError, match="Frame length mismatch"):
        parse_observation(header, b"\x00" * FRAME_NUM_BYTES, FRAME_SHAPE)


def test_parse_observation_frame_size_mismatch():
    bad_frame = b"\x00" * (FRAME_NUM_BYTES - 1)
    header = _build_header(0.0, len(bad_frame))

    with pytest.raises(ValueError, match="Frame data size mismatch"):
        parse_observation(header, bad_frame, FRAME_SHAPE)


def test_parse_observation_default_frame_shape():
    height, width = 240, 320
    num_bytes = height * width * 3
    frame_data = b"\x00" * num_bytes
    header = _build_header(0.0, num_bytes)

    obs = parse_observation(header, frame_data)

    assert obs.frame.shape == (240, 320, 3)
