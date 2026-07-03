import struct

import numpy as np
import pytest

from mineagent.client.protocol import (
    GLFW,
    KEY_TO_INDEX,
    NUM_KEYS,
    ActionMessage,
    MSG_TYPE_ACTION,
    MSG_TYPE_PING,
    MSG_TYPE_RESET,
    MSG_TYPE_TEXT,
    held_state_diff,
    parse_observation,
)


# --- to_bytes: header / message types ---


def test_pure_hold_is_one_byte():
    data = ActionMessage().to_bytes()
    assert len(data) == 1
    assert data[0] == MSG_TYPE_ACTION  # no event flags


def test_reset_is_one_byte():
    data = ActionMessage.reset().to_bytes()
    assert len(data) == 1
    assert data[0] == MSG_TYPE_RESET


def test_ping_is_one_byte():
    data = ActionMessage.ping().to_bytes()
    assert len(data) == 1
    assert data[0] == MSG_TYPE_PING


def test_text_message():
    data = ActionMessage(msg_type=MSG_TYPE_TEXT, text="hello").to_bytes()
    assert data[0] == MSG_TYPE_TEXT
    (text_len,) = struct.unpack(">H", data[1:3])
    assert text_len == 5
    assert data[3:] == b"hello"


# --- to_bytes: worked examples from the spec ---


def test_look_only_is_nine_bytes():
    data = ActionMessage(has_mouse=True, mouse_dx=1.5, mouse_dy=-2.25).to_bytes()
    # 1 header + 4 dx + 4 dy
    assert len(data) == 9
    assert data[0] == MSG_TYPE_ACTION | (1 << 3)
    assert struct.unpack(">f", data[1:5])[0] == pytest.approx(1.5)
    assert struct.unpack(">f", data[5:9])[0] == pytest.approx(-2.25)


def test_press_w_is_five_bytes():
    data = ActionMessage(key_press=[GLFW.KEY_W]).to_bytes()
    # 1 header + 1 numPress + 1 numRelease + 2 keyCode (no release codes)
    assert len(data) == 5
    assert data[0] == MSG_TYPE_ACTION | (1 << 2)
    assert data[1] == 1
    assert data[2] == 0
    assert struct.unpack(">h", data[3:5])[0] == GLFW.KEY_W


def test_release_left_and_scroll():
    data = ActionMessage(
        has_buttons=True,
        button_release=0b001,  # left
        has_scroll=True,
        scroll=-1.0,
    ).to_bytes()
    # 1 header + 1 buttons + 4 scroll
    assert len(data) == 6
    assert data[0] == MSG_TYPE_ACTION | (1 << 4) | (1 << 5)
    # byte = press(0) | (release(0b001) << 3)
    assert data[1] == (0b001 << 3)
    assert struct.unpack(">f", data[2:6])[0] == pytest.approx(-1.0)


def test_button_byte_packs_press_and_release_nibbles():
    data = ActionMessage(
        has_buttons=True,
        button_press=0b001,  # left press
        button_release=0b100,  # middle release
    ).to_bytes()
    assert data[1] == 0b001 | (0b100 << 3)


def test_keys_press_and_release_lists():
    data = ActionMessage(
        key_press=[GLFW.KEY_W, GLFW.KEY_SPACE],
        key_release=[GLFW.KEY_A],
    ).to_bytes()
    assert data[1] == 2  # numPress
    assert data[2] == 1  # numRelease
    codes = struct.unpack(">hh", data[3:7])
    assert codes == (GLFW.KEY_W, GLFW.KEY_SPACE)
    assert struct.unpack(">h", data[7:9])[0] == GLFW.KEY_A


# --- to_bytes: validation ---


def test_too_many_press_keys():
    with pytest.raises(ValueError, match="Too many press keys"):
        ActionMessage(key_press=list(range(256))).to_bytes()


def test_too_many_release_keys():
    with pytest.raises(ValueError, match="Too many release keys"):
        ActionMessage(key_release=list(range(256))).to_bytes()


def test_key_in_both_lists_rejected():
    with pytest.raises(ValueError, match="both press and release"):
        ActionMessage(
            key_press=[GLFW.KEY_W], key_release=[GLFW.KEY_W]
        ).to_bytes()


def test_button_in_both_nibbles_rejected():
    with pytest.raises(ValueError, match="both press and release nibbles"):
        ActionMessage(
            has_buttons=True, button_press=0b001, button_release=0b001
        ).to_bytes()


def test_button_press_out_of_range():
    with pytest.raises(ValueError, match="button_press must be 3 bits"):
        ActionMessage(has_buttons=True, button_press=0b1000).to_bytes()


def test_text_too_long():
    with pytest.raises(ValueError, match="Text too long"):
        ActionMessage(msg_type=MSG_TYPE_TEXT, text="x" * 65536).to_bytes()


# --- from_bytes round trip ---


def test_round_trip_full_action():
    msg = ActionMessage(
        key_press=[GLFW.KEY_W, GLFW.KEY_D],
        key_release=[GLFW.KEY_S],
        has_mouse=True,
        mouse_dx=12.5,
        mouse_dy=-7.25,
        has_buttons=True,
        button_press=0b010,  # right
        button_release=0b001,  # left
        has_scroll=True,
        scroll=2.5,
    )
    parsed = ActionMessage.from_bytes(msg.to_bytes())
    assert parsed.msg_type == MSG_TYPE_ACTION
    assert parsed.key_press == [GLFW.KEY_W, GLFW.KEY_D]
    assert parsed.key_release == [GLFW.KEY_S]
    assert parsed.has_mouse is True
    assert parsed.mouse_dx == pytest.approx(12.5)
    assert parsed.mouse_dy == pytest.approx(-7.25)
    assert parsed.has_buttons is True
    assert parsed.button_press == 0b010
    assert parsed.button_release == 0b001
    assert parsed.has_scroll is True
    assert parsed.scroll == pytest.approx(2.5)


def test_round_trip_pure_hold():
    parsed = ActionMessage.from_bytes(ActionMessage().to_bytes())
    assert parsed.msg_type == MSG_TYPE_ACTION
    assert parsed.key_press == []
    assert parsed.key_release == []
    assert parsed.has_mouse is False
    assert parsed.has_buttons is False
    assert parsed.has_scroll is False


def test_round_trip_reset_and_ping():
    assert ActionMessage.from_bytes(ActionMessage.reset().to_bytes()).msg_type == MSG_TYPE_RESET
    assert ActionMessage.from_bytes(ActionMessage.ping().to_bytes()).msg_type == MSG_TYPE_PING


def test_round_trip_text_unicode():
    text = "héllo 🎮"
    parsed = ActionMessage.from_bytes(
        ActionMessage(msg_type=MSG_TYPE_TEXT, text=text).to_bytes()
    )
    assert parsed.msg_type == MSG_TYPE_TEXT
    assert parsed.text == text


def test_from_bytes_rejects_reserved_bits():
    with pytest.raises(ValueError, match="Reserved flag bits set"):
        ActionMessage.from_bytes(b"\xC0")


def test_from_bytes_rejects_trailing_bytes():
    data = ActionMessage.reset().to_bytes() + b"\x00"
    with pytest.raises(ValueError, match="Trailing bytes"):
        ActionMessage.from_bytes(data)


# --- held_state_diff ---


def test_held_state_diff_press_release_transitions():
    prev_keys = np.zeros(NUM_KEYS, dtype=np.int8)
    prev_keys[KEY_TO_INDEX[GLFW.KEY_W]] = 1
    new_keys = np.zeros(NUM_KEYS, dtype=np.int8)
    new_keys[KEY_TO_INDEX[GLFW.KEY_S]] = 1  # W released, S pressed

    prev_buttons = np.array([1, 0, 0], dtype=np.int8)  # left held
    new_buttons = np.array([0, 1, 0], dtype=np.int8)  # left released, right pressed

    key_press, key_release, button_press, button_release = held_state_diff(
        prev_keys, new_keys, prev_buttons, new_buttons
    )

    assert key_press == [GLFW.KEY_S]
    assert key_release == [GLFW.KEY_W]
    assert button_press == 0b010  # right
    assert button_release == 0b001  # left


def test_held_state_diff_hold_is_no_edge():
    keys = np.zeros(NUM_KEYS, dtype=np.int8)
    keys[KEY_TO_INDEX[GLFW.KEY_W]] = 1
    buttons = np.array([1, 0, 0], dtype=np.int8)

    key_press, key_release, button_press, button_release = held_state_diff(
        keys, keys, buttons, buttons
    )

    assert key_press == []
    assert key_release == []
    assert button_press == 0
    assert button_release == 0


def test_held_state_diff_rejects_wrong_shape():
    with pytest.raises(ValueError, match="keys vectors must have shape"):
        held_state_diff(
            np.zeros(NUM_KEYS, dtype=np.int8),
            np.zeros(NUM_KEYS - 1, dtype=np.int8),
            np.zeros(3, dtype=np.int8),
            np.zeros(3, dtype=np.int8),
        )


# --- parse_observation (unchanged) ---


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
