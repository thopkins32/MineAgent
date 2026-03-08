import struct

import pytest

from mineagent.client.protocol import GLFW, RawInput


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
