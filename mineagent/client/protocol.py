import struct
from dataclasses import dataclass, field

import numpy as np


class GLFW:
    """GLFW key and mouse button constants."""

    KEY_SPACE = 32
    KEY_APOSTROPHE = 39
    KEY_COMMA = 44
    KEY_MINUS = 45
    KEY_PERIOD = 46
    KEY_SLASH = 47
    KEY_0 = 48
    KEY_1 = 49
    KEY_2 = 50
    KEY_3 = 51
    KEY_4 = 52
    KEY_5 = 53
    KEY_6 = 54
    KEY_7 = 55
    KEY_8 = 56
    KEY_9 = 57
    KEY_SEMICOLON = 59
    KEY_EQUAL = 61
    KEY_A = 65
    KEY_B = 66
    KEY_C = 67
    KEY_D = 68
    KEY_E = 69
    KEY_F = 70
    KEY_G = 71
    KEY_H = 72
    KEY_I = 73
    KEY_J = 74
    KEY_K = 75
    KEY_L = 76
    KEY_M = 77
    KEY_N = 78
    KEY_O = 79
    KEY_P = 80
    KEY_Q = 81
    KEY_R = 82
    KEY_S = 83
    KEY_T = 84
    KEY_U = 85
    KEY_V = 86
    KEY_W = 87
    KEY_X = 88
    KEY_Y = 89
    KEY_Z = 90
    KEY_LEFT_BRACKET = 91
    KEY_BACKSLASH = 92
    KEY_RIGHT_BRACKET = 93
    KEY_GRAVE_ACCENT = 96

    KEY_ESCAPE = 256
    KEY_ENTER = 257
    KEY_TAB = 258
    KEY_BACKSPACE = 259
    KEY_INSERT = 260
    KEY_DELETE = 261
    KEY_RIGHT = 262
    KEY_LEFT = 263
    KEY_DOWN = 264
    KEY_UP = 265
    KEY_PAGE_UP = 266
    KEY_PAGE_DOWN = 267
    KEY_HOME = 268
    KEY_END = 269
    KEY_CAPS_LOCK = 280
    KEY_SCROLL_LOCK = 281
    KEY_NUM_LOCK = 282
    KEY_PRINT_SCREEN = 283
    KEY_PAUSE = 284
    KEY_F1 = 290
    KEY_F2 = 291
    KEY_F3 = 292
    KEY_F4 = 293
    KEY_F5 = 294
    KEY_F6 = 295
    KEY_F7 = 296
    KEY_F8 = 297
    KEY_F9 = 298
    KEY_F10 = 299
    KEY_F11 = 300
    KEY_F12 = 301

    KEY_LEFT_SHIFT = 340
    KEY_LEFT_CONTROL = 341
    KEY_LEFT_ALT = 342
    KEY_LEFT_SUPER = 343
    KEY_RIGHT_SHIFT = 344
    KEY_RIGHT_CONTROL = 345
    KEY_RIGHT_ALT = 346
    KEY_RIGHT_SUPER = 347
    KEY_MENU = 348

    MOUSE_BUTTON_LEFT = 0
    MOUSE_BUTTON_RIGHT = 1
    MOUSE_BUTTON_MIDDLE = 2


COMMAND_TO_KEY: dict[str, int] = {
    "w": GLFW.KEY_W,
    "forward": GLFW.KEY_W,
    "s": GLFW.KEY_S,
    "back": GLFW.KEY_S,
    "a": GLFW.KEY_A,
    "left": GLFW.KEY_A,
    "d": GLFW.KEY_D,
    "right": GLFW.KEY_D,
    "space": GLFW.KEY_SPACE,
    "jump": GLFW.KEY_SPACE,
    "shift": GLFW.KEY_LEFT_SHIFT,
    "sneak": GLFW.KEY_LEFT_SHIFT,
    "ctrl": GLFW.KEY_LEFT_CONTROL,
    "sprint": GLFW.KEY_LEFT_CONTROL,
    "e": GLFW.KEY_E,
    "inventory": GLFW.KEY_E,
    "q": GLFW.KEY_Q,
    "drop": GLFW.KEY_Q,
    "f": GLFW.KEY_F,
    "swap": GLFW.KEY_F,
    "1": GLFW.KEY_1,
    "hotbar1": GLFW.KEY_1,
    "2": GLFW.KEY_2,
    "hotbar2": GLFW.KEY_2,
    "3": GLFW.KEY_3,
    "hotbar3": GLFW.KEY_3,
    "4": GLFW.KEY_4,
    "hotbar4": GLFW.KEY_4,
    "5": GLFW.KEY_5,
    "hotbar5": GLFW.KEY_5,
    "6": GLFW.KEY_6,
    "hotbar6": GLFW.KEY_6,
    "7": GLFW.KEY_7,
    "hotbar7": GLFW.KEY_7,
    "8": GLFW.KEY_8,
    "hotbar8": GLFW.KEY_8,
    "9": GLFW.KEY_9,
    "hotbar9": GLFW.KEY_9,
    "esc": GLFW.KEY_ESCAPE,
    "escape": GLFW.KEY_ESCAPE,
    "enter": GLFW.KEY_ENTER,
    "tab": GLFW.KEY_TAB,
    "t": GLFW.KEY_T,
    "chat": GLFW.KEY_T,
    "/": GLFW.KEY_SLASH,
    "command": GLFW.KEY_SLASH,
}

# Canonical ordered list of GLFW key codes included in the action space.
# Index in this list == index in the MultiBinary vector.
KEY_LIST: list[int] = [
    # Movement
    GLFW.KEY_W,  # 0: forward
    GLFW.KEY_S,  # 1: back
    GLFW.KEY_A,  # 2: strafe left
    GLFW.KEY_D,  # 3: strafe right
    GLFW.KEY_SPACE,  # 4: jump
    GLFW.KEY_LEFT_SHIFT,  # 5: sneak
    GLFW.KEY_LEFT_CONTROL,  # 6: sprint
    # Interaction
    GLFW.KEY_E,  # 7: inventory
    GLFW.KEY_Q,  # 8: drop
    GLFW.KEY_F,  # 9: swap offhand
    # Hotbar
    GLFW.KEY_1,  # 10
    GLFW.KEY_2,  # 11
    GLFW.KEY_3,  # 12
    GLFW.KEY_4,  # 13
    GLFW.KEY_5,  # 14
    GLFW.KEY_6,  # 15
    GLFW.KEY_7,  # 16
    GLFW.KEY_8,  # 17
    GLFW.KEY_9,  # 18
    # UI / Menu
    GLFW.KEY_ESCAPE,  # 19
    GLFW.KEY_ENTER,  # 20
    GLFW.KEY_TAB,  # 21
    GLFW.KEY_BACKSPACE,  # 22
    # Chat
    GLFW.KEY_T,  # 23: open chat
    GLFW.KEY_SLASH,  # 24: open command
    # Letters (for typing in chat/signs/etc.)
    GLFW.KEY_B,  # 25
    GLFW.KEY_C,  # 26
    GLFW.KEY_G,  # 27
    GLFW.KEY_H,  # 28
    GLFW.KEY_I,  # 29
    GLFW.KEY_J,  # 30
    GLFW.KEY_K,  # 31
    GLFW.KEY_L,  # 32
    GLFW.KEY_M,  # 33
    GLFW.KEY_N,  # 34
    GLFW.KEY_O,  # 35
    GLFW.KEY_P,  # 36
    GLFW.KEY_R,  # 37
    GLFW.KEY_U,  # 38
    GLFW.KEY_V,  # 39
    GLFW.KEY_X,  # 40
    GLFW.KEY_Y,  # 41
    GLFW.KEY_Z,  # 42
    # Debug
    GLFW.KEY_F1,  # 43
    GLFW.KEY_F2,  # 44
    GLFW.KEY_F3,  # 45
    GLFW.KEY_F5,  # 46
]

NUM_KEYS: int = len(KEY_LIST)

KEY_TO_INDEX: dict[int, int] = {code: idx for idx, code in enumerate(KEY_LIST)}


@dataclass
class RawInput:
    """
    Raw input data to send to Minecraft.

    Protocol format (variable size):
    - 1 byte: numKeysPressed (0-255)
    - N*2 bytes: keyCodes (shorts, big-endian)
    - 4 bytes: mouseDeltaX (float, big-endian)
    - 4 bytes: mouseDeltaY (float, big-endian)
    - 1 byte: mouseButtons (bits: 0=left, 1=right, 2=middle)
    - 4 bytes: scrollDelta (float, big-endian)
    - 2 bytes: textLength (big-endian)
    - M bytes: textBytes (UTF-8)
    """

    key_codes: list[int] = field(default_factory=list)
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    mouse_buttons: int = 0
    scroll_delta: float = 0.0
    text: str = ""

    def to_bytes(self) -> bytes:
        """Serialize to binary protocol format."""
        data = bytearray()

        num_keys = len(self.key_codes)
        if num_keys > 255:
            raise ValueError(f"Too many keys pressed: {num_keys} (max 255)")
        data.append(num_keys)

        for key_code in self.key_codes:
            data.extend(struct.pack(">h", key_code))

        data.extend(struct.pack(">f", self.mouse_dx))
        data.extend(struct.pack(">f", self.mouse_dy))

        data.append(self.mouse_buttons & 0xFF)

        data.extend(struct.pack(">f", self.scroll_delta))

        text_bytes = self.text.encode("utf-8")
        text_length = len(text_bytes)
        if text_length > 65535:
            raise ValueError(f"Text too long: {text_length} bytes (max 65535)")
        data.extend(struct.pack(">H", text_length))
        data.extend(text_bytes)

        return bytes(data)

    def set_left_mouse(self, pressed: bool) -> None:
        if pressed:
            self.mouse_buttons |= 1 << GLFW.MOUSE_BUTTON_LEFT
        else:
            self.mouse_buttons &= ~(1 << GLFW.MOUSE_BUTTON_LEFT)

    def set_right_mouse(self, pressed: bool) -> None:
        if pressed:
            self.mouse_buttons |= 1 << GLFW.MOUSE_BUTTON_RIGHT
        else:
            self.mouse_buttons &= ~(1 << GLFW.MOUSE_BUTTON_RIGHT)

    def set_middle_mouse(self, pressed: bool) -> None:
        if pressed:
            self.mouse_buttons |= 1 << GLFW.MOUSE_BUTTON_MIDDLE
        else:
            self.mouse_buttons &= ~(1 << GLFW.MOUSE_BUTTON_MIDDLE)

    @staticmethod
    def release_all() -> "RawInput":
        """Create an empty input that releases all keys and mouse buttons."""
        return RawInput()


@dataclass
class Observation:
    """Observation received from the Minecraft mod."""

    reward: float
    frame: np.ndarray


def parse_observation(
    header: bytes, frame_data: bytes, frame_shape: tuple[int, int] = (240, 320)
) -> Observation:
    """
    Parse observation from raw bytes.

    Parameters
    ----------
    header : bytes
        12 bytes: reward (double, 8 bytes) + frame length (uint32, 4 bytes)
    frame_data : bytes
        Raw RGB frame data (H*W*3 bytes, RGB order)
    frame_shape : tuple[int, int]
        (height, width) of the frame

    Returns
    -------
    Observation
        Parsed observation with reward and frame
    """
    if len(header) != 12:
        raise ValueError(f"Header must be 12 bytes, got {len(header)}")

    reward = struct.unpack(">d", header[0:8])[0]
    frame_length = struct.unpack(">I", header[8:12])[0]

    if frame_length != len(frame_data):
        raise ValueError(
            f"Frame length mismatch: header says {frame_length}, got {len(frame_data)}"
        )

    height, width = frame_shape
    if len(frame_data) != height * width * 3:
        raise ValueError(
            f"Frame data size mismatch: expected {height * width * 3}, got {len(frame_data)}"
        )

    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 3)

    return Observation(reward=reward, frame=frame)


# ---------------------------------------------------------------------------
# Action space helpers
# ---------------------------------------------------------------------------

MOUSE_DX_RANGE = (-180.0, 180.0)
MOUSE_DY_RANGE = (-180.0, 180.0)
SCROLL_RANGE = (-10.0, 10.0)


def make_action_space():
    """Build the Gymnasium Dict action space that mirrors RawInput."""
    from gymnasium import spaces

    return spaces.Dict(
        {
            "keys": spaces.MultiBinary(NUM_KEYS),
            "mouse_dx": spaces.Box(*MOUSE_DX_RANGE, shape=(), dtype=np.float32),
            "mouse_dy": spaces.Box(*MOUSE_DY_RANGE, shape=(), dtype=np.float32),
            "mouse_buttons": spaces.MultiBinary(3),
            "scroll_delta": spaces.Box(*SCROLL_RANGE, shape=(), dtype=np.float32),
        }
    )


def action_to_raw_input(action: dict[str, np.ndarray]) -> RawInput:
    """
    Convert a Dict-space action sample into a RawInput for the wire protocol.

    Parameters
    ----------
    action : dict[str, np.ndarray]
        A sample from the action space returned by ``make_action_space()``.

    Returns
    -------
    RawInput
        Ready to serialize with ``to_bytes()`` and send to the Forge mod.
    """
    keys_vec = np.asarray(action["keys"], dtype=np.int8).ravel()
    key_codes = [KEY_LIST[i] for i, pressed in enumerate(keys_vec) if pressed]

    mouse_buttons_vec = np.asarray(action["mouse_buttons"], dtype=np.int8).ravel()
    mouse_buttons = 0
    for bit, pressed in enumerate(mouse_buttons_vec):
        if pressed:
            mouse_buttons |= 1 << bit

    return RawInput(
        key_codes=key_codes,
        mouse_dx=float(action["mouse_dx"]),
        mouse_dy=float(action["mouse_dy"]),
        mouse_buttons=mouse_buttons,
        scroll_delta=float(action["scroll_delta"]),
    )


def raw_input_to_action(raw_input: RawInput) -> dict[str, np.ndarray]:
    """
    Convert a RawInput back into a Dict-space action sample.

    Useful for imitation learning or replaying recorded human input.

    Parameters
    ----------
    raw_input : RawInput
        The raw input to convert.

    Returns
    -------
    dict[str, np.ndarray]
        A valid sample for the action space returned by ``make_action_space()``.
    """
    keys = np.zeros(NUM_KEYS, dtype=np.int8)
    for code in raw_input.key_codes:
        idx = KEY_TO_INDEX.get(code)
        if idx is not None:
            keys[idx] = 1

    mouse_buttons = np.zeros(3, dtype=np.int8)
    for bit in range(3):
        if raw_input.mouse_buttons & (1 << bit):
            mouse_buttons[bit] = 1

    return {
        "keys": keys,
        "mouse_dx": np.float32(raw_input.mouse_dx),
        "mouse_dy": np.float32(raw_input.mouse_dy),
        "mouse_buttons": mouse_buttons,
        "scroll_delta": np.float32(raw_input.scroll_delta),
    }
