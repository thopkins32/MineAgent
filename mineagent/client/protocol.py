import struct
from dataclasses import dataclass, field

import numpy as np
from gymnasium import spaces


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
    # Debug
    GLFW.KEY_F3,  # 23
]

NUM_KEYS: int = len(KEY_LIST)

KEY_TO_INDEX: dict[int, int] = {code: idx for idx, code in enumerate(KEY_LIST)}


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
# Action space helpers (agent-facing, absolute state)
# ---------------------------------------------------------------------------

MOUSE_DX_RANGE = (-180.0, 180.0)
MOUSE_DY_RANGE = (-180.0, 180.0)
SCROLL_RANGE = (-10.0, 10.0)


def make_action_space():
    """Build the Gymnasium Dict action space the agent samples from.

    This is absolute-state: ``keys`` and ``mouse_buttons`` are binary vectors
    describing the *desired held state* for the tick. ``MinecraftEnv`` diffs
    this against its held-state register and emits an event-based
    ``ActionMessage`` on the wire (see :func:`held_state_diff`).
    """

    return spaces.Dict(
        {
            "keys": spaces.MultiBinary(NUM_KEYS),
            "mouse_dx": spaces.Box(*MOUSE_DX_RANGE, shape=(), dtype=np.float32),
            "mouse_dy": spaces.Box(*MOUSE_DY_RANGE, shape=(), dtype=np.float32),
            "mouse_buttons": spaces.MultiBinary(3),
            "scroll_delta": spaces.Box(*SCROLL_RANGE, shape=(), dtype=np.float32),
        }
    )


# ---------------------------------------------------------------------------
# v2 event-based wire protocol (ActionMessage)
# ---------------------------------------------------------------------------

# Message types (low 2 bits of the flags byte).
MSG_TYPE_ACTION = 0
MSG_TYPE_RESET = 1
MSG_TYPE_TEXT = 2
MSG_TYPE_PING = 3

# Flag-bit positions within the 1-byte header (only meaningful for ACTION).
FLAG_HAS_KEYS = 1 << 2
FLAG_HAS_MOUSE = 1 << 3
FLAG_HAS_BUTTONS = 1 << 4
FLAG_HAS_SCROLL = 1 << 5
FLAG_MASK_RESERVED = 0xC0


@dataclass
class ActionMessage:
    """
    Event-based action message sent to the Forge mod.

    Wire format (big-endian):

    - 1 byte ``flags``:
        - bits 0-1: message type (ACTION / RESET / TEXT / PING)
        - bit 2: has key events
        - bit 3: has mouse move
        - bit 4: has button events
        - bit 5: has scroll
        - bits 6-7: reserved (must be 0)
    - key events (if bit 2): ``u8 numPress``, ``u8 numRelease``,
      ``numPress`` x ``i16`` key codes to PRESS, ``numRelease`` x ``i16`` key
      codes to RELEASE. Keys in neither list are HOLD.
    - mouse move (if bit 3): ``f32 dx``, ``f32 dy``.
    - button events (if bit 4): 1 byte, low 3 bits = buttons to PRESS,
      next 3 bits = buttons to RELEASE. Buttons in neither nibble are HOLD.
    - scroll (if bit 5): ``f32 delta``.
    - TEXT body: ``u16`` UTF-8 length + bytes (its own message type).
    - RESET / PING: no body.

    ``key_press``/``key_release`` hold GLFW key codes (see :data:`KEY_LIST`).
    ``button_press``/``button_release`` are 3-bit masks where bit 0 = left,
    bit 1 = right, bit 2 = middle (matching :class:`GLFW` MOUSE_BUTTON_*).
    """

    msg_type: int = MSG_TYPE_ACTION
    key_press: list[int] = field(default_factory=list)
    key_release: list[int] = field(default_factory=list)
    has_mouse: bool = False
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    has_buttons: bool = False
    button_press: int = 0
    button_release: int = 0
    has_scroll: bool = False
    scroll: float = 0.0
    text: str = ""

    def _validate(self) -> None:
        if self.msg_type & ~0x3:
            raise ValueError(f"msg_type must fit in 2 bits, got {self.msg_type}")
        if self.button_press & ~0x7:
            raise ValueError(
                f"button_press must be 3 bits, got {self.button_press}"
            )
        if self.button_release & ~0x7:
            raise ValueError(
                f"button_release must be 3 bits, got {self.button_release}"
            )
        if self.button_press & self.button_release:
            raise ValueError(
                "a button may not be in both press and release nibbles"
            )
        if len(self.key_press) > 255:
            raise ValueError(
                f"Too many press keys: {len(self.key_press)} (max 255)"
            )
        if len(self.key_release) > 255:
            raise ValueError(
                f"Too many release keys: {len(self.key_release)} (max 255)"
            )
        press_set = set(self.key_press)
        release_set = set(self.key_release)
        overlap = press_set & release_set
        if overlap:
            raise ValueError(
                f"keys may not be in both press and release lists: {sorted(overlap)}"
            )
        if self.msg_type == MSG_TYPE_TEXT:
            text_bytes = self.text.encode("utf-8")
            if len(text_bytes) > 65535:
                raise ValueError(
                    f"Text too long: {len(text_bytes)} bytes (max 65535)"
                )

    def to_bytes(self) -> bytes:
        """Serialize to the v2 wire format."""
        self._validate()

        flags = self.msg_type & 0x3
        if self.msg_type == MSG_TYPE_ACTION:
            has_keys = bool(self.key_press or self.key_release)
            if has_keys:
                flags |= FLAG_HAS_KEYS
            if self.has_mouse:
                flags |= FLAG_HAS_MOUSE
            if self.has_buttons:
                flags |= FLAG_HAS_BUTTONS
            if self.has_scroll:
                flags |= FLAG_HAS_SCROLL

        data = bytearray()
        data.append(flags)

        if self.msg_type == MSG_TYPE_ACTION:
            if flags & FLAG_HAS_KEYS:
                data.append(len(self.key_press))
                data.append(len(self.key_release))
                for code in self.key_press:
                    data.extend(struct.pack(">h", code))
                for code in self.key_release:
                    data.extend(struct.pack(">h", code))
            if flags & FLAG_HAS_MOUSE:
                data.extend(struct.pack(">f", self.mouse_dx))
                data.extend(struct.pack(">f", self.mouse_dy))
            if flags & FLAG_HAS_BUTTONS:
                data.append(
                    (self.button_press & 0x7)
                    | ((self.button_release & 0x7) << 3)
                )
            if flags & FLAG_HAS_SCROLL:
                data.extend(struct.pack(">f", self.scroll))
        elif self.msg_type == MSG_TYPE_TEXT:
            text_bytes = self.text.encode("utf-8")
            data.extend(struct.pack(">H", len(text_bytes)))
            data.extend(text_bytes)
        # RESET and PING carry no body.

        return bytes(data)

    @staticmethod
    def reset() -> "ActionMessage":
        """A RESET message: clears all held key/button state on Java."""
        return ActionMessage(msg_type=MSG_TYPE_RESET)

    @staticmethod
    def ping() -> "ActionMessage":
        """A PING heartbeat message (no body)."""
        return ActionMessage(msg_type=MSG_TYPE_PING)

    @staticmethod
    def from_bytes(data: bytes) -> "ActionMessage":
        """Parse a v2 wire message. Primarily for tests / replay."""
        if len(data) < 1:
            raise ValueError("Empty action message")
        flags = data[0]
        if flags & FLAG_MASK_RESERVED:
            raise ValueError(f"Reserved flag bits set: {flags:#04x}")
        msg_type = flags & 0x3

        cur = 1
        msg = ActionMessage(msg_type=msg_type)

        if msg_type == MSG_TYPE_ACTION:
            if flags & FLAG_HAS_KEYS:
                num_press, num_release = data[cur], data[cur + 1]
                cur += 2
                msg.key_press = list(
                    struct.unpack_from(f">{num_press}h", data, cur)
                )
                cur += num_press * 2
                msg.key_release = list(
                    struct.unpack_from(f">{num_release}h", data, cur)
                )
                cur += num_release * 2
            if flags & FLAG_HAS_MOUSE:
                msg.mouse_dx, msg.mouse_dy = struct.unpack_from(">ff", data, cur)
                msg.has_mouse = True
                cur += 8
            if flags & FLAG_HAS_BUTTONS:
                byte = data[cur]
                cur += 1
                msg.button_press = byte & 0x7
                msg.button_release = (byte >> 3) & 0x7
                msg.has_buttons = True
            if flags & FLAG_HAS_SCROLL:
                (msg.scroll,) = struct.unpack_from(">f", data, cur)
                msg.has_scroll = True
                cur += 4
        elif msg_type == MSG_TYPE_TEXT:
            (text_len,) = struct.unpack_from(">H", data, cur)
            cur += 2
            msg.text = data[cur : cur + text_len].decode("utf-8")
            cur += text_len
        # RESET / PING: no body.

        if cur != len(data):
            raise ValueError(
                f"Trailing bytes: consumed {cur} of {len(data)}"
            )
        return msg


def held_state_diff(
    prev_keys: np.ndarray,
    new_keys: np.ndarray,
    prev_buttons: np.ndarray,
    new_buttons: np.ndarray,
) -> tuple[list[int], list[int], int, int]:
    """
    Compute PRESS/RELEASE edges between two absolute held states.

    Parameters
    ----------
    prev_keys, new_keys : np.ndarray
        int8 vectors of length ``NUM_KEYS`` (1 = held). The diff is taken
        ``new_keys - prev_keys``: a 0->1 transition is PRESS, 1->0 is RELEASE.
    prev_buttons, new_buttons : np.ndarray
        int8 vectors of length 3 (left, right, middle), same semantics.

    Returns
    -------
    (key_press, key_release, button_press, button_release)
        ``key_*`` are lists of GLFW key codes from :data:`KEY_LIST`;
        ``button_*`` are 3-bit masks (bit 0 = left, 1 = right, 2 = middle).
    """
    prev_keys = np.asarray(prev_keys, dtype=np.int8).ravel()
    new_keys = np.asarray(new_keys, dtype=np.int8).ravel()
    prev_buttons = np.asarray(prev_buttons, dtype=np.int8).ravel()
    new_buttons = np.asarray(new_buttons, dtype=np.int8).ravel()

    if prev_keys.shape != (NUM_KEYS,) or new_keys.shape != (NUM_KEYS,):
        raise ValueError(
            f"keys vectors must have shape ({NUM_KEYS},), got "
            f"{prev_keys.shape} and {new_keys.shape}"
        )
    if prev_buttons.shape != (3,) or new_buttons.shape != (3,):
        raise ValueError(
            f"buttons vectors must have shape (3,), got "
            f"{prev_buttons.shape} and {new_buttons.shape}"
        )

    press_idx = np.where((new_keys == 1) & (prev_keys == 0))[0]
    release_idx = np.where((new_keys == 0) & (prev_keys == 1))[0]
    key_press = [KEY_LIST[i] for i in press_idx]
    key_release = [KEY_LIST[i] for i in release_idx]

    button_press = 0
    button_release = 0
    for bit in range(3):
        if new_buttons[bit] == 1 and prev_buttons[bit] == 0:
            button_press |= 1 << bit
        elif new_buttons[bit] == 0 and prev_buttons[bit] == 1:
            button_release |= 1 << bit

    return key_press, key_release, button_press, button_release
