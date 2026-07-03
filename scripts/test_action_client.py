#!/usr/bin/env python3
"""
Test client for debugging action communication with the MineAgent Minecraft mod.

This client connects to the Unix domain sockets created by the MineAgent mod and
sends v2 event-based ActionMessages (PRESS / RELEASE / HOLD edges for keys and
mouse buttons, plus mouse move, scroll, and text) for manual testing.
"""

import argparse
import socket
import struct
import threading
import time
from typing import Optional

from mineagent.client import (
    GLFW,
    COMMAND_TO_KEY,
    ActionMessage,
    MSG_TYPE_TEXT,
)


def _key_edges(prev: set[int], new: set[int]) -> tuple[list[int], list[int]]:
    """Return (press_codes, release_codes) for a held-set transition."""
    return sorted(new - prev), sorted(prev - new)


def _button_edges(prev: int, new: int) -> tuple[int, int]:
    """Return (press_mask, release_mask) for a 3-bit button-state transition."""
    press = 0
    release = 0
    for bit in range(3):
        if (new >> bit) & 1 and not (prev >> bit) & 1:
            press |= 1 << bit
        if not (new >> bit) & 1 and (prev >> bit) & 1:
            release |= 1 << bit
    return press, release


class ActionTestClient:
    """Test client for sending ActionMessages to the Minecraft mod."""

    def __init__(self, action_socket_path: str = "/tmp/mineagent_action.sock"):
        self.action_socket_path = action_socket_path
        self.action_socket: Optional[socket.socket] = None
        self.connected = False

    def connect(self) -> bool:
        try:
            self.action_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.action_socket.connect(self.action_socket_path)
            self.connected = True
            print(f"✓ Connected to action socket: {self.action_socket_path}")
            # Reset Java held state on connect so we start clean.
            self.send(ActionMessage.reset())
            return True
        except Exception as e:
            print(f"✗ Failed to connect to action socket: {e}")
            return False

    def disconnect(self):
        if self.action_socket:
            try:
                self.send(ActionMessage.reset())
            except Exception:
                pass
            self.action_socket.close()
            self.action_socket = None
        self.connected = False
        print("✓ Disconnected from action socket")

    def send(self, message: ActionMessage) -> bool:
        if not self.connected or not self.action_socket:
            print("✗ Not connected to action socket")
            return False
        try:
            data = message.to_bytes()
            self.action_socket.send(data)
            print(f"✓ Sent ActionMessage: {len(data)} bytes")
            return True
        except Exception as e:
            print(f"✗ Failed to send action: {e}")
            return False


class ObservationTestClient:
    """Test client for receiving observations from the Minecraft mod."""

    def __init__(self, observation_socket_path: str = "/tmp/mineagent_observation.sock"):
        self.observation_socket_path = observation_socket_path
        self.observation_socket: Optional[socket.socket] = None
        self.connected = False
        self.running = False

    def connect(self) -> bool:
        try:
            self.observation_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.observation_socket.connect(self.observation_socket_path)
            self.connected = True
            print(f"✓ Connected to observation socket: {self.observation_socket_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to observation socket: {e}")
            return False

    def disconnect(self):
        self.running = False
        if self.observation_socket:
            self.observation_socket.close()
            self.observation_socket = None
        self.connected = False
        print("✓ Disconnected from observation socket")

    def receive_observations(self):
        if not self.connected or not self.observation_socket:
            print("✗ Not connected to observation socket")
            return
        self.running = True
        count = 0
        try:
            while self.running:
                reward_data = self._read_exact(8)
                if not reward_data:
                    break
                reward = struct.unpack(">d", reward_data)[0]
                length_data = self._read_exact(4)
                if not length_data:
                    break
                frame_length = struct.unpack(">I", length_data)[0]
                if not self._read_exact(frame_length):
                    break
                count += 1
                print(f"✓ Observation #{count}: reward={reward:.3f}, frame={frame_length} bytes")
        except Exception as e:
            print(f"✗ Error receiving observations: {e}")

    def _read_exact(self, n: int) -> Optional[bytes]:
        if not self.observation_socket:
            return None
        data = b""
        while len(data) < n:
            try:
                chunk = self.observation_socket.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except Exception:
                return None
        return data


class HeldState:
    """Tracks held keys and mouse buttons across commands."""

    def __init__(self):
        self.keys: set[int] = set()
        self.mouse_buttons: int = 0

    def set_mouse_button(self, button: int, pressed: bool) -> None:
        if pressed:
            self.mouse_buttons |= 1 << button
        else:
            self.mouse_buttons &= ~(1 << button)

    def clear(self) -> None:
        self.keys.clear()
        self.mouse_buttons = 0


def show_help():
    print("\n" + "=" * 70)
    print("MINEAGENT ACTION TEST CLIENT - COMMAND REFERENCE (v2 event protocol)")
    print("=" * 70)
    print("\nMOVEMENT (momentary: press then release):")
    print("  w, s, a, d, space, shift, ctrl   - tap a movement key")
    print("  combo <keys>                     - tap several keys together")
    print("\nHOLD (persistent until release):")
    print("  hold <keys>                      - hold keys continuously")
    print("  release                          - release all held keys + buttons")
    print("\nMOUSE:")
    print("  mouse <dx> <dy>                  - move mouse by delta")
    print("  lclick / rclick / mclick         - tap a mouse button")
    print("  ldown/lup rdown/rup              - hold/release mouse buttons")
    print("  scroll <amount>                  - scroll wheel (positive = up)")
    print("\nTEXT:")
    print("  text <string>                    - send text (chat/signs when open)")
    print("  say <message>                    - open chat, type, send")
    print("\nUTILITY:")
    print("  help, h, ?      - show this help")
    print("  status          - show connection + held state")
    print("  test            - run quick test sequence")
    print("  quit, q         - exit")
    print("=" * 70)


def run_command(
    line: str, client: ActionTestClient, held: HeldState
) -> Optional[bool]:
    """Execute one command line. Returns False to exit, None otherwise."""
    parts = line.strip().split()
    if not parts:
        return None
    cmd = parts[0].lower()

    if cmd in ("quit", "q"):
        return False
    if cmd in ("help", "h", "?"):
        show_help()
        return None
    if cmd == "status":
        print(f"✓ Action socket: {'Connected' if client.connected else 'Disconnected'}")
        print(f"✓ Held keys: {sorted(held.keys)}")
        print(
            f"✓ Held mouse: {held.mouse_buttons} "
            f"(L={bool(held.mouse_buttons & 1)}, R={bool(held.mouse_buttons & 2)}, "
            f"M={bool(held.mouse_buttons & 4)})"
        )
        return None

    if cmd == "release":
        prev_keys = set(held.keys)
        prev_buttons = held.mouse_buttons
        held.clear()
        kp, kr = _key_edges(prev_keys, held.keys)
        bp, br = _button_edges(prev_buttons, held.mouse_buttons)
        client.send(
            ActionMessage(
                key_press=kp, key_release=kr, has_buttons=True,
                button_press=bp, button_release=br,
            )
        )
        return None

    if cmd == "mouse" and len(parts) >= 3:
        try:
            dx, dy = float(parts[1]), float(parts[2])
        except ValueError:
            print("✗ Invalid mouse coordinates. Use: mouse <dx> <dy>")
            return None
        client.send(ActionMessage(has_mouse=True, mouse_dx=dx, mouse_dy=dy))
        return None

    if cmd == "scroll" and len(parts) >= 2:
        try:
            amount = float(parts[1])
        except ValueError:
            print("✗ Invalid scroll amount. Use: scroll <amount>")
            return None
        client.send(ActionMessage(has_scroll=True, scroll=amount))
        return None

    if cmd == "text" and len(parts) >= 2:
        text = " ".join(parts[1:])
        client.send(ActionMessage(msg_type=MSG_TYPE_TEXT, text=text))
        return None

    if cmd == "say" and len(parts) >= 2:
        message = " ".join(parts[1:])
        _say(client, message)
        return None

    # Mouse button hold/release.
    button_map = {
        "ldown": (GLFW.MOUSE_BUTTON_LEFT, True),
        "lup": (GLFW.MOUSE_BUTTON_LEFT, False),
        "rdown": (GLFW.MOUSE_BUTTON_RIGHT, True),
        "rup": (GLFW.MOUSE_BUTTON_RIGHT, False),
        "mdown": (GLFW.MOUSE_BUTTON_MIDDLE, True),
        "mup": (GLFW.MOUSE_BUTTON_MIDDLE, False),
    }
    if cmd in button_map:
        button, pressed = button_map[cmd]
        prev_buttons = held.mouse_buttons
        held.set_mouse_button(button, pressed)
        bp, br = _button_edges(prev_buttons, held.mouse_buttons)
        client.send(
            ActionMessage(has_buttons=True, button_press=bp, button_release=br)
        )
        return None

    # Momentary mouse click: press, sleep, release.
    click_map = {
        "lclick": GLFW.MOUSE_BUTTON_LEFT,
        "rclick": GLFW.MOUSE_BUTTON_RIGHT,
        "mclick": GLFW.MOUSE_BUTTON_MIDDLE,
    }
    if cmd in click_map:
        button = click_map[cmd]
        client.send(ActionMessage(has_buttons=True, button_press=1 << button))
        time.sleep(0.05)
        client.send(ActionMessage(has_buttons=True, button_release=1 << button))
        return None

    # Hold keys persistently.
    if cmd == "hold" and len(parts) > 1:
        prev_keys = set(held.keys)
        for name in parts[1:]:
            code = COMMAND_TO_KEY.get(name.lower())
            if code is not None:
                held.keys.add(code)
        kp, kr = _key_edges(prev_keys, held.keys)
        client.send(ActionMessage(key_press=kp, key_release=kr))
        return None

    # Momentary key tap(s): press, sleep, release.
    if cmd == "combo" and len(parts) > 1:
        codes = [COMMAND_TO_KEY[n] for n in parts[1:] if n in COMMAND_TO_KEY]
        if codes:
            client.send(ActionMessage(key_press=codes))
            time.sleep(0.05)
            client.send(ActionMessage(key_release=codes))
        return None

    if cmd in COMMAND_TO_KEY:
        code = COMMAND_TO_KEY[cmd]
        client.send(ActionMessage(key_press=[code]))
        time.sleep(0.05)
        client.send(ActionMessage(key_release=[code]))
        return None

    # Raw integer key code.
    try:
        code = int(cmd)
        client.send(ActionMessage(key_press=[code]))
        time.sleep(0.05)
        client.send(ActionMessage(key_release=[code]))
        return None
    except ValueError:
        pass

    print(f"✗ Unknown command: {cmd}")
    return None


def _say(client: ActionTestClient, message: str) -> None:
    """Open chat, type the message, and send it."""
    print(f"  Opening chat and typing: {message}")
    # Press T to open chat, then release.
    client.send(ActionMessage(key_press=[GLFW.KEY_T]))
    time.sleep(0.15)
    client.send(ActionMessage(key_release=[GLFW.KEY_T]))
    time.sleep(0.05)
    # Type the message.
    client.send(ActionMessage(msg_type=MSG_TYPE_TEXT, text=message))
    time.sleep(0.05)
    # Press Enter to send, then release.
    client.send(ActionMessage(key_press=[GLFW.KEY_ENTER]))
    time.sleep(0.05)
    client.send(ActionMessage(key_release=[GLFW.KEY_ENTER]))


def run_test_sequence(client: ActionTestClient):
    """Run a quick test sequence."""
    print("Running test sequence...")
    steps = [
        ("Move Forward (W)", lambda c: _tap_key(c, GLFW.KEY_W)),
        ("Move Left (A)", lambda c: _tap_key(c, GLFW.KEY_A)),
        ("Move Back (S)", lambda c: _tap_key(c, GLFW.KEY_S)),
        ("Move Right (D)", lambda c: _tap_key(c, GLFW.KEY_D)),
        ("Jump (Space)", lambda c: _tap_key(c, GLFW.KEY_SPACE)),
        ("Turn Right", lambda c: c.send(ActionMessage(has_mouse=True, mouse_dx=50.0))),
        ("Turn Left", lambda c: c.send(ActionMessage(has_mouse=True, mouse_dx=-50.0))),
        ("Look Up", lambda c: c.send(ActionMessage(has_mouse=True, mouse_dy=-30.0))),
        ("Look Down", lambda c: c.send(ActionMessage(has_mouse=True, mouse_dy=30.0))),
        ("Left Click", lambda c: _tap_button(c, GLFW.MOUSE_BUTTON_LEFT)),
        ("Right Click", lambda c: _tap_button(c, GLFW.MOUSE_BUTTON_RIGHT)),
        ("Scroll Up", lambda c: c.send(ActionMessage(has_scroll=True, scroll=1.0))),
        ("Release All", lambda c: c.send(ActionMessage.reset())),
    ]
    for name, action in steps:
        print(f"  Testing: {name}")
        action(client)
        time.sleep(0.3)
    print("✓ Test sequence completed")


def _tap_key(client: ActionTestClient, code: int) -> None:
    client.send(ActionMessage(key_press=[code]))
    time.sleep(0.05)
    client.send(ActionMessage(key_release=[code]))


def _tap_button(client: ActionTestClient, button: int) -> None:
    client.send(ActionMessage(has_buttons=True, button_press=1 << button))
    time.sleep(0.05)
    client.send(ActionMessage(has_buttons=True, button_release=1 << button))


def run_interactive_mode():
    client = ActionTestClient()
    observation_client = ObservationTestClient()
    held = HeldState()

    print("MineAgent Action Test Client - Interactive Mode")
    print("=" * 50)

    if not client.connect():
        return

    print("\nMonitor observations? (y/n): ", end="")
    try:
        if input().lower().startswith("y"):
            if observation_client.connect():
                threading.Thread(
                    target=observation_client.receive_observations, daemon=True
                ).start()
    except EOFError:
        pass

    print("\nInteractive Action Testing Started")
    print("Type 'help' for command reference, 'quit' to exit")
    print("-" * 50)

    try:
        while True:
            try:
                line = input("Action> ").strip()
                if not line:
                    continue
                result = run_command(line, client, held)
                if result is False:
                    break
            except EOFError:
                print("\nEOF received, exiting...")
                break
            except Exception as e:
                print(f"✗ Error processing command: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        client.disconnect()
        observation_client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="MineAgent Action Test Client")
    parser.add_argument("--auto", action="store_true", help="Run automated test sequence")
    parser.add_argument("--action-socket", default="/tmp/mineagent_action.sock")
    parser.add_argument("--obs-socket", default="/tmp/mineagent_observation.sock")
    args = parser.parse_args()

    if args.auto:
        client = ActionTestClient(args.action_socket)
        print("MineAgent Automated Action Test")
        print("=" * 50)
        if not client.connect():
            return
        try:
            run_test_sequence(client)
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        finally:
            client.disconnect()
    else:
        run_interactive_mode()


if __name__ == "__main__":
    main()
