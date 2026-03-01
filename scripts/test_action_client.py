#!/usr/bin/env python3
"""
Test client for debugging raw input communication with the MineAgent Minecraft mod.

This client connects to the Unix domain sockets created by the MineAgent mod
and allows you to send raw input (GLFW key codes, mouse, scroll) for testing.
"""

import socket
import struct
import time
import threading
from dataclasses import dataclass
from typing import Optional
import argparse

from mineagent.client import GLFW, RawInput, COMMAND_TO_KEY


class RawInputTestClient:
    """Test client for sending raw input to the Minecraft mod."""

    def __init__(self, action_socket_path: str = "/tmp/mineagent_action.sock"):
        self.action_socket_path = action_socket_path
        self.action_socket: Optional[socket.socket] = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to the action socket."""
        try:
            self.action_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.action_socket.connect(self.action_socket_path)
            self.connected = True
            print(f"✓ Connected to action socket: {self.action_socket_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to action socket: {e}")
            return False

    def disconnect(self):
        """Disconnect from the action socket."""
        if self.action_socket:
            self.action_socket.close()
            self.action_socket = None
        self.connected = False
        print("✓ Disconnected from action socket")

    def send_raw_input(self, raw_input: RawInput) -> bool:
        """Send raw input to the mod."""
        if not self.connected or not self.action_socket:
            print("✗ Not connected to action socket")
            return False

        try:
            input_bytes = raw_input.to_bytes()
            self.action_socket.send(input_bytes)
            print(
                f"✓ Sent raw input: {len(input_bytes)} bytes, {len(raw_input.key_codes)} keys"
            )
            return True
        except Exception as e:
            print(f"✗ Failed to send raw input: {e}")
            return False


class ObservationTestClient:
    """Test client for receiving observations from the Minecraft mod."""

    def __init__(
        self, observation_socket_path: str = "/tmp/mineagent_observation.sock"
    ):
        self.observation_socket_path = observation_socket_path
        self.observation_socket: Optional[socket.socket] = None
        self.connected = False
        self.running = False

    def connect(self) -> bool:
        """Connect to the observation socket."""
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
        """Disconnect from the observation socket."""
        self.running = False
        if self.observation_socket:
            self.observation_socket.close()
            self.observation_socket = None
        self.connected = False
        print("✓ Disconnected from observation socket")

    def receive_observations(self):
        """Continuously receive and print observation info."""
        if not self.connected or not self.observation_socket:
            print("✗ Not connected to observation socket")
            return

        self.running = True
        observation_count = 0

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

                frame_data = self._read_exact(frame_length)
                if not frame_data:
                    break

                observation_count += 1
                print(
                    f"✓ Observation #{observation_count}: reward={reward:.3f}, frame={frame_length} bytes"
                )

        except Exception as e:
            print(f"✗ Error receiving observations: {e}")

    def _read_exact(self, n: int) -> Optional[bytes]:
        """Read exactly n bytes from the socket."""
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


def show_help():
    """Display comprehensive help information."""
    print("\n" + "=" * 70)
    print("MINEAGENT RAW INPUT TEST CLIENT - COMMAND REFERENCE")
    print("=" * 70)
    print("\nMOVEMENT COMMANDS (momentary - pressed then released):")
    print("  w, forward      - Move forward (GLFW_KEY_W = 87)")
    print("  s, back         - Move backward (GLFW_KEY_S = 83)")
    print("  a, left         - Move left (GLFW_KEY_A = 65)")
    print("  d, right        - Move right (GLFW_KEY_D = 68)")
    print("  space, jump     - Jump (GLFW_KEY_SPACE = 32)")
    print("  shift, sneak    - Sneak (GLFW_KEY_LEFT_SHIFT = 340)")
    print("  ctrl, sprint    - Sprint (GLFW_KEY_LEFT_CONTROL = 341)")
    print()
    print("INTERACTION COMMANDS:")
    print("  e, inventory    - Open inventory (GLFW_KEY_E = 69)")
    print("  q, drop         - Drop item (GLFW_KEY_Q = 81)")
    print("  f, swap         - Swap offhand (GLFW_KEY_F = 70)")
    print()
    print("HOTBAR COMMANDS:")
    print("  1-9             - Select hotbar slots (GLFW_KEY_1-9)")
    print()
    print("MOUSE COMMANDS:")
    print("  mouse <x> <y>   - Move mouse by delta (x=yaw/right, y=pitch/down)")
    print("  lclick          - Left mouse click (one-shot, for single attack/interact)")
    print("  rclick          - Right mouse click (one-shot)")
    print("  mclick          - Middle mouse click (one-shot)")
    print("  ldown, lup      - Hold/release left mouse (persistent until lup)")
    print("  rdown, rup      - Hold/release right mouse (persistent until rup)")
    print("  scroll <amount> - Scroll wheel (positive = up)")
    print()
    print("TEXT INPUT:")
    print("  text <string>   - Send raw text input (for chat/signs when open)")
    print("  say <message>   - Open chat, type message, and send it")
    print()
    print("COMBO/HOLD COMMANDS:")
    print("  combo <keys>    - Multiple keys pressed together (momentary)")
    print("  hold <keys>     - Hold keys continuously (persistent until release)")
    print("  release         - Release all held keys AND mouse buttons")
    print()
    print("UTILITY COMMANDS:")
    print("  help, h, ?      - Show this help")
    print("  status          - Show connection status and held state")
    print("  test            - Run quick test sequence")
    print("  clear           - Clear screen")
    print("  quit, q         - Exit program")
    print()
    print("EXAMPLES:")
    print("  w                    - Press W once (brief forward movement)")
    print("  hold w               - Hold W (continuous forward movement)")
    print("  hold w ctrl          - Hold W and sprint together")
    print("  combo w space        - Move forward and jump together (momentary)")
    print("  mouse 100 0          - Turn right")
    print("  mouse 0 -50          - Look up")
    print("  lclick               - Left click once (attack)")
    print("  ldown                - Hold left mouse (for breaking blocks)")
    print("  lup                  - Release left mouse")
    print("  scroll 1             - Scroll hotbar up")
    print("  say Hello world!     - Open chat, type 'Hello world!', and send")
    print("  release              - Release everything")
    print("=" * 70)


class HeldState:
    """Tracks held keys and mouse buttons across commands."""

    def __init__(self):
        self.keys: set[int] = set()
        self.mouse_buttons: int = 0

    def set_mouse_button(self, button: int, pressed: bool):
        """Set a mouse button state. button: 0=left, 1=right, 2=middle"""
        if pressed:
            self.mouse_buttons |= 1 << button
        else:
            self.mouse_buttons &= ~(1 << button)

    def clear(self):
        """Clear all held state."""
        self.keys.clear()
        self.mouse_buttons = 0


def parse_command(
    command_line: str, held_keys: set[int], held_state: Optional["HeldState"] = None
) -> Optional[RawInput]:
    """
    Parse command line input and return a RawInput.

    Args:
        command_line: The command string to parse
        held_keys: Set of currently held key codes (modified in place) - DEPRECATED, use held_state
        held_state: HeldState object tracking keys and mouse buttons

    Returns:
        RawInput to send, or None if command was informational only
    """
    if held_state is None:
        held_state = HeldState()
        held_state.keys = held_keys

    parts = command_line.strip().lower().split()
    if not parts:
        return None

    main_command = parts[0]

    if main_command in ["help", "h", "?"]:
        show_help()
        return None

    raw_input = RawInput()

    if main_command == "mouse" and len(parts) >= 3:
        try:
            raw_input.mouse_dx = float(parts[1])
            raw_input.mouse_dy = float(parts[2])
            raw_input.key_codes = list(held_state.keys)
            raw_input.mouse_buttons = held_state.mouse_buttons
            return raw_input
        except ValueError:
            print("✗ Invalid mouse coordinates. Use: mouse <x> <y>")
            return None

    if main_command == "scroll" and len(parts) >= 2:
        try:
            raw_input.scroll_delta = float(parts[1])
            raw_input.key_codes = list(held_state.keys)
            raw_input.mouse_buttons = held_state.mouse_buttons
            return raw_input
        except ValueError:
            print("✗ Invalid scroll amount. Use: scroll <amount>")
            return None

    if main_command == "text" and len(parts) >= 2:
        raw_input.text = " ".join(parts[1:])
        raw_input.key_codes = list(held_state.keys)
        raw_input.mouse_buttons = held_state.mouse_buttons
        return raw_input

    if main_command == "say" and len(parts) >= 2:
        message = " ".join(parts[1:])
        raw_input.key_codes = [GLFW.KEY_T]
        raw_input.text = f"__SAY__{message}"
        return raw_input

    if main_command == "lclick":
        raw_input.mouse_buttons = held_state.mouse_buttons | (
            1 << GLFW.MOUSE_BUTTON_LEFT
        )
        raw_input.key_codes = list(held_state.keys)
        return raw_input
    if main_command == "rclick":
        raw_input.mouse_buttons = held_state.mouse_buttons | (
            1 << GLFW.MOUSE_BUTTON_RIGHT
        )
        raw_input.key_codes = list(held_state.keys)
        return raw_input
    if main_command == "mclick":
        raw_input.mouse_buttons = held_state.mouse_buttons | (
            1 << GLFW.MOUSE_BUTTON_MIDDLE
        )
        raw_input.key_codes = list(held_state.keys)
        return raw_input

    if main_command == "ldown":
        held_state.set_mouse_button(GLFW.MOUSE_BUTTON_LEFT, True)
        raw_input.mouse_buttons = held_state.mouse_buttons
        raw_input.key_codes = list(held_state.keys)
        return raw_input
    if main_command == "lup":
        held_state.set_mouse_button(GLFW.MOUSE_BUTTON_LEFT, False)
        raw_input.mouse_buttons = held_state.mouse_buttons
        raw_input.key_codes = list(held_state.keys)
        return raw_input
    if main_command == "rdown":
        held_state.set_mouse_button(GLFW.MOUSE_BUTTON_RIGHT, True)
        raw_input.mouse_buttons = held_state.mouse_buttons
        raw_input.key_codes = list(held_state.keys)
        return raw_input
    if main_command == "rup":
        held_state.set_mouse_button(GLFW.MOUSE_BUTTON_RIGHT, False)
        raw_input.mouse_buttons = held_state.mouse_buttons
        raw_input.key_codes = list(held_state.keys)
        return raw_input

    if main_command == "release":
        held_keys.clear()
        held_state.clear()
        return raw_input

    if main_command == "combo" and len(parts) > 1:
        for action_name in parts[1:]:
            if action_name in COMMAND_TO_KEY:
                raw_input.key_codes.append(COMMAND_TO_KEY[action_name])
        raw_input.mouse_buttons = held_state.mouse_buttons
        return raw_input

    if main_command == "hold" and len(parts) > 1:
        for action_name in parts[1:]:
            if action_name in COMMAND_TO_KEY:
                held_state.keys.add(COMMAND_TO_KEY[action_name])
                held_keys.add(COMMAND_TO_KEY[action_name])
        raw_input.key_codes = list(held_state.keys)
        raw_input.mouse_buttons = held_state.mouse_buttons
        return raw_input

    if main_command in COMMAND_TO_KEY:
        raw_input.key_codes = [COMMAND_TO_KEY[main_command]]
        raw_input.mouse_buttons = held_state.mouse_buttons
        return raw_input

    try:
        key_code = int(main_command)
        raw_input.key_codes = [key_code]
        raw_input.mouse_buttons = held_state.mouse_buttons
        return raw_input
    except ValueError:
        pass

    print(f"✗ Unknown command: {main_command}")
    return None


def run_interactive_mode():
    """Run interactive mode for testing raw input."""
    client = RawInputTestClient()
    observation_client = ObservationTestClient()
    held_state = HeldState()
    held_keys = held_state.keys

    print("MineAgent Raw Input Test Client - Interactive Mode")
    print("=" * 50)

    if not client.connect():
        return

    print("\nDo you want to monitor observations? (y/n): ", end="")
    try:
        if input().lower().startswith("y"):
            if observation_client.connect():
                obs_thread = threading.Thread(
                    target=observation_client.receive_observations, daemon=True
                )
                obs_thread.start()
    except EOFError:
        pass

    print("\nInteractive Raw Input Testing Started")
    print("Type 'help' for command reference, 'quit' to exit")
    print("-" * 50)

    try:
        while True:
            try:
                command_line = input("RawInput> ").strip()

                if not command_line:
                    continue

                if command_line.lower() in ["quit", "q"]:
                    break
                elif command_line.lower() == "status":
                    print(
                        f"✓ Action socket: {'Connected' if client.connected else 'Disconnected'}"
                    )
                    print(
                        f"✓ Observation socket: {'Connected' if observation_client.connected else 'Disconnected'}"
                    )
                    print(f"✓ Held keys: {held_state.keys}")
                    print(
                        f"✓ Held mouse buttons: {held_state.mouse_buttons} (L={bool(held_state.mouse_buttons & 1)}, R={bool(held_state.mouse_buttons & 2)}, M={bool(held_state.mouse_buttons & 4)})"
                    )
                    continue
                elif command_line.lower() == "clear":
                    print("\033[2J\033[H")
                    continue
                elif command_line.lower() == "test":
                    run_test_sequence(client)
                    continue

                raw_input = parse_command(command_line, held_keys, held_state)
                if raw_input is not None:
                    if raw_input.text.startswith("__SAY__"):
                        message = raw_input.text[7:]
                        print(f"  Opening chat and typing: {message}")

                        client.send_raw_input(RawInput(key_codes=[GLFW.KEY_T]))
                        time.sleep(0.15)

                        client.send_raw_input(RawInput())
                        time.sleep(0.05)

                        client.send_raw_input(RawInput(text=message))
                        time.sleep(0.05)

                        client.send_raw_input(RawInput(key_codes=[GLFW.KEY_ENTER]))
                        time.sleep(0.05)

                        client.send_raw_input(RawInput())
                    else:
                        client.send_raw_input(raw_input)

            except EOFError:
                print("\nEOF received, exiting...")
                break
            except Exception as e:
                print(f"✗ Error processing command: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        client.send_raw_input(RawInput())
        client.disconnect()
        observation_client.disconnect()


def run_test_sequence(client: RawInputTestClient):
    """Run a quick test sequence."""
    print("Running test sequence...")

    tests = [
        ("Move Forward (W)", RawInput(key_codes=[GLFW.KEY_W])),
        ("Move Left (A)", RawInput(key_codes=[GLFW.KEY_A])),
        ("Move Back (S)", RawInput(key_codes=[GLFW.KEY_S])),
        ("Move Right (D)", RawInput(key_codes=[GLFW.KEY_D])),
        ("Jump (Space)", RawInput(key_codes=[GLFW.KEY_SPACE])),
        ("Turn Right", RawInput(mouse_dx=50.0)),
        ("Turn Left", RawInput(mouse_dx=-50.0)),
        ("Look Up", RawInput(mouse_dy=-30.0)),
        ("Look Down", RawInput(mouse_dy=30.0)),
        ("Left Click", RawInput(mouse_buttons=1)),
        ("Right Click", RawInput(mouse_buttons=2)),
        ("Scroll Up", RawInput(scroll_delta=1.0)),
        ("Release All", RawInput()),
    ]

    for name, raw_input in tests:
        print(f"  Testing: {name}")
        client.send_raw_input(raw_input)
        time.sleep(0.3)

    print("✓ Test sequence completed")


def run_automated_test():
    """Run automated test sequence."""
    client = RawInputTestClient()

    print("MineAgent Automated Raw Input Test")
    print("=" * 50)

    if not client.connect():
        return

    try:
        run_test_sequence(client)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        client.send_raw_input(RawInput())
        client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="MineAgent Raw Input Test Client")
    parser.add_argument(
        "--auto", action="store_true", help="Run automated test sequence"
    )
    parser.add_argument(
        "--action-socket",
        default="/tmp/mineagent_action.sock",
        help="Path to action socket (default: /tmp/mineagent_action.sock)",
    )
    parser.add_argument(
        "--obs-socket",
        default="/tmp/mineagent_observation.sock",
        help="Path to observation socket (default: /tmp/mineagent_observation.sock)",
    )

    args = parser.parse_args()

    if args.auto:
        run_automated_test()
    else:
        run_interactive_mode()


if __name__ == "__main__":
    main()
