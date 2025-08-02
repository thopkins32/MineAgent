#!/usr/bin/env python3
"""
Simple test client for debugging action communication with the Minecraft Forge mod.

This client connects to the Unix domain sockets created by the MVI Minecraft mod
and allows you to send test actions and receive observations for debugging.
"""

import socket
import struct
import time
import threading
from dataclasses import dataclass
from typing import Optional
import argparse
import sys


@dataclass
class ActionState:
    """Represents the state of all persistent actions (keys that can be held down)"""
    up: bool = False
    down: bool = False
    left: bool = False
    right: bool = False
    jump: bool = False
    sneak: bool = False
    sprint: bool = False
    inventory: bool = False
    drop: bool = False
    swap: bool = False
    use: bool = False
    attack: bool = False
    pick_item: bool = False
    hotbar1: bool = False
    hotbar2: bool = False
    hotbar3: bool = False
    hotbar4: bool = False
    hotbar5: bool = False
    hotbar6: bool = False
    hotbar7: bool = False
    hotbar8: bool = False

    def to_bytes(self) -> bytes:
        """Convert action state to 3-byte packed format"""
        # First byte: up, down, left, right, jump, sneak, sprint, inventory
        first_byte = 0
        first_byte |= (self.up << 7)
        first_byte |= (self.down << 6)
        first_byte |= (self.left << 5)
        first_byte |= (self.right << 4)
        first_byte |= (self.jump << 3)
        first_byte |= (self.sneak << 2)
        first_byte |= (self.sprint << 1)
        first_byte |= self.inventory

        # Second byte: drop, swap, use, attack, pick_item, hotbar1, hotbar2, hotbar3
        second_byte = 0
        second_byte |= (self.drop << 7)
        second_byte |= (self.swap << 6)
        second_byte |= (self.use << 5)
        second_byte |= (self.attack << 4)
        second_byte |= (self.pick_item << 3)
        second_byte |= (self.hotbar1 << 2)
        second_byte |= (self.hotbar2 << 1)
        second_byte |= self.hotbar3

        # Third byte: hotbar4, hotbar5, hotbar6, hotbar7, hotbar8, (3 bits padding)
        third_byte = 0
        third_byte |= (self.hotbar4 << 7)
        third_byte |= (self.hotbar5 << 6)
        third_byte |= (self.hotbar6 << 5)
        third_byte |= (self.hotbar7 << 4)
        third_byte |= (self.hotbar8 << 3)
        # Bits 0-2 are padding

        return bytes([first_byte, second_byte, third_byte])


@dataclass
class Action:
    """Complete action including persistent and non-persistent actions"""
    action_state: ActionState
    exit_menu: bool = False
    mouse_control_x: float = 0.0
    mouse_control_y: float = 0.0

    def to_bytes(self) -> bytes:
        """Convert action to 12-byte format for network transmission"""
        data = bytearray(12)
        
        # First 3 bytes: ActionState
        action_bytes = self.action_state.to_bytes()
        data[0:3] = action_bytes
        
        # Byte 3: exit_menu flag (bit 7, other bits are padding)
        menu_byte = (1 << 7) if self.exit_menu else 0
        data[3] = menu_byte
        
        # Bytes 4-7: mouse_control_x (float)
        mouse_x_bytes = struct.pack('>f', self.mouse_control_x)
        data[4:8] = mouse_x_bytes
        
        # Bytes 8-11: mouse_control_y (float)
        mouse_y_bytes = struct.pack('>f', self.mouse_control_y)
        data[8:12] = mouse_y_bytes
        
        return bytes(data)


class ActionTestClient:
    """Test client for sending actions to the Minecraft mod"""
    
    def __init__(self, action_socket_path: str = "/tmp/mvi_action.sock"):
        self.action_socket_path = action_socket_path
        self.action_socket: Optional[socket.socket] = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to the action socket"""
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
        """Disconnect from the action socket"""
        if self.action_socket:
            self.action_socket.close()
            self.action_socket = None
        self.connected = False
        print("✓ Disconnected from action socket")
    
    def send_action(self, action: Action) -> bool:
        """Send an action to the mod"""
        if not self.connected or not self.action_socket:
            print("✗ Not connected to action socket")
            return False
        
        try:
            action_bytes = action.to_bytes()
            self.action_socket.send(action_bytes)
            print(f"✓ Sent action: {len(action_bytes)} bytes")
            return True
        except Exception as e:
            print(f"✗ Failed to send action: {e}")
            return False


class ObservationTestClient:
    """Test client for receiving observations from the Minecraft mod"""
    
    def __init__(self, observation_socket_path: str = "/tmp/mvi_observation.sock"):
        self.observation_socket_path = observation_socket_path
        self.observation_socket: Optional[socket.socket] = None
        self.connected = False
        self.running = False
        
    def connect(self) -> bool:
        """Connect to the observation socket"""
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
        """Disconnect from the observation socket"""
        self.running = False
        if self.observation_socket:
            self.observation_socket.close()
            self.observation_socket = None
        self.connected = False
        print("✓ Disconnected from observation socket")
    
    def receive_observations(self):
        """Continuously receive and print observation info"""
        if not self.connected or not self.observation_socket:
            print("✗ Not connected to observation socket")
            return
        
        self.running = True
        observation_count = 0
        
        try:
            while self.running:
                # Read reward (8 bytes, double)
                reward_data = self._read_exact(8)
                if not reward_data:
                    break
                reward = struct.unpack('>d', reward_data)[0]
                
                # Read frame length (4 bytes, int)
                length_data = self._read_exact(4)
                if not length_data:
                    break
                frame_length = struct.unpack('>I', length_data)[0]
                
                # Read frame data
                frame_data = self._read_exact(frame_length)
                if not frame_data:
                    break
                
                observation_count += 1
                print(f"✓ Observation #{observation_count}: reward={reward:.3f}, frame={frame_length} bytes")
                
        except Exception as e:
            print(f"✗ Error receiving observations: {e}")
    
    def _read_exact(self, n: int) -> Optional[bytes]:
        """Read exactly n bytes from the socket"""
        if not self.observation_socket:
            return None
            
        data = b''
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
    """Display comprehensive help information"""
    print("\n" + "=" * 60)
    print("MVI ACTION TEST CLIENT - COMMAND REFERENCE")
    print("=" * 60)
    print("\nMOVEMENT COMMANDS:")
    print("  w, forward      - Move forward")
    print("  s, back         - Move backward") 
    print("  a, left         - Move left")
    print("  d, right        - Move right")
    print("  space, jump     - Jump")
    print("  shift, sneak    - Sneak/crouch")
    print("  ctrl, sprint    - Sprint")
    print()
    print("INTERACTION COMMANDS:")
    print("  attack, lmb     - Attack/left mouse button")
    print("  use, rmb        - Use/right mouse button")
    print("  e, inventory    - Open inventory")
    print("  q, drop         - Drop item")
    print("  f, swap         - Swap offhand")
    print("  pick            - Pick block")
    print()
    print("HOTBAR COMMANDS:")
    print("  1-8             - Select hotbar slots 1-8")
    print("  hotbar1-8       - Select hotbar slots (alternative)")
    print()
    print("MOUSE COMMANDS:")
    print("  mouse <x> <y>   - Move mouse by x,y amount")
    print("  turn <x>        - Turn horizontally by x amount")
    print("  look <y>        - Look vertically by y amount")
    print("  mleft, mright   - Quick turn left/right")
    print("  mup, mdown      - Quick look up/down")
    print()
    print("SPECIAL COMMANDS:")
    print("  esc, exit       - Exit menu/GUI")
    print("  combo <actions> - Execute multiple actions together")
    print()
    print("UTILITY COMMANDS:")
    print("  help, h, ?      - Show this help")
    print("  status          - Show connection status")
    print("  test            - Run quick test sequence")
    print("  clear           - Clear screen")
    print("  quit, exit, q   - Exit program")
    print("\nEXAMPLES:")
    print("  combo w space   - Move forward and jump")
    print("  mouse -10 5     - Turn left 10 units, look up 5 units")
    print("  turn 45         - Turn right 45 degrees")
    print("=" * 60)


def parse_command(command_line: str) -> tuple[ActionState, bool, float, float]:
    """
    Parse command line input and return action components
    Returns: (action_state, exit_menu, mouse_x, mouse_y)
    """
    parts = command_line.strip().lower().split()
    if not parts:
        return ActionState(), False, 0.0, 0.0
    
    main_command = parts[0]
    action_state = ActionState()
    exit_menu = False
    mouse_x, mouse_y = 0.0, 0.0
    
    # Handle special commands first
    if main_command in ["help", "h", "?"]:
        show_help()
        return ActionState(), False, 0.0, 0.0
    
    if main_command == "combo" and len(parts) > 1:
        # Execute multiple actions together
        for action_name in parts[1:]:
            _apply_action_to_state(action_state, action_name)
        return action_state, exit_menu, mouse_x, mouse_y
    
    if main_command == "mouse" and len(parts) >= 3:
        try:
            mouse_x = float(parts[1])
            mouse_y = float(parts[2])
        except ValueError:
            print("✗ Invalid mouse coordinates. Use: mouse <x> <y>")
        return action_state, exit_menu, mouse_x, mouse_y
    
    if main_command == "turn" and len(parts) >= 2:
        try:
            mouse_x = float(parts[1])
        except ValueError:
            print("✗ Invalid turn amount. Use: turn <x>")
        return action_state, exit_menu, mouse_x, mouse_y
    
    if main_command == "look" and len(parts) >= 2:
        try:
            mouse_y = float(parts[1])
        except ValueError:
            print("✗ Invalid look amount. Use: look <y>")
        return action_state, exit_menu, mouse_x, mouse_y
    
    # Handle regular single actions
    _apply_action_to_state(action_state, main_command)
    
    # Handle special cases
    if main_command in ["esc", "exit"]:
        exit_menu = True
    elif main_command == "mleft":
        mouse_x = -10.0
    elif main_command == "mright":
        mouse_x = 10.0
    elif main_command == "mup":
        mouse_y = -10.0
    elif main_command == "mdown":
        mouse_y = 10.0
    
    return action_state, exit_menu, mouse_x, mouse_y


def _apply_action_to_state(action_state: ActionState, action_name: str):
    """Apply a single action to the action state"""
    action_name = action_name.lower()
    
    # Movement
    if action_name in ["w", "forward"]:
        action_state.up = True
    elif action_name in ["s", "back"]:
        action_state.down = True
    elif action_name in ["a", "left"]:
        action_state.left = True
    elif action_name in ["d", "right"]:
        action_state.right = True
    elif action_name in ["space", "jump"]:
        action_state.jump = True
    elif action_name in ["shift", "sneak"]:
        action_state.sneak = True
    elif action_name in ["ctrl", "sprint"]:
        action_state.sprint = True
    
    # Interactions
    elif action_name in ["attack", "lmb"]:
        action_state.attack = True
    elif action_name in ["use", "rmb"]:
        action_state.use = True
    elif action_name in ["e", "inventory"]:
        action_state.inventory = True
    elif action_name in ["q", "drop"]:
        action_state.drop = True
    elif action_name in ["f", "swap"]:
        action_state.swap = True
    elif action_name == "pick":
        action_state.pick_item = True
    
    # Hotbar
    elif action_name in ["1", "hotbar1"]:
        action_state.hotbar1 = True
    elif action_name in ["2", "hotbar2"]:
        action_state.hotbar2 = True
    elif action_name in ["3", "hotbar3"]:
        action_state.hotbar3 = True
    elif action_name in ["4", "hotbar4"]:
        action_state.hotbar4 = True
    elif action_name in ["5", "hotbar5"]:
        action_state.hotbar5 = True
    elif action_name in ["6", "hotbar6"]:
        action_state.hotbar6 = True
    elif action_name in ["7", "hotbar7"]:
        action_state.hotbar7 = True
    elif action_name in ["8", "hotbar8"]:
        action_state.hotbar8 = True


def run_interactive_mode():
    """Run enhanced interactive mode for testing actions"""
    action_client = ActionTestClient()
    observation_client = ObservationTestClient()
    
    print("MVI Action Test Client - Interactive Mode")
    print("=" * 50)
    
    # Connect to action socket
    if not action_client.connect():
        return
    
    # Optionally connect to observation socket
    print("\nDo you want to monitor observations? (y/n): ", end="")
    if input().lower().startswith('y'):
        if observation_client.connect():
            obs_thread = threading.Thread(target=observation_client.receive_observations, daemon=True)
            obs_thread.start()
    
    print("\nInteractive Action Testing Started")
    print("Type 'help' for command reference, 'quit' to exit")
    print("-" * 50)
    
    try:
        while True:
            try:
                command_line = input("Action> ").strip()
                
                if not command_line:
                    continue
                
                # Handle special meta commands
                if command_line.lower() in ["quit", "exit", "q"]:
                    break
                elif command_line.lower() == "status":
                    print(f"✓ Action socket: {'Connected' if action_client.connected else 'Disconnected'}")
                    print(f"✓ Observation socket: {'Connected' if observation_client.connected else 'Disconnected'}")
                    continue
                elif command_line.lower() == "clear":
                    print("\033[2J\033[H")  # Clear screen
                    continue

                elif command_line.lower() == "test":
                    print("Running quick test sequence...")
                    test_sequence = ["w", "a", "s", "d", "space", "attack"]
                    for test_cmd in test_sequence:
                        print(f"  Testing: {test_cmd}")
                        action_state, exit_menu, mouse_x, mouse_y = parse_command(test_cmd)
                        action = Action(action_state, exit_menu, mouse_x, mouse_y)
                        action_client.send_action(action)
                        time.sleep(0.3)
                    print("✓ Test sequence completed")
                    continue
                
                # Parse the command
                action_state, exit_menu, mouse_x, mouse_y = parse_command(command_line)
                
                # Skip if it was just a help command
                if command_line.lower() in ["help", "h", "?"]:
                    continue
                
                # Send the action
                action = Action(action_state, exit_menu, mouse_x, mouse_y)
                action_client.send_action(action)
                
            except EOFError:
                print("\nEOF received, exiting...")
                break
            except Exception as e:
                print(f"✗ Error processing command: {e}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        action_client.disconnect()
        observation_client.disconnect()


def run_automated_test():
    """Run automated test sequence"""
    action_client = ActionTestClient()
    
    print("MVI Automated Action Test")
    print("=" * 50)
    
    if not action_client.connect():
        return
    
    try:
        test_actions = [
            ("Forward", ActionState(up=True)),
            ("Back", ActionState(down=True)),
            ("Left", ActionState(left=True)),
            ("Right", ActionState(right=True)),
            ("Jump", ActionState(jump=True)),
            ("Sneak", ActionState(sneak=True)),
            ("Sprint", ActionState(sprint=True)),
            ("Attack", ActionState(attack=True)),
            ("Use", ActionState(use=True)),
            ("Hotbar 1", ActionState(hotbar1=True)),
            ("Turn Left", ActionState(), False, -10.0, 0.0),
            ("Turn Right", ActionState(), False, 10.0, 0.0),
            ("Exit Menu", ActionState(), True, 0.0, 0.0),
        ]
        
        print("Running automated test sequence...")
        for i, test_data in enumerate(test_actions):
            name = test_data[0]
            action_state = test_data[1]
            exit_menu = test_data[2] if len(test_data) > 2 else False
            mouse_x = test_data[3] if len(test_data) > 3 else 0.0
            mouse_y = test_data[4] if len(test_data) > 4 else 0.0
            
            print(f"  {i+1:2d}. {name}")
            action = Action(action_state, exit_menu, mouse_x, mouse_y)
            action_client.send_action(action)
            time.sleep(0.5)
        
        print("✓ Automated test completed successfully")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        action_client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="MVI Action Test Client")
    parser.add_argument("--auto", action="store_true", help="Run automated test sequence")
    parser.add_argument("--action-socket", default="/tmp/mvi_action.sock", 
                       help="Path to action socket (default: /tmp/mvi_action.sock)")
    parser.add_argument("--obs-socket", default="/tmp/mvi_observation.sock",
                       help="Path to observation socket (default: /tmp/mvi_observation.sock)")
    
    args = parser.parse_args()
    
    if args.auto:
        run_automated_test()
    else:
        run_interactive_mode()


if __name__ == "__main__":
    main() 