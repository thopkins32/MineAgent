---
name: mineagent-ipc
description: >-
  IPC contract between Python mineagent and the Forge mod: Unix socket paths,
  observation framing (reward + frame length + RGB bytes), and RawInput action
  serialization. Use when changing client/connection code, NetworkHandler, or
  debugging desync between Java and Python.
---

# MineAgent IPC (Python ↔ Forge)

## Transport

- **Unix domain sockets** (not TCP in the current `NetworkHandler` / `AsyncMinecraftClient` defaults).
- Default paths (must match **Python** `mineagent/client/connection.py` `ConnectionConfig` and **Java** `NetworkHandler` constants):
  - Observations (server sends, Python reads): `/tmp/mineagent_observation.sock`
  - Actions (Python sends, server reads): `/tmp/mineagent_action.sock`

Changing paths requires updating **both** sides (or making Java read the same config source).

## Observation wire format

**Java** (`NetworkHandler.sendObservationImmediate`): one message per frame:

1. `double` **reward**, big-endian (8 bytes)
2. `int` **frameLength**, big-endian (4 bytes)
3. **frameLength** bytes of raw **RGB** row-major pixel data (`height * width * 3`)

**Python** (`AsyncMinecraftClient.receive_observation`): reads 12-byte header, then `frame_length` bytes; `parse_observation` in `mineagent/client/protocol.py` validates length against `ConnectionConfig.frame_height/frame_width`.

If `frame_length == 0`, Python still returns an `Observation` with a **zero** frame matrix of the configured shape (see `connection.py`).

## Action wire format (`RawInput`)

**Python** `mineagent/client/protocol.py` `RawInput.to_bytes()`:

1. `uint8` number of pressed keys `N`
2. `N` × **big-endian int16** key codes (GLFW codes; canonical order for the Gym space is `KEY_LIST` / `NUM_KEYS` in the same module)
3. `float32` **mouse_dx**, big-endian
4. `float32` **mouse_dy**, big-endian
5. `uint8` **mouse_buttons** (bit flags: left, right, middle)
6. `float32` **scroll_delta**, big-endian
7. `uint16` **text** UTF-8 length, big-endian
8. `text` bytes (UTF-8), may be empty

**Java** `NetworkHandler.handleActionClient` documents the same layout; parsing must stay byte-for-byte compatible.

## Gymnasium action space vs wire format

- `make_action_space()` builds a `Dict` space: `keys` (MultiBinary `NUM_KEYS`), `mouse_dx`, `mouse_dy`, `mouse_buttons`, `scroll_delta`.
- `action_to_raw_input()` maps dict → `RawInput` → bytes.
- **Focus / ROI** is **not** sent on the wire; it is internal to `AgentV1` / perception (see `mineagent-python` skill).

## Parity checklist (when editing protocol)

- [ ] Field order and endianness match in Python `RawInput.to_bytes` and Java read loop
- [ ] Observation header 12 bytes; frame byte count equals `H*W*3` when non-zero
- [ ] Default socket paths identical in `ConnectionConfig` and `NetworkHandler`
- [ ] Frame dimensions: `MinecraftEnv` copies `engine.image_size` into `ConnectionConfig` / env config—keep consistent with mod capture resolution
