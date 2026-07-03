---
name: mineagent-ipc
description: >-
  IPC contract between Python mineagent and the Forge mod: Unix socket paths,
  observation framing, and the event-based ActionMessage wire format. Use when
  changing client/connection code, NetworkHandler, InputInjector, or debugging
  desync between Java and Python.
---

# MineAgent IPC (Python ↔ Forge)

## Transport

- **Unix domain sockets** (not TCP in the current `NetworkHandler` / `AsyncMinecraftClient` defaults).
- Default paths (must match **Python** `mineagent/client/connection.py` `ConnectionConfig` and **Java** `NetworkHandler` constants):
  - Observations (server sends, Python reads): `/tmp/mineagent_observation.sock`
  - Actions (Python sends, server reads): `/tmp/mineagent_action.sock`

Changing paths requires updating **both** sides (or making Java read the same config source).

## Implementation status

The action protocol below is the **target (v2) event-based spec**. The current
code still ships the legacy absolute-state `RawInput` format (see
`mineagent/client/protocol.py` `RawInput.to_bytes` and Java
`NetworkHandler.handleActionClient`). During migration, both sides must agree on
which format is in use; do not mix them on a socket. The observation format is
unchanged in this revision.

## Observation wire format

**Java** (`NetworkHandler.sendObservationImmediate`): one message per frame:

1. `double` **reward**, big-endian (8 bytes)
2. `int` **frameLength**, big-endian (4 bytes)
3. **frameLength** bytes of raw **RGB** row-major pixel data (`height * width * 3`)

**Python** (`AsyncMinecraftClient.receive_observation`): reads 12-byte header, then `frame_length` bytes; `parse_observation` in `mineagent/client/protocol.py` validates length against `ConnectionConfig.frame_height/frame_width`.

If `frame_length == 0`, Python still returns an `Observation` with a **zero** frame matrix of the configured shape (see `connection.py`).

## Action wire format (v2 — `ActionMessage`, event-based)

Actions are **edge-triggered**: each message describes *changes* to the held
input state, not an absolute snapshot. Anything not listed in a message is
**HOLD** — Java keeps the previous state. This makes a true no-op expressible
(an `ACTION` with no event flags set = hold everything), unifies keys and mouse
buttons under one model, and lets the agent run slower than the game tick rate
without dropping held inputs (Java maintains held state between messages).

All multi-byte fields are **big-endian**. Fields are present **only** when their
flag bit is set, and appear in the fixed order: header → keys → mouse → buttons
→ scroll. Java reads in that order.

### Header — 1 byte `flags`

| Bits   | Meaning                                                                 |
|--------|-------------------------------------------------------------------------|
| 0-1    | message type: `0=ACTION`, `1=RESET`, `2=TEXT`, `3=PING`                |
| 2      | (ACTION) has key events                                                 |
| 3      | (ACTION) has mouse move                                                 |
| 4      | (ACTION) has button events                                              |
| 5      | (ACTION) has scroll                                                     |
| 6-7    | reserved — must be `0`; Java rejects non-zero                            |

### `ACTION` body (fields present only if their flag is set)

**Key events** (flag bit 2):

1. `uint8` `numPress`
2. `uint8` `numRelease`
3. `numPress` × `int16` **GLFW key codes** to **PRESS** (big-endian)
4. `numRelease` × `int16` **GLFW key codes** to **RELEASE** (big-endian)

Key codes not listed in either list = **HOLD**. Key codes use the same `GLFW`
constants / `KEY_LIST` defined in `mineagent/client/protocol.py`. `numPress` and
`numRelease` are independent counts; a key may appear in at most one of the two
lists per message (Java treats PRESS|RELEASE on the same key in one message as
undefined — sender must not do this).

**Mouse move** (flag bit 3):

1. `float32` `dx`, big-endian
2. `float32` `dy`, big-endian

Semantics unchanged from legacy: pixel deltas scaled by Minecraft sensitivity
into yaw/pitch via `mc.player.turn()`. `dx=0, dy=0` is allowed and is a no-op
for movement (do not set the mouse-move flag if you have nothing to send).

**Button events** (flag bit 4) — 1 byte:

| Bits   | Meaning                                                       |
|--------|---------------------------------------------------------------|
| 0-2    | buttons to **PRESS**: bit0=left, bit1=right, bit2=middle      |
| 3-5    | buttons to **RELEASE**: bit3=left, bit4=right, bit5=middle    |
| 6-7    | reserved — must be `0`                                         |

A button set in **neither** nibble = **HOLD**. A button must not be set in both
nibbles simultaneously. Button index mapping (`0=left, 1=right, 2=middle`)
matches `GLFW.MOUSE_BUTTON_*` in `protocol.py` and the existing
`InputInjector.handleMouseButtons` convention.

**Scroll** (flag bit 5):

1. `float32` `scroll_delta`, big-endian

### `RESET` body

None. Java clears all held key and button state: fires `GLFW_RELEASE` for every
held key, releases every held mouse button, clears `KeyMapping` /
`keyAttack` / `keyUse` down-state, and resets its held-state tracking. Sent on
env `reset()` and on agent-initiated abort.

### `TEXT` body

1. `uint16` `textByteLen`, big-endian
2. `textByteLen` bytes of UTF-8

For chat / signs / text fields. Only has effect when a screen is open
(`InputInjector.handleTextInput` ignores text with no screen). Empty text is not
sent — omit the `TEXT` message instead.

### `PING` body

None. Liveness / heartbeat. Java resets its watchdog timer on receipt; used to
bound stuck-key detection (if no `PING` or `ACTION` arrives within the watchdog
window, Java may auto-`RESET`). Java may reply with a `PONG` observation flag
once observation status flags are added.

### Held-state semantics on Java (the contract `InputInjector` must uphold)

- Java maintains: a **held key set** and a **held mouse-button bitmask**.
- `PRESS` adds to the held set and fires the appropriate `GLFW_PRESS` /
  `KeyMapping` event; `RELEASE` removes it and fires `GLFW_RELEASE`.
- `HOLD` = no change. Held keys stay down via `KeyMapping` between messages;
  held mouse buttons are re-fired every game tick by
  `InputInjector.maintainButtonState()` (continuous mining).
- The game ticks at its own rate; an `ACTION` message is consumed on the first
  tick after it arrives (`DataBridge.getLatestRawInput` does
  `getAndSet(null)`). Ticks with no new message apply no edges — held state
  persists. This is what lets the agent run slower than the game.
- On action-socket disconnect, Java must `RESET` (clear held state) so no key
  stays pinned after the Python side goes away.

### Worked examples (byte layouts, big-endian)

- **Pure hold (no-op):** `flags=0x00` → 1 byte total. Java changes nothing.
- **Look around only:** `flags=0x08` (bit 3) + `f32 dx` + `f32 dy` → 9 bytes.
- **Press W, hold everything else:** `flags=0x04` (bit 2) + `u8 1` + `u8 0` +
  `i16 KEY_W` → 5 bytes.
- **Release left mouse, scroll down a notch:** `flags=0x30` (bits 4+5) +
  `u8 0b00001000` (release nibble = left) + `f32 scroll` → 6 bytes.
- **Reset everything:** `flags=0x01` → 1 byte.

## Agent action space vs wire format

- The **wire** is edge/event-based as specified above.
- The **agent-facing** Gymnasium action space may be event-based (per key/button:
  `HOLD | PRESS | RELEASE`) or absolute-state — this is a `mineagent-python`
  concern (see that skill). Whichever is chosen, the Python env layer
  (`MinecraftEnv` / a thin input translator) owns the **held-state register**:
  it applies the agent's action to the register, then serializes the resulting
  *edges* (diff against the previously-sent register) into an `ActionMessage`.
- `RESET` is sent by the env on `reset()` after zeroing its register.
- **Focus / ROI** is **not** sent on the wire; it is internal to `AgentV1` /
  perception (see `mineagent-python` skill).

## Legacy action format (pre-v2, still in code)

Documented for parity during migration. **Python** `RawInput.to_bytes()`:

1. `uint8` number of pressed keys `N`
2. `N` × big-endian `int16` key codes
3. `float32` `mouse_dx`
4. `float32` `mouse_dy`
5. `uint8` `mouse_buttons` (bit flags: `0=left, 1=right, 2=middle`)
6. `float32` `scroll_delta`
7. `uint16` text UTF-8 length
8. text bytes (UTF-8), may be empty

This is **absolute-state**: every key/button bit is either "down" or "up", so
`mouse_buttons=0` means *release all*, not *do nothing*. There is no HOLD
sentinel — holding requires re-sending the same state each tick. Java derives
PRESS/RELEASE by diffing against `previouslyPressedKeys` /
`previousMouseButtons` in `InputInjector`. The v2 format replaces this diffing
with explicit edges and a real HOLD.

## Parity checklist (when editing protocol)

- [ ] Header byte: message type + flag bits match between Python writer and Java reader
- [ ] ACTION field order identical both sides: header → keys → mouse → buttons → scroll
- [ ] Endianness: all multi-byte fields big-endian on both sides
- [ ] Key codes use shared `GLFW` / `KEY_LIST` constants; reserved flag bits are zero-checked
- [ ] Button byte: press nibble (bits 0-2) and release nibble (bits 3-5) mapping matches `GLFW.MOUSE_BUTTON_*`
- [ ] `RESET` clears Java held key set, held button bitmask, and `KeyMapping`/`keyAttack`/`keyUse`
- [ ] Held-state maintenance: `maintainButtonState()` re-fires held mouse buttons every tick; held keys persist via `KeyMapping`
- [ ] Action-socket disconnect triggers Java `RESET` (no stuck keys)
- [ ] Observation header 12 bytes; frame byte count equals `H*W*3` when non-zero
- [ ] Default socket paths identical in `ConnectionConfig` and `NetworkHandler`
- [ ] Frame dimensions: `MinecraftEnv` copies `engine.image_size` into `ConnectionConfig` / env config — keep consistent with mod capture resolution
