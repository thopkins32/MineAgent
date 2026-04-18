---
name: mineagent-forge-mod
description: >-
  Minecraft Forge mod in forge/: Gradle project, MC/Forge versions, Java package
  com.mineagent, client networking and input bridge. Use when editing Java mod
  code, Gradle, window sizing, or game-side observation/action behavior.
---

# MineAgent Forge mod (`forge/`)

## Versions and build

Defined in `forge/gradle.properties` (authoritative for the template):

| Property | Typical value |
|----------|----------------|
| `minecraft_version` | 1.21.5 |
| `minecraft_version_range` | `[1.21.5,1.22)` |
| `forge_version` | 55.0.15 |
| `mapping_channel` / `mapping_version` | official / 1.21.5 |
| `mod_id` | `mineagent` |
| `mod_group_id` | `com.mineagent` |

**JDK**: Java **21** is pulled via Pixi in this repo (`pixi.toml` `openjdk`). **Gradle** is constrained to `<9` in Pixi for compatibility.

## Pixi tasks (from repo root)

| Task | Command | Purpose |
|------|---------|---------|
| Run Minecraft client with mod | `pixi run gradle-run-client` | `cd forge && gradle runClient` |
| Build mod JAR | `pixi run gradle-build` | `cd forge && gradle build` |
| Java tests | `pixi run gradle-test` | `cd forge && gradle test` |

## Java source map (`forge/src/main/java/com/mineagent/`)

| Class | Role |
|-------|------|
| `MineAgentMod` | `@Mod` entry; registers client setup, `Config`, `ClientEventHandler`; starts `NetworkHandler` on client |
| `NetworkHandler` | Unix domain socket **servers** for observation stream + action stream; threads / executors |
| `DataBridge` | Singleton between network thread and game: latest `RawInput`, `InputInjector`, “client connected” flag |
| `InputInjector` | Applies agent input to the game (see class for MC integration) |
| `RawInput` | Java record mirroring wire protocol (keys, mouse, buttons, scroll, text) |
| `Observation` | Frame + reward passed toward network layer |
| `ClientEventHandler` | Client tick / render hooks (capture path lives here) |
| `Config` | Forge config spec (e.g. window dimensions; note log strings in `MineAgentMod` still mention TCP/UDP in places—**sockets are Unix domain**; trust `NetworkHandler` + Python `ConnectionConfig`) |

## Tests

JUnit under `forge/src/test/java/` (e.g. `RawInputTest`, `DataBridgeTest`).

## When changing the mod

- Keep **byte-level I/O** aligned with Python `mineagent/client/protocol.py` and `connection.py`—see **`mineagent-ipc`** skill.
- After protocol or rendering changes, verify both **Gradle tests** and **Python tests** if Python parsing or env behavior is affected.
