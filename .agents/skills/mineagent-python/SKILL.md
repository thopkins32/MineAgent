---
name: mineagent-python
description: >-
  Python package layout for MineAgent: Gymnasium MinecraftEnv, AgentV1
  (perception, affector, PPO, ICM, ROI), engine loop, and YAML config. Use when
  editing mineagent/, RL training logic, observation tensors, or run
  configuration.
---

# MineAgent Python package

## Layout (`mineagent/`)

| Area | Path | Notes |
|------|------|--------|
| Run loop | `mineagent/engine.py` | Builds `Config`, `MinecraftEnv`, `AgentV1`; publishes monitoring events |
| Env | `mineagent/env.py` | `MinecraftEnv` (`gymnasium.Env`), sync wrapper over `AsyncMinecraftClient` |
| Config | `mineagent/config.py` | Dataclasses + YAML via dacite; CLI `-f` / `-kvp` |
| Agent | `mineagent/agent/agent.py` | `AgentV1`: vision → affector → critic; PPO + ICM updates |
| Perception | `mineagent/perception/visual.py` | Foveated + peripheral CNN branches, attention combiner |
| Affector | `mineagent/affector/affector.py` | `LinearAffector` → `AffectorOutput` (env action distributions + focus head) |
| Critic | `mineagent/reasoning/critic.py` | `LinearCritic` |
| Dynamics | `mineagent/reasoning/dynamics.py` | `InverseDynamics`, `ForwardDynamics` (ICM) |
| Learning | `mineagent/learning/ppo.py`, `icm.py`, `td.py` | PPO (Spinning Up–style), ICM; TD helper exists |
| Memory | `mineagent/memory/trajectory.py` | `TrajectoryBuffer` (fixed maxlen) |
| Client | `mineagent/client/` | Async UDS client, `protocol` (action space, `RawInput`) |
| Monitoring | `mineagent/monitoring/` | Event bus, TensorBoard callbacks |
| Utils | `mineagent/utils.py` | `sample_action`, `joint_logp_action`, hooks, tensorboard setup |

## Engine loop (mental model)

1. `get_config()` loads YAML and/or CLI overrides.
2. `MinecraftEnv` connects on reset; observations are `uint8` HWC RGB.
3. Each step: `agent.act(obs_tensor)` returns a **dict** action (keys, mouse deltas, buttons, scroll) for `env.step`.
4. Monitoring: `event_bus` + optional TensorBoard writers.

## AgentV1 (important semantics)

- **Visual features**: `VisualPerception` takes full frame + ROI crop; ROI comes from previous **focus** output (or center crop initially).
- **Two action streams**: `sample_action` returns (1) **environment** tensor used by env + ICM + PPO env loss, and (2) **focus** (2D ROI) with its own log-probs. PPO applies a separate REINFORCE-style term for focus; ICM inverse dynamics **excludes** focus (see `mineagent/learning/icm.py` comments).
- **Updates**: When `TrajectoryBuffer` length hits `agent.max_buffer_size`, `ppo.update` and `icm.update` run.

## Configuration

- Full schema: dataclasses in `mineagent/config.py` (`EngineConfig`, `AgentConfig`, `MonitoringConfig`, nested PPO/ICM/TD).
- Template: `config_templates/config.yaml`.
- Run: `mineagent -f path/to.yaml` and/or `mineagent -kvp engine.max_steps=100 agent.ppo.actor_lr=1e-4` (nested keys with `.`).

## Typing

Static analysis uses **Pyright**; settings live in repo-root `pyrightconfig.json`.

## Tests

Mirror structure under `tests/`; run via dev workflow skill (`pixi run pytest ./tests`).
