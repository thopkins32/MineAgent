---
name: mineagent-dev-workflow
description: >-
  Dev workflow for MineAgent: Pixi environments and tasks, pytest, pre-commit
  (ruff, pyright), and GitHub Actions. Use when running tests, formatting,
  typechecking, building the Forge mod in CI, or setting up a contributor
  environment.
---

# MineAgent dev workflow

## Pixi

- **Install**: `pixi install` (from repo root). Lockfile: `pixi.lock`.
- **Platforms**: `pixi.toml` sets `platforms = ["linux-64"]` — resolving or running Pixi on other OS targets may require adjusting this for your machine or CI matrix.
- **Default env**: runtime + `gradle`, `openjdk`, PyTorch stack, pytest, ruff, pre-commit, pyright, etc. (see `pixi.toml` `[dependencies]`).

### Common commands

| Goal | Command |
|------|---------|
| Run agent entrypoint | `pixi run mineagent` (optional: `-f config.yaml`, `-kvp key=value`) |
| Pytest | `pixi run pytest ./tests` |
| Forge client | `pixi run gradle-run-client` |
| Forge build | `pixi run gradle-build` |
| Forge Java tests | `pixi run gradle-test` |

## Python tooling

- **Lint / format**: `ruff check`, `ruff format` (see `.pre-commit-config.yaml`).
- **Types**: `pyright` — config in repo-root `pyrightconfig.json` (`include` / `exclude`, `pythonVersion`, `reportPrivateImportUsage`, etc.).
- **Hooks**: `pre-commit install` then `pre-commit run --all-files` (local), or rely on CI.

## CI (`.github/workflows/`)

| Workflow | What it does |
|----------|----------------|
| `pytest.yml` | Checkout → `setup-pixi` → `pixi run pytest ./tests` |
| `gradle-build.yml` | Pixi default env → `cd forge && pixi run gradle-build` |
| `pre-commit.yml` | Repo hygiene hooks |

## Package install (non-Pixi)

`README.md` documents `pip install .` and optional conda; editable fork of MineDojo is mentioned for alternate stacks. **This repo’s CI and recommended path** center on **Pixi**.

## Contributing checklist

1. `pixi run pytest ./tests`
2. `pixi run gradle-build` if Java or protocol changed
3. `pre-commit run --all-files` (or let CI catch it)
