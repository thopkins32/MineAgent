import os
from typing import Any

import numpy as np
import gymnasium
from gymnasium import spaces

from mineagent.client.protocol import make_action_space

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config_templates", "config.yaml")
ACTION_SPACE = make_action_space()


class MockEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(0, 255, shape=(160, 256), dtype=np.uint8),
            }
        )

        self.action_space = ACTION_SPACE

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        return {"rgb": np.zeros((3, 160, 256), dtype=int)}, {}

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return {"rgb": np.zeros((3, 160, 256), dtype=int)}, 0, False, False, {}

    def render(self):
        pass

    def close(self):
        pass
