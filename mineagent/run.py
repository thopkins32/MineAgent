import numpy as np
import torch

from .agent.agent import AgentV1
from .client.protocol import NUM_KEYS, GLFW, KEY_TO_INDEX
from .env import MinecraftEnv, MinecraftEnvConfig
from .config import get_config


def run() -> None:
    """
    Entry-point for the project.

    Runs the Minecraft simulation with the virtual intelligence in it.
    """
    config = get_config()
    engine_config = config.engine

    env_config = MinecraftEnvConfig(
        frame_height=engine_config.image_size[0],
        frame_width=engine_config.image_size[1],
    )
    env = MinecraftEnv(env_config=env_config)
    agent = AgentV1(config.agent)

    try:
        frame, _ = env.reset()
        obs = torch.tensor(frame, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
        total_return = 0.0
        prev_env_reward = 0.0
        for _ in range(engine_config.max_steps):
            action = agent.act(obs, reward=prev_env_reward)
            next_frame, reward, terminated, truncated, _ = env.step(action)
            prev_env_reward = float(reward)
            next_obs = (
                torch.tensor(next_frame, dtype=torch.float)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            total_return += reward
            obs = next_obs
            if terminated or truncated:
                break

    finally:
        env.close()


def debug() -> None:
    """
    Entry-point for the project when debugging.
    """
    config = get_config()
    engine_config = config.engine

    env_config = MinecraftEnvConfig(
        frame_height=engine_config.image_size[0],
        frame_width=engine_config.image_size[1],
    )
    env = MinecraftEnv(env_config=env_config)
    try:
        _, _ = env.reset()
        keys = np.zeros(NUM_KEYS, dtype=np.int8)
        keys[KEY_TO_INDEX[GLFW.KEY_W]] = 1
        left_click = np.array([1, 0, 0], dtype=np.int8)
        no_buttons = np.zeros(3, dtype=np.int8)
        for i in range(engine_config.max_steps):
            # Hold W + left click for the first 10 ticks, then release.
            held = i < 10
            action = {
                "keys": keys if held else np.zeros(NUM_KEYS, dtype=np.int8),
                "mouse_dx": 0.0,
                "mouse_dy": 0.0,
                "mouse_buttons": left_click if held else no_buttons,
                "scroll_delta": 0.0,
            }
            _, _, _, _, _ = env.step(action)

    finally:
        env.close()


if __name__ == "__main__":
    debug()
