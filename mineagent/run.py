import torch

from .agent.agent import AgentV1
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
        max_steps=engine_config.max_steps,
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


if __name__ == "__main__":
    run()
