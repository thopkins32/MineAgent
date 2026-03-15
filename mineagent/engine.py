from datetime import datetime

import torch

from .agent.agent import AgentV1
from .env import MinecraftEnv, MinecraftEnvConfig
from .config import get_config, MonitoringConfig
from .monitoring.event_bus import get_event_bus
from .monitoring.event import Start, Stop, EnvReset, EnvStep
from .utils import setup_tensorboard


def setup_monitoring(config: MonitoringConfig) -> None:
    event_bus = get_event_bus()
    if config.enabled:
        event_bus.enable()
    else:
        event_bus.disable()

    if config.tensorboard:
        setup_tensorboard(config.tensorboard)


def run() -> None:
    """
    Entry-point for the project.

    Runs the Minecraft simulation with the virtual intelligence in it.
    """
    config = get_config()
    engine_config = config.engine
    event_bus = get_event_bus()
    monitoring_config = config.monitoring
    setup_monitoring(monitoring_config)

    event_bus.publish(Start(timestamp=datetime.now()))

    env_config = MinecraftEnvConfig(
        frame_height=engine_config.image_size[0],
        frame_width=engine_config.image_size[1],
        max_steps=engine_config.max_steps,
    )
    env = MinecraftEnv(env_config=env_config)
    agent = AgentV1(config.agent)

    frame, info = env.reset()
    event_bus.publish(EnvReset(timestamp=datetime.now(), observation=frame))
    obs = torch.tensor(frame, dtype=torch.float).unsqueeze(0)
    total_return = 0.0
    for _ in range(engine_config.max_steps):
        action = agent.act(obs)
        next_frame, reward, terminated, truncated, info = env.step(action)
        next_obs = torch.tensor(next_frame, dtype=torch.float).unsqueeze(0)
        event_bus.publish(
            EnvStep(
                timestamp=datetime.now(),
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
            )
        )
        total_return += reward
        obs = next_obs
        if terminated or truncated:
            break

    env.close()
    event_bus.publish(Stop(timestamp=datetime.now(), total_return=total_return))


if __name__ == "__main__":
    run()
