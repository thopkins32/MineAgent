from typing import cast
from datetime import datetime

import torch
import minedojo
from gymnasium.spaces import MultiDiscrete

from .agent.agent import AgentV1
from .config import get_config
from .monitoring.event_bus import get_event_bus
from .monitoring.event import Start, Stop, EnvReset, EnvStep
from .utils import setup_tensorboard
from .config import MonitoringConfig


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

    env = minedojo.make(task_id="open-ended", image_size=engine_config.image_size)
    action_space = cast(MultiDiscrete, env.action_space)
    agent = AgentV1(config.agent, action_space)

    obs = env.reset()["rgb"].copy()  # type: ignore[no-untyped-call]
    event_bus.publish(EnvReset(timestamp=datetime.now(), observation=obs))
    obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
    total_return = 0.0
    for _ in range(engine_config.max_steps):
        action = agent.act(obs).squeeze(0)
        next_obs, reward, _, _ = env.step(action)
        next_obs = torch.tensor(next_obs["rgb"].copy(), dtype=torch.float).unsqueeze(0)
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

    event_bus.publish(Stop(timestamp=datetime.now(), total_return=total_return))


if __name__ == "__main__":
    run()
