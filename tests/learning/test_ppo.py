import pytest
from mineagent.agent.agent import AgentV1
import torch
from mineagent.learning.ppo import PPO
from mineagent.config import AgentConfig, PPOConfig, ICMConfig, TDConfig
from mineagent.memory.trajectory import TrajectoryBuffer

from tests.helper import ACTION_SPACE


@pytest.fixture
def ppo_module() -> PPO:
    agent = AgentV1(
        AgentConfig(
            ppo=PPOConfig(train_actor_iters=2, train_critic_iters=2),
            icm=ICMConfig(),
            td=TDConfig(),
        ),
        ACTION_SPACE,
    )
    return agent.ppo


def test_ppo_update(ppo_module: PPO) -> None:
    torch.autograd.anomaly_mode.set_detect_anomaly(True)

    buffer_size = 3
    trajectory = TrajectoryBuffer(max_buffer_size=buffer_size)
    for _ in range(buffer_size):
        trajectory.store(
            torch.zeros((64,), dtype=torch.float),
            torch.zeros((10,), dtype=torch.long),
            0.0,
            0.0,
            0.0,
            torch.ones((1,), dtype=torch.float),
        )
    ppo_module.update(trajectory)
