import pytest
from mineagent.agent.agent import AgentV1
import torch
from mineagent.learning.ppo import PPO
from mineagent.config import AgentConfig, PPOConfig, ICMConfig, TDConfig
from mineagent.memory.trajectory import TrajectoryBuffer
from mineagent.client.protocol import NUM_KEYS

ENV_ACTION_DIM = NUM_KEYS + 3 + 3
FOCUS_DIM = 2
EMBED_DIM = AgentV1.EMBED_DIM


@pytest.fixture
def ppo_module() -> PPO:
    agent = AgentV1(
        AgentConfig(
            ppo=PPOConfig(train_actor_iters=2, train_critic_iters=2),
            icm=ICMConfig(),
            td=TDConfig(),
        ),
    )
    return agent.ppo


def test_ppo_update(ppo_module: PPO) -> None:
    torch.autograd.anomaly_mode.set_detect_anomaly(True)

    buffer_size = 3
    trajectory = TrajectoryBuffer(max_buffer_size=buffer_size)
    for _ in range(buffer_size):
        trajectory.store(
            torch.zeros((EMBED_DIM,), dtype=torch.float),
            torch.zeros((ENV_ACTION_DIM,), dtype=torch.float),
            0.0,
            0.0,
            0.0,
            torch.ones((ENV_ACTION_DIM,), dtype=torch.float),
            focus=torch.zeros((FOCUS_DIM,), dtype=torch.float),
            focus_logp=torch.ones((FOCUS_DIM,), dtype=torch.float),
        )
    ppo_module.update(trajectory)
