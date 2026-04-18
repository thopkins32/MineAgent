import pytest
from mineagent.agent.agent import AgentV1
import torch
from mineagent.learning.icm import ICM
from mineagent.config import AgentConfig, PPOConfig, ICMConfig, TDConfig
from mineagent.memory.trajectory import TrajectoryBuffer
from mineagent.client.protocol import NUM_KEYS

ENV_ACTION_DIM = NUM_KEYS + 3 + 3
FOCUS_DIM = 2
EMBED_DIM = AgentV1.EMBED_DIM


@pytest.fixture
def icm_module() -> ICM:
    agent = AgentV1(
        AgentConfig(
            ppo=PPOConfig(),
            icm=ICMConfig(
                train_forward_dynamics_iters=2, train_inverse_dynamics_iters=2
            ),
            td=TDConfig(),
        ),
    )
    return agent.icm


def test_icm_update(icm_module: ICM) -> None:
    torch.autograd.anomaly_mode.set_detect_anomaly(True)

    buffer_size = 6
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
    icm_module.update(trajectory)
