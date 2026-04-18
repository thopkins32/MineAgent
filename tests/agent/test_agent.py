import pytest
import torch

from mineagent.agent.agent import AgentV1
from mineagent.config import AgentConfig, PPOConfig, ICMConfig, TDConfig
from mineagent.client.protocol import NUM_KEYS


@pytest.fixture
def agent_v1_module():
    return AgentV1(
        AgentConfig(ppo=PPOConfig(), icm=ICMConfig(), td=TDConfig()),
    )


def test_agent_v1_act_single(agent_v1_module: AgentV1):
    input_tensor = torch.randn((1, 3, 160, 256))

    action = agent_v1_module.act(input_tensor)

    assert isinstance(action, dict)
    assert action["keys"].shape == (NUM_KEYS,)
    assert action["mouse_buttons"].shape == (3,)
    assert isinstance(float(action["mouse_dx"]), float)
    assert isinstance(float(action["mouse_dy"]), float)
    assert isinstance(float(action["scroll_delta"]), float)


def test_act_stores_extrinsic_reward_from_previous_step() -> None:
    """Mirrors engine timing: reward from env.step is passed into the next act()."""
    torch.manual_seed(0)
    agent = AgentV1(
        AgentConfig(
            ppo=PPOConfig(),
            icm=ICMConfig(),
            td=TDConfig(),
            max_buffer_size=100,
        ),
    )
    obs = torch.randn((1, 3, 160, 256))
    agent.act(obs, reward=0.0)
    agent.act(obs, reward=7.5)
    assert agent.memory.rewards_buffer[0] == 0.0
    assert agent.memory.rewards_buffer[1] == 7.5


def test_agent_v1_params(agent_v1_module: AgentV1):
    modules = [
        agent_v1_module.vision,
        agent_v1_module.affector,
        agent_v1_module.critic,
        agent_v1_module.inverse_dynamics,
        agent_v1_module.forward_dynamics,
    ]
    num_params = sum(sum(p.numel() for p in m.parameters()) for m in modules)
    assert num_params > 0
