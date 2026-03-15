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
