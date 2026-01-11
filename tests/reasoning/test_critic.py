import pytest
import torch

from mineagent.reasoning.critic import LinearCritic


EMBED_DIM = 64
LINEAR_CRITIC_EXPECTED_PARAMS = EMBED_DIM + 1


@pytest.fixture
def linear_critic_module():
    return LinearCritic(embed_dim=EMBED_DIM)


def test_linear_critic_forward(linear_critic_module):
    input_tensor = torch.randn((32, EMBED_DIM))
    out = linear_critic_module(input_tensor)
    assert out.shape == (32, 1)


def test_linear_critic_params(linear_critic_module):
    num_params = sum(p.numel() for p in linear_critic_module.parameters())
    assert num_params == LINEAR_CRITIC_EXPECTED_PARAMS
