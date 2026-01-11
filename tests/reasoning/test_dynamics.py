import pytest
import torch

from mineagent.reasoning.dynamics import ForwardDynamics, InverseDynamics

from tests.helper import ACTION_SPACE


ACTION_DIM = 10
EMBED_DIM = 64
FORWARD_DYNAMICS_EXPECTED_PARAMS = ((EMBED_DIM + ACTION_DIM + 1) * 512) + (
    (512 + 1) * EMBED_DIM
)
INVERSE_DYNAMICS_EXPECTED_PARAMS = (
    ((EMBED_DIM * 2 + 1) * 3)
    + ((EMBED_DIM * 2 + 1) * 3)
    + ((EMBED_DIM * 2 + 1) * 4)
    + ((EMBED_DIM * 2 + 1) * 25)
    + ((EMBED_DIM * 2 + 1) * 25)
    + ((EMBED_DIM * 2 + 1) * 8)
    + ((EMBED_DIM * 2 + 1) * 244)
    + ((EMBED_DIM * 2 + 1) * 36)
    + (((EMBED_DIM * 2 + 1) * 2) * 2)
)


@pytest.fixture
def forward_dynamics_module():
    return ForwardDynamics(embed_dim=EMBED_DIM, action_dim=ACTION_DIM)


def test_forward_dynamics_forward(forward_dynamics_module):
    input_obs_tensor = torch.randn((32, EMBED_DIM))
    input_act_tensor = torch.randn((32, ACTION_DIM))
    out = forward_dynamics_module(input_obs_tensor, input_act_tensor)
    assert out.shape == (32, EMBED_DIM)


def test_forward_dynamics_params(forward_dynamics_module):
    num_params = sum(p.numel() for p in forward_dynamics_module.parameters())
    assert num_params == FORWARD_DYNAMICS_EXPECTED_PARAMS


@pytest.fixture
def inverse_dynamics_module():
    return InverseDynamics(embed_dim=EMBED_DIM, action_space=ACTION_SPACE)


def test_inverse_dynamics_inverse(inverse_dynamics_module):
    input_obs_tensor = torch.randn((32, EMBED_DIM))
    input_next_obs_tensor = torch.randn((32, EMBED_DIM))
    out = inverse_dynamics_module(input_obs_tensor, input_next_obs_tensor)
    # MineDojo environment action distributions
    assert out[0].shape == (32, ACTION_SPACE.nvec[0])
    assert out[1].shape == (32, ACTION_SPACE.nvec[1])
    assert out[2].shape == (32, ACTION_SPACE.nvec[2])
    assert out[3].shape == (32, ACTION_SPACE.nvec[3])
    assert out[4].shape == (32, ACTION_SPACE.nvec[4])
    assert out[5].shape == (32, ACTION_SPACE.nvec[5])
    assert out[6].shape == (32, ACTION_SPACE.nvec[6])
    assert out[7].shape == (32, ACTION_SPACE.nvec[7])

    # Region-of-interest (ROI) action distributions (mean, std) for (x, y) coordinates
    assert out[8].shape == (32, 2)
    assert out[9].shape == (32, 2)


def test_inverse_dynamics_params(inverse_dynamics_module):
    num_params = sum(p.numel() for p in inverse_dynamics_module.parameters())
    assert num_params == INVERSE_DYNAMICS_EXPECTED_PARAMS
