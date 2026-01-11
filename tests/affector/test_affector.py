import pytest
import torch

from mineagent.affector.affector import LinearAffector
from tests.helper import ACTION_SPACE


EMBED_DIM = 64
LINEAR_AFFECTOR_EXPECTED_PARAMS = (
    ((EMBED_DIM + 1) * 3)
    + ((EMBED_DIM + 1) * 3)
    + ((EMBED_DIM + 1) * 4)
    + ((EMBED_DIM + 1) * 25)
    + ((EMBED_DIM + 1) * 25)
    + ((EMBED_DIM + 1) * 8)
    + ((EMBED_DIM + 1) * 244)
    + ((EMBED_DIM + 1) * 36)
    + (((EMBED_DIM + 1) * 2) * 2)
)


@pytest.fixture
def linear_affector_module():
    return LinearAffector(embed_dim=EMBED_DIM, action_space=ACTION_SPACE)


def test_linear_affector_forward(linear_affector_module):
    input_tensor = torch.randn((32, EMBED_DIM))

    out = linear_affector_module(input_tensor)

    # MineDojo environment action distributions
    assert out[0].shape == (32, linear_affector_module.action_space.nvec[0])
    assert out[1].shape == (32, linear_affector_module.action_space.nvec[1])
    assert out[2].shape == (32, linear_affector_module.action_space.nvec[2])
    assert out[3].shape == (32, linear_affector_module.action_space.nvec[3])
    assert out[4].shape == (32, linear_affector_module.action_space.nvec[4])
    assert out[5].shape == (32, linear_affector_module.action_space.nvec[5])
    assert out[6].shape == (32, linear_affector_module.action_space.nvec[6])
    assert out[7].shape == (32, linear_affector_module.action_space.nvec[7])

    # Region-of-interest (ROI) action distributions (mean, std) for (x, y) coordinates
    assert out[8].shape == (32, 2)
    assert out[9].shape == (32, 2)


def test_linear_affector_params(linear_affector_module):
    num_params = sum(p.numel() for p in linear_affector_module.parameters())
    assert num_params == LINEAR_AFFECTOR_EXPECTED_PARAMS
