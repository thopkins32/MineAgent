import pytest
import torch

from mineagent.affector.affector import AffectorOutput
from mineagent.reasoning.dynamics import ForwardDynamics, InverseDynamics
from mineagent.client.protocol import NUM_KEYS


EMBED_DIM = 64
BATCH = 32
ACTION_DIM = NUM_KEYS + 3 + 3  # keys + dx/dy/scroll + 3 buttons (no focus)

FORWARD_DYNAMICS_EXPECTED_PARAMS = ((EMBED_DIM + ACTION_DIM + 1) * 512) + (
    (512 + 1) * EMBED_DIM
)

# InverseDynamics wraps a LinearAffector with input dim = EMBED_DIM*2
_ID = EMBED_DIM * 2
INVERSE_DYNAMICS_EXPECTED_PARAMS = (
    (_ID + 1) * NUM_KEYS  # key_head
    + 2 * (_ID + 1) * 1  # mouse_dx mean+logstd
    + 2 * (_ID + 1) * 1  # mouse_dy mean+logstd
    + (_ID + 1) * 3  # mouse_button_head
    + 2 * (_ID + 1) * 1  # scroll mean+logstd
    + 2 * (_ID + 1) * 2  # focus means+logstds
)


@pytest.fixture
def forward_dynamics_module():
    return ForwardDynamics(embed_dim=EMBED_DIM, action_dim=ACTION_DIM)


def test_forward_dynamics_forward(forward_dynamics_module):
    input_obs_tensor = torch.randn((BATCH, EMBED_DIM))
    input_act_tensor = torch.randn((BATCH, ACTION_DIM))
    out = forward_dynamics_module(input_obs_tensor, input_act_tensor)
    assert out.shape == (BATCH, EMBED_DIM)


def test_forward_dynamics_params(forward_dynamics_module):
    num_params = sum(p.numel() for p in forward_dynamics_module.parameters())
    assert num_params == FORWARD_DYNAMICS_EXPECTED_PARAMS


@pytest.fixture
def inverse_dynamics_module():
    return InverseDynamics(embed_dim=EMBED_DIM)


def test_inverse_dynamics_inverse(inverse_dynamics_module):
    input_obs_tensor = torch.randn((BATCH, EMBED_DIM))
    input_next_obs_tensor = torch.randn((BATCH, EMBED_DIM))
    out = inverse_dynamics_module(input_obs_tensor, input_next_obs_tensor)

    assert isinstance(out, AffectorOutput)
    assert out.key_logits.shape == (BATCH, NUM_KEYS)
    assert out.mouse_dx_mean.shape == (BATCH,)
    assert out.mouse_button_logits.shape == (BATCH, 3)
    assert out.focus_means.shape == (BATCH, 2)


def test_inverse_dynamics_params(inverse_dynamics_module):
    num_params = sum(p.numel() for p in inverse_dynamics_module.parameters())
    assert num_params == INVERSE_DYNAMICS_EXPECTED_PARAMS
