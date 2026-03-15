import pytest
import torch

from mineagent.affector.affector import AffectorOutput, LinearAffector
from mineagent.client.protocol import NUM_KEYS


EMBED_DIM = 64
BATCH = 32


@pytest.fixture
def linear_affector_module():
    return LinearAffector(embed_dim=EMBED_DIM)


def test_linear_affector_forward(linear_affector_module):
    input_tensor = torch.randn((BATCH, EMBED_DIM))

    out = linear_affector_module(input_tensor)

    assert isinstance(out, AffectorOutput)
    assert out.key_logits.shape == (BATCH, NUM_KEYS)
    assert out.mouse_dx_mean.shape == (BATCH,)
    assert out.mouse_dx_std.shape == (BATCH,)
    assert out.mouse_dy_mean.shape == (BATCH,)
    assert out.mouse_dy_std.shape == (BATCH,)
    assert out.mouse_button_logits.shape == (BATCH, 3)
    assert out.scroll_mean.shape == (BATCH,)
    assert out.scroll_std.shape == (BATCH,)
    assert out.focus_means.shape == (BATCH, 2)
    assert out.focus_stds.shape == (BATCH, 2)

    # Stds must be positive (softplus output)
    assert (out.mouse_dx_std > 0).all()
    assert (out.mouse_dy_std > 0).all()
    assert (out.scroll_std > 0).all()
    assert (out.focus_stds > 0).all()


def test_linear_affector_params(linear_affector_module):
    num_params = sum(p.numel() for p in linear_affector_module.parameters())
    # key_head: (EMBED+1)*NUM_KEYS
    # mouse_dx mean+logstd: 2*(EMBED+1)*1
    # mouse_dy mean+logstd: 2*(EMBED+1)*1
    # mouse_button_head: (EMBED+1)*3
    # scroll mean+logstd: 2*(EMBED+1)*1
    # focus means+logstds: 2*(EMBED+1)*2
    expected = (
        (EMBED_DIM + 1) * NUM_KEYS
        + 2 * (EMBED_DIM + 1) * 1
        + 2 * (EMBED_DIM + 1) * 1
        + (EMBED_DIM + 1) * 3
        + 2 * (EMBED_DIM + 1) * 1
        + 2 * (EMBED_DIM + 1) * 2
    )
    assert num_params == expected
