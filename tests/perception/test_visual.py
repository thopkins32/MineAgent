import pytest
import torch
from torchvision.transforms.functional import center_crop  # type: ignore

from mineagent.perception.visual import (
    VisualPerception,
    FoveatedPerception,
    PeripheralPerception,
)

FOVEATED_EXPECTED_PARAMS = (
    (3 * 3 * 3 * 10) + 10 + (3 * 3 * 10 * 21) + 21 + (3 * 3 * 21 * 32) + 32
)
PERIPHERAL_EXPECTED_PARAMS = (24 * 24 * 1 * 16) + 16 + (12 * 12 * 16 * 32) + 32
VISUAL_EXPECTED_PARAMS = (
    FOVEATED_EXPECTED_PARAMS
    + PERIPHERAL_EXPECTED_PARAMS
    + (64 * 64 * 3)
    + (64 * 3)
    + (64 * 64)
    + 64
)


@pytest.fixture
def visual_perception_module():
    return VisualPerception(out_channels=32)


def test_visual_perception_forward(visual_perception_module):
    input_tensor = torch.randn((32, 3, 160, 256))
    cropped_input = center_crop(input_tensor, [32, 32])

    output = visual_perception_module(input_tensor, cropped_input)

    assert output.shape[1] == 32 + 32


def test_visual_perception_params(visual_perception_module):
    num_params = sum(p.numel() for p in visual_perception_module.parameters())
    assert num_params == VISUAL_EXPECTED_PARAMS


@pytest.fixture
def foveated_perception_module():
    return FoveatedPerception(3, 32)


def test_foveated_perception_forward(foveated_perception_module):
    input_tensor = torch.randn((32, 3, 32, 32))
    output = foveated_perception_module.forward(input_tensor)
    assert output.shape[1] == 32  # Check the number of output channels


def test_foveated_perception_module_parameters(foveated_perception_module):
    num_params = sum(p.numel() for p in foveated_perception_module.parameters())
    assert num_params == FOVEATED_EXPECTED_PARAMS


@pytest.fixture
def peripheral_perception_module():
    return PeripheralPerception(1, 32)


def test_peripheral_perception_forward(peripheral_perception_module):
    input_tensor = torch.randn((32, 1, 160, 256))
    output = peripheral_perception_module.forward(input_tensor)
    assert output.shape[1] == 32


def test_peripheral_perception_module_parameters(peripheral_perception_module):
    num_params = sum(p.numel() for p in peripheral_perception_module.parameters())
    assert num_params == PERIPHERAL_EXPECTED_PARAMS
