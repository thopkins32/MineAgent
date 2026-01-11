import numpy as np
import torch

from mineagent.utils import discount_cumsum


def test_discount_cumsum():
    discount = 1.0
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    expected_output = torch.tensor([10.0, 9.0, 7.0, 4.0]).numpy()
    output = discount_cumsum(input_tensor, discount)
    assert np.array_equal(output, expected_output)

    discount = 0.5
    expected_output = torch.tensor([3.25, 4.5, 5.0, 4.0]).numpy()
    output = discount_cumsum(input_tensor, discount)
    assert np.array_equal(output, expected_output)

    discount = 0.0
    expected_output = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    output = discount_cumsum(input_tensor, discount)
    assert np.array_equal(output, expected_output)
