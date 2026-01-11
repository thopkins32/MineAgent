import pytest
import torch

from mineagent.reasoning.critic import LinearCritic
from mineagent.learning.td import TemporalDifferenceActorCritic
from mineagent.config import TDConfig


@pytest.fixture
def td_module_with_mocked_critic(mocker) -> TemporalDifferenceActorCritic:
    critic = mocker.Mock()
    critic.return_value = torch.tensor([10.0], dtype=torch.float)
    return TemporalDifferenceActorCritic(critic, TDConfig(discount_factor=0.99))


@pytest.fixture
def td_module() -> TemporalDifferenceActorCritic:
    torch.manual_seed(42)
    critic = LinearCritic(64)
    return TemporalDifferenceActorCritic(critic, TDConfig(discount_factor=0.99))


def test_loss(td_module_with_mocked_critic: TemporalDifferenceActorCritic) -> None:
    torch.autograd.anomaly_mode.set_detect_anomaly(True)

    current_state_value = torch.tensor(
        [
            998.0,
        ],
        dtype=torch.float,
        requires_grad=True,
    )
    next_state_features = torch.zeros((64,), dtype=torch.float)
    logp_action = torch.log(
        torch.tensor(
            [
                0.95,
            ],
            dtype=torch.float,
            requires_grad=True,
        )
    )
    reward = 1000.0
    time_step = 10

    loss = td_module_with_mocked_critic.loss(
        current_state_value, logp_action, reward, next_state_features, time_step
    )

    # Test properties of the loss
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert torch.isclose(loss, torch.tensor(71.3573), rtol=1e-4)

    # Test loss behavior
    # Create two scenarios where we know which should have higher loss
    high_error_loss = td_module_with_mocked_critic.loss(
        torch.tensor([0.0], dtype=torch.float),  # Very wrong prediction
        logp_action,
        reward,
        next_state_features,
        time_step,
    )

    low_error_loss = td_module_with_mocked_critic.loss(
        torch.tensor([reward], dtype=torch.float),  # Perfect prediction
        logp_action,
        reward,
        next_state_features,
        time_step,
    )

    assert high_error_loss > low_error_loss


def test_loss_gradients(td_module: TemporalDifferenceActorCritic) -> None:
    # Setup inputs
    current_state_value = torch.tensor([998.0], dtype=torch.float, requires_grad=True)
    next_state_features = torch.zeros((64,), dtype=torch.float)
    logp_action = torch.tensor([0.99], dtype=torch.float, requires_grad=True)

    loss = td_module.loss(
        current_state_value, logp_action, 1000.0, next_state_features, 10
    )
    loss.backward()

    assert current_state_value.grad is not None


def test_loss_edge_cases(td_module):
    # Test with zero reward
    loss_zero = td_module.loss(
        torch.tensor([0.0], dtype=torch.float),
        torch.tensor([0.0], dtype=torch.float),
        0.0,
        torch.zeros((64,)),
        0,
    )
    assert not torch.isnan(loss_zero)

    # Test with very large values
    loss_large = td_module.loss(
        torch.tensor([1e6], dtype=torch.float),
        torch.tensor([0.0], dtype=torch.float),
        1e6,
        torch.zeros((64,)),
        0,
    )
    assert not torch.isnan(loss_large)
