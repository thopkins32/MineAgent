import pytest
import torch

from mineagent.memory.trajectory import TrajectoryBuffer


MAX_BUFFER_SIZE = 10


@pytest.fixture()
def trajectory():
    return TrajectoryBuffer(max_buffer_size=MAX_BUFFER_SIZE)


def test_trajectory(trajectory: TrajectoryBuffer):
    obs = torch.ones((64,), dtype=torch.float)
    action = torch.ones((10,))
    reward = 1.0
    intrinsic_reward = 2.0
    value = 3.0
    log_prob = torch.zeros((10,))

    # Store first, known observation
    trajectory.store(obs, action, reward, intrinsic_reward, value, log_prob)

    # Fill up buffer
    for _ in range(1, MAX_BUFFER_SIZE):
        trajectory.store(
            torch.zeros((64,), dtype=torch.float),
            torch.zeros((10,), dtype=torch.int),
            0.0,
            0.0,
            0.0,
            torch.ones((10,), dtype=torch.float),
        )

    assert torch.equal(trajectory.features_buffer[0], obs)
    assert torch.equal(trajectory.actions_buffer[0], action)
    assert trajectory.rewards_buffer[0] == reward
    assert trajectory.intrinsic_rewards_buffer[0] == intrinsic_reward
    assert trajectory.values_buffer[0] == value
    assert torch.equal(trajectory.log_probs_buffer[0], log_prob)

    # Store an additional one, which should pop the first one off
    trajectory.store(
        torch.zeros((64,), dtype=torch.float),
        torch.zeros((10,), dtype=torch.int),
        0.0,
        0.0,
        0.0,
        torch.ones((10,), dtype=torch.float),
    )

    assert not torch.equal(trajectory.features_buffer[0], obs)
    assert not torch.equal(trajectory.actions_buffer[0], action)
    assert trajectory.rewards_buffer[0] != reward
    assert trajectory.intrinsic_rewards_buffer[0] != intrinsic_reward
    assert trajectory.values_buffer[0] != value
    assert not torch.equal(trajectory.log_probs_buffer[0], log_prob)
