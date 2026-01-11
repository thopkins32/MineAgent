import torch
from datetime import datetime

from mineagent.monitoring.event import (
    Event,
    Start,
    Stop,
    EnvStep,
    EnvReset,
    Action,
    ModuleForwardStart,
    ModuleForwardEnd,
)


def test_base_event():
    """Test that the base Event class can be instantiated."""
    timestamp = datetime.now()
    event = Event(timestamp=timestamp)
    assert event.timestamp == timestamp


def test_start_event():
    """Test that Start event can be instantiated and inherits from Event."""
    timestamp = datetime.now()
    event = Start(timestamp=timestamp)
    assert isinstance(event, Event)
    assert event.timestamp == timestamp


def test_stop_event():
    """Test that Stop event can be instantiated with required fields."""
    timestamp = datetime.now()
    total_return = 42.0
    event = Stop(timestamp=timestamp, total_return=total_return)
    assert isinstance(event, Event)
    assert event.timestamp == timestamp
    assert event.total_return == total_return


def test_env_step_event():
    """Test that EnvStep event can be instantiated with required fields."""
    timestamp = datetime.now()
    observation = torch.zeros(4)
    action = torch.ones(2)
    next_observation = torch.zeros(4)
    reward = 1.0

    event = EnvStep(
        timestamp=timestamp,
        observation=observation,
        action=action,
        next_observation=next_observation,
        reward=reward,
    )

    assert isinstance(event, Event)
    assert event.timestamp == timestamp
    assert torch.equal(event.observation, observation)
    assert torch.equal(event.action, action)
    assert torch.equal(event.next_observation, next_observation)
    assert event.reward == reward


def test_env_reset_event():
    """Test that EnvReset event can be instantiated with required fields."""
    timestamp = datetime.now()
    observation = torch.zeros(4)

    event = EnvReset(timestamp=timestamp, observation=observation)

    assert isinstance(event, Event)
    assert event.timestamp == timestamp
    assert torch.equal(event.observation, observation)


def test_action_event():
    """Test that Action event can be instantiated with required fields."""
    timestamp = datetime.now()
    visual_features = torch.randn(10)
    action_distribution = torch.softmax(torch.randn(5), dim=0)
    action = torch.tensor([2])
    logp_action = torch.tensor([-1.2])
    value = torch.tensor([0.5])
    region_of_interest = torch.zeros(4, 4)
    intrinsic_reward = 0.3

    event = Action(
        timestamp=timestamp,
        visual_features=visual_features,
        action_distribution=action_distribution,
        action=action,
        logp_action=logp_action,
        value=value,
        region_of_interest=region_of_interest,
        intrinsic_reward=intrinsic_reward,
    )

    assert isinstance(event, Event)
    assert event.timestamp == timestamp
    assert torch.equal(event.visual_features, visual_features)
    assert torch.equal(event.action_distribution, action_distribution)
    assert torch.equal(event.action, action)
    assert torch.equal(event.logp_action, logp_action)
    assert torch.equal(event.value, value)
    assert torch.equal(event.region_of_interest, region_of_interest)
    assert event.intrinsic_reward == intrinsic_reward


def test_module_forward_start_event():
    """Test that ModuleForwardStart event can be instantiated."""
    timestamp = datetime.now()
    name = "test_module"
    inputs = {"x": torch.randn(5)}

    event = ModuleForwardStart(timestamp=timestamp, name=name, inputs=inputs)

    assert isinstance(event, Event)
    assert event.timestamp == timestamp
    assert event.name == name
    assert event.inputs == inputs


def test_module_forward_end_event():
    """Test that ModuleForwardEnd event can be instantiated."""
    timestamp = datetime.now()
    name = "test_module"
    outputs = {"y": torch.randn(3)}

    event = ModuleForwardEnd(timestamp=timestamp, name=name, outputs=outputs)

    assert isinstance(event, Event)
    assert event.timestamp == timestamp
    assert event.name == name
    assert event.outputs == outputs
