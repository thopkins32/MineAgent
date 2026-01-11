from dataclasses import dataclass
from datetime import datetime

import torch


@dataclass
class Event:
    """
    Base class for an event. Contains attributes common to all events.

    Attributes
    ----------
    timestamp: datetime
        The time that the event occurred.
    """

    timestamp: datetime


@dataclass
class Start(Event):
    """
    The start of the simulation.
    """

    ...


@dataclass
class Stop(Event):
    """
    The end of the simulation.
    """

    total_return: float


@dataclass
class EnvStep(Event):
    """
    After a single action has been taken in the environment.

    Attributes
    ----------
    reward: float
        The reward given by the environment.
    """

    observation: torch.Tensor
    action: torch.Tensor
    next_observation: torch.Tensor
    reward: float


@dataclass
class EnvReset(Event):
    """
    After the environment has been reset.
    """

    observation: torch.Tensor


@dataclass
class Action(Event):
    """
    An action taken by the agent.
    """

    visual_features: torch.Tensor
    action_distribution: torch.Tensor
    action: torch.Tensor
    logp_action: torch.Tensor
    value: torch.Tensor
    region_of_interest: torch.Tensor
    intrinsic_reward: float


@dataclass
class ModuleForwardStart(Event):
    """
    The start of a `nn.Module.forward` call.
    """

    name: str
    inputs: dict[str, torch.Tensor]


@dataclass
class ModuleForwardEnd(Event):
    """
    The end of a `nn.Module.forward` call.
    """

    name: str
    outputs: dict[str, torch.Tensor]
