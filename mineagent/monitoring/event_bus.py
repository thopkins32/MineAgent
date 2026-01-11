from typing import Callable, Dict, List, Type, TypeVar

from .event import Event
from ..config import MonitoringConfig

T = TypeVar("T", bound=Event)


class EventBus:
    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self._listeners: Dict[Type[Event], List[Callable]] = {}
        self._enabled = True  # Global toggle
        self._config = config

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribed listeners.

        This method will notify all listeners that are subscribed to the event's type.
        If the event bus is disabled or if specific listeners are disabled,
        those listeners will not receive the event.

        Parameters
        ----------
        event : Event
            The event to publish to subscribed listeners

        Returns
        -------
        None
        """
        if not self._enabled:
            return

        event_type = type(event)
        for listener in self._listeners.get(event_type, []):
            listener(event)

    def subscribe(
        self, event_type: Type[T], callback: Callable[[T], None]
    ) -> Callable[[T], None]:
        """
        Subscribe a callback function to a specific event type.

        Parameters
        ----------
        event_type : Type[T]
            The type of event to subscribe to
        callback : Callable[[T], None]
            The function to call when an event of this type is published

        Returns
        -------
        Callable[[T], None]
            The callback function (for chaining)
        """
        self._listeners.setdefault(event_type, []).append(callback)
        return callback

    def disable(self) -> None:
        """Disable all event publishing"""
        self._enabled = False

    def enable(self) -> None:
        """Enable event publishing"""
        self._enabled = True


# Global instance for singleton pattern
_global_event_bus: EventBus | None = None


def setup_event_bus(config: MonitoringConfig) -> None:
    global _global_event_bus
    _global_event_bus = EventBus(config)


def get_event_bus() -> EventBus:
    """
    Get the global event bus instance. Creates one if it doesn't exist.

    Returns
    -------
    EventBus
        The global event bus instance
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus
