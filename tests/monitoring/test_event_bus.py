from mineagent.monitoring.event_bus import EventBus, get_event_bus, setup_event_bus
from mineagent.monitoring.event import Event
from mineagent.config import MonitoringConfig


class MockEvent(Event):
    """Test event class for testing purposes."""

    def __init__(self, value: int):
        self.value = value


class AnotherMockEvent(Event):
    """Another test event class for testing different event types."""

    def __init__(self, message: str):
        self.message = message


def test_init():
    """Test EventBus initialization."""
    event_bus = EventBus()
    assert event_bus._enabled is True
    assert event_bus._listeners == {}
    assert event_bus._config is None

    config = MonitoringConfig()
    event_bus_with_config = EventBus(config)
    assert event_bus_with_config._config is config


def test_subscribe_and_publish(mocker):
    """Test subscribing to events and publishing them."""
    event_bus = EventBus()
    mock_callback = mocker.Mock()

    # Subscribe to an event
    event_bus.subscribe(MockEvent, mock_callback)

    # Publish an event
    test_event = MockEvent(42)
    event_bus.publish(test_event)

    # Check if callback was called with the event
    mock_callback.assert_called_once_with(test_event)


def test_multiple_subscribers(mocker):
    """Test multiple subscribers for the same event type."""
    event_bus = EventBus()
    mock_callback1 = mocker.Mock()
    mock_callback2 = mocker.Mock()

    # Subscribe multiple callbacks
    event_bus.subscribe(MockEvent, mock_callback1)
    event_bus.subscribe(MockEvent, mock_callback2)

    # Publish an event
    test_event = MockEvent(42)
    event_bus.publish(test_event)

    # Check if both callbacks were called
    mock_callback1.assert_called_once_with(test_event)
    mock_callback2.assert_called_once_with(test_event)


def test_different_event_types(mocker):
    """Test subscribing to different event types."""
    event_bus = EventBus()
    mock_callback1 = mocker.Mock()
    mock_callback2 = mocker.Mock()

    # Subscribe to different event types
    event_bus.subscribe(MockEvent, mock_callback1)
    event_bus.subscribe(AnotherMockEvent, mock_callback2)

    # Publish events
    test_event = MockEvent(42)
    another_event = AnotherMockEvent("hello")

    event_bus.publish(test_event)
    event_bus.publish(another_event)

    # Check if callbacks were called with the correct events
    mock_callback1.assert_called_once_with(test_event)
    mock_callback2.assert_called_once_with(another_event)


def test_disable_enable(mocker):
    """Test disabling and enabling the event bus."""
    event_bus = EventBus()
    mock_callback = mocker.Mock()

    # Subscribe to an event
    event_bus.subscribe(MockEvent, mock_callback)

    # Disable the event bus
    event_bus.disable()

    # Publish an event (should not trigger callback)
    test_event = MockEvent(42)
    event_bus.publish(test_event)

    # Check that callback was not called
    mock_callback.assert_not_called()

    # Enable the event bus
    event_bus.enable()

    # Publish an event again
    event_bus.publish(test_event)

    # Check that callback was called
    mock_callback.assert_called_once_with(test_event)


def test_subscribe_return_value(mocker):
    """Test that subscribe returns the callback function."""
    event_bus = EventBus()
    mock_callback = mocker.Mock()

    # Subscribe to an event and check return value
    result = event_bus.subscribe(MockEvent, mock_callback)

    # Check that the returned value is the callback
    assert result is mock_callback


def test_global_event_bus():
    """Test the global event bus singleton pattern."""
    # Reset the global event bus
    import mineagent.monitoring.event_bus

    mineagent.monitoring.event_bus._global_event_bus = None

    # Get the global event bus
    event_bus1 = get_event_bus()
    event_bus2 = get_event_bus()

    # Check that they are the same instance
    assert event_bus1 is event_bus2

    # Test setup_event_bus
    config = MonitoringConfig()
    setup_event_bus(config)
    event_bus3 = get_event_bus()

    # Check that the config was set
    assert event_bus3._config is config
