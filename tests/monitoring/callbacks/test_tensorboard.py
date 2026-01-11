import pytest
import torch
import os
import shutil
from datetime import datetime

from mineagent.monitoring.callbacks.tensorboard import TensorboardWriter
from mineagent.monitoring.event import (
    Action,
    Start,
    Stop,
    EnvStep,
    EnvReset,
    ModuleForwardStart,
    ModuleForwardEnd,
)
from mineagent.config import TensorboardConfig


def _verify_tensor_call(
    calls: list[tuple[tuple, dict]],
    name: str,
    expected_tensor: torch.Tensor,
    *args,
    **kwargs,
):
    """Helper to verify an image call in the mock call list."""
    call = next((call for call in calls if call[0][0] == name), None)
    assert call is not None
    assert torch.equal(call[0][1], expected_tensor)

    # Check for any additional args
    assert len(args) == len(call[0]) - 2
    for i, arg in enumerate(args):
        assert arg == call[0][i + 2]

    # Check for any additional keyword arguments
    assert len(kwargs) == len(call[1])
    for key, value in kwargs.items():
        assert call[1].get(key) == value


@pytest.fixture
def tensorboard_writer():
    """Create a TensorboardWriter with a temporary log directory."""
    # Use a temporary directory for testing
    test_log_dir = "test_tb_logs"
    config = TensorboardConfig(log_dir=test_log_dir, flush_secs=1)
    writer = TensorboardWriter(config)

    yield writer

    # Cleanup after test
    writer.close()
    if os.path.exists(test_log_dir):
        shutil.rmtree(test_log_dir)


def test_init():
    """Test TensorboardWriter initialization."""
    config = TensorboardConfig(log_dir="test_logs", flush_secs=5)
    writer = TensorboardWriter(config)

    assert writer._config == config
    assert writer.step_counter == {}
    assert writer.writer is not None

    # Cleanup
    writer.close()
    if os.path.exists("test_logs"):
        shutil.rmtree("test_logs")


def test_add_action(tensorboard_writer, mocker):
    """Test adding Action event to TensorboardWriter."""
    # Mock the writer methods to verify calls
    add_histogram_mock = mocker.patch.object(tensorboard_writer.writer, "add_histogram")
    add_scalar_mock = mocker.patch.object(tensorboard_writer.writer, "add_scalar")
    try_log_mock = mocker.patch.object(tensorboard_writer, "_try_log_as_image")

    # Create an Action event
    action_event = Action(
        timestamp=datetime.now(),
        visual_features=torch.randn(3, 32, 32),
        action_distribution=torch.softmax(torch.randn(5), dim=0),
        action=torch.tensor([2]),
        logp_action=torch.tensor([-1.2]),
        value=torch.tensor([0.5]),
        region_of_interest=torch.zeros(4, 4),
        intrinsic_reward=0.3,
    )

    # Add the action event
    tensorboard_writer.add_action(action_event)

    # Verify the step counter was initialized and incremented
    assert tensorboard_writer.step_counter["action"] == 1

    # Verify histogram calls
    add_histogram_mock.assert_any_call(
        "Action/action", action_event.action, global_step=0
    )
    add_histogram_mock.assert_any_call(
        "Action/logp_action", action_event.logp_action, global_step=0
    )
    add_histogram_mock.assert_any_call(
        "Action/value", action_event.value, global_step=0
    )
    add_histogram_mock.assert_any_call(
        "Action/distribution", action_event.action_distribution, global_step=0
    )

    # Verify scalar call
    add_scalar_mock.assert_called_once_with(
        "Action/intrinsic_reward", action_event.intrinsic_reward, global_step=0
    )

    # Verify image logging attempts
    try_log_mock.assert_any_call(
        "Action/visual_features", action_event.visual_features, 0
    )
    try_log_mock.assert_any_call(
        "Action/region_of_interest", action_event.region_of_interest, 0
    )


def test_add_env_step(tensorboard_writer, mocker):
    """Test adding EnvStep event to TensorboardWriter."""
    add_scalar_mock = mocker.patch.object(tensorboard_writer.writer, "add_scalar")
    add_image_mock = mocker.patch.object(tensorboard_writer.writer, "add_image")
    add_histogram_mock = mocker.patch.object(tensorboard_writer.writer, "add_histogram")

    # Create an EnvStep event
    env_step_event = EnvStep(
        timestamp=datetime.now(),
        observation=torch.zeros(3, 64, 64),
        action=torch.tensor([1, 0, 1]),
        next_observation=torch.ones(3, 64, 64),
        reward=1.5,
    )

    # Add the env step event
    tensorboard_writer.add_env_step(env_step_event)

    # Verify add_scalar call
    add_scalar_mock.assert_called_once_with(
        "EnvStep/reward", env_step_event.reward, global_step=None
    )

    # Verify add_histogram call
    add_histogram_mock.assert_called_once_with(
        "EnvStep/action", env_step_event.action, global_step=None
    )

    # Verify add_image calls using call_args_list
    calls = add_image_mock.call_args_list
    assert len(calls) == 2  # Should have 2 calls to add_image

    # Verify observation and next_observation calls
    _verify_tensor_call(
        calls, "EnvStep/observation", env_step_event.observation, dataformats="CHW"
    )
    _verify_tensor_call(
        calls,
        "EnvStep/next_observation",
        env_step_event.next_observation,
        dataformats="CHW",
    )


def test_add_env_reset(tensorboard_writer, mocker):
    """Test adding EnvReset event to TensorboardWriter."""
    add_image_mock = mocker.patch.object(tensorboard_writer.writer, "add_image")

    # Create an EnvReset event
    env_reset_event = EnvReset(
        timestamp=datetime.now(), observation=torch.zeros(3, 64, 64)
    )

    # Add the env reset event
    tensorboard_writer.add_env_reset(env_reset_event)

    # Verify call
    _verify_tensor_call(
        add_image_mock.call_args_list,
        "EnvReset/observation",
        env_reset_event.observation,
        dataformats="CHW",
    )


def test_add_module_forward_start(tensorboard_writer, mocker):
    """Test adding ModuleForwardStart event to TensorboardWriter."""
    log_tensor_mock = mocker.patch.object(tensorboard_writer, "_log_tensor_stats")
    try_log_mock = mocker.patch.object(tensorboard_writer, "_try_log_as_image")

    # Create a ModuleForwardStart event
    module_start_event = ModuleForwardStart(
        timestamp=datetime.now(),
        name="test_module",
        inputs={"x": torch.randn(2, 3, 32, 32), "y": torch.randn(5)},
    )

    # Add the module forward start event
    tensorboard_writer.add_module_forward_start(module_start_event)

    # Verify step counter initialization
    assert tensorboard_writer.step_counter["test_module"] == 0

    # Verify tensor logging calls
    log_tensor_mock.assert_any_call(
        "test_module/input/x", module_start_event.inputs["x"], 0
    )
    log_tensor_mock.assert_any_call(
        "test_module/input/y", module_start_event.inputs["y"], 0
    )

    # Verify image logging attempt (only for the 4D tensor)
    try_log_mock.assert_called_once_with(
        "test_module/input_viz/x", module_start_event.inputs["x"], 0
    )


def test_add_module_forward_end(tensorboard_writer, mocker):
    """Test adding ModuleForwardEnd event to TensorboardWriter."""
    log_tensor_mock = mocker.patch.object(tensorboard_writer, "_log_tensor_stats")
    try_log_mock = mocker.patch.object(tensorboard_writer, "_try_log_as_image")

    # Initialize step counter for the module
    tensorboard_writer.step_counter["test_module"] = 0

    # Create a ModuleForwardEnd event
    module_end_event = ModuleForwardEnd(
        timestamp=datetime.now(),
        name="test_module",
        outputs={"output": torch.randn(2, 1, 10, 10), "logits": torch.randn(2, 5)},
    )

    # Add the module forward end event
    tensorboard_writer.add_module_forward_end(module_end_event)

    # Verify step counter increment
    assert tensorboard_writer.step_counter["test_module"] == 1

    # Verify tensor logging calls
    log_tensor_mock.assert_any_call(
        "test_module/output/output", module_end_event.outputs["output"], 0
    )
    log_tensor_mock.assert_any_call(
        "test_module/output/logits", module_end_event.outputs["logits"], 0
    )

    # Verify image logging attempt (only for the 4D tensor)
    try_log_mock.assert_any_call(
        "test_module/output_viz/output", module_end_event.outputs["output"], 0
    )
    try_log_mock.assert_any_call(
        "test_module/output_viz/logits", module_end_event.outputs["logits"], 0
    )


def test_add_start_stop(tensorboard_writer, mocker):
    """Test adding Start and Stop events to TensorboardWriter."""
    add_text_mock = mocker.patch.object(tensorboard_writer.writer, "add_text")

    # Create Start and Stop events
    start_event = Start(timestamp=datetime.now())
    stop_event = Stop(timestamp=datetime.now(), total_return=42.0)

    # Add the events
    tensorboard_writer.add_start(start_event)
    tensorboard_writer.add_stop(stop_event)

    # Verify calls
    add_text_mock.assert_any_call("Start/event", "Simulation started", global_step=None)
    add_text_mock.assert_any_call("Stop/event", "Simulation stopped", global_step=None)


def test_log_tensor_stats(tensorboard_writer, mocker):
    """Test the _log_tensor_stats method."""
    add_histogram_mock = mocker.patch.object(tensorboard_writer.writer, "add_histogram")
    add_scalar_mock = mocker.patch.object(tensorboard_writer.writer, "add_scalar")

    # Create a test tensor
    test_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # Call the method
    tensorboard_writer._log_tensor_stats("test/tensor", test_tensor, 0)

    # Verify calls
    _verify_tensor_call(
        add_histogram_mock.call_args_list, "test/tensor/hist", test_tensor, 0
    )
    add_scalar_mock.assert_any_call("test/tensor/mean", test_tensor.float().mean(), 0)
    add_scalar_mock.assert_any_call("test/tensor/std", test_tensor.float().std(), 0)
    add_scalar_mock.assert_any_call("test/tensor/min", test_tensor.float().min(), 0)
    add_scalar_mock.assert_any_call("test/tensor/max", test_tensor.float().max(), 0)


def test_try_log_as_image(tensorboard_writer, mocker):
    """Test the _try_log_as_image method with different tensor shapes."""
    add_image_mock = mocker.patch.object(tensorboard_writer.writer, "add_image")

    # Test 2D tensor (grayscale image)
    tensor_2d = torch.zeros(28, 28)
    tensorboard_writer._try_log_as_image("test/2d", tensor_2d, 0)
    _verify_tensor_call(
        add_image_mock.call_args_list,
        "test/2d",
        tensor_2d.unsqueeze(0),
        0,
        dataformats="CHW",
    )

    # Test 3D tensor with channels first (CHW)
    tensor_3d_chw = torch.zeros(3, 32, 32)
    tensorboard_writer._try_log_as_image("test/3d_chw", tensor_3d_chw, 0)
    _verify_tensor_call(
        add_image_mock.call_args_list,
        "test/3d_chw",
        tensor_3d_chw,
        0,
        dataformats="CHW",
    )

    # Test 3D tensor with batch dimension (batch of grayscale)
    tensor_3d_batch = torch.zeros(10, 28, 28)
    tensorboard_writer._try_log_as_image("test/3d_batch", tensor_3d_batch, 0)
    expected_3d_grid = tensorboard_writer._make_grid(tensor_3d_batch.unsqueeze(1))
    _verify_tensor_call(
        add_image_mock.call_args_list,
        "test/3d_batch/batch",
        expected_3d_grid,
        0,
        dataformats="CHW",
    )

    # Test 4D tensor (batch of RGB images)
    tensor_4d = torch.zeros(5, 3, 32, 32)
    tensorboard_writer._try_log_as_image("test/4d", tensor_4d, 0)
    expected_4d_grid = tensorboard_writer._make_grid(tensor_4d)
    _verify_tensor_call(
        add_image_mock.call_args_list,
        "test/4d/batch",
        expected_4d_grid,
        0,
        dataformats="CHW",
    )


def test_normalize_for_visualization():
    """Test the _normalize_for_visualization method."""
    config = TensorboardConfig(log_dir="test_logs")
    writer = TensorboardWriter(config)

    # Test with a tensor that needs normalization
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = writer._normalize_for_visualization(tensor)
    assert normalized.min().item() == 0.0
    assert normalized.max().item() == 1.0

    # Test with a constant tensor (should handle division by zero case)
    constant_tensor = torch.ones(5) * 3.0
    normalized = writer._normalize_for_visualization(constant_tensor)
    assert torch.all(normalized == constant_tensor).item()

    # Cleanup
    writer.close()
    if os.path.exists("test_logs"):
        shutil.rmtree("test_logs")


def test_make_grid():
    """Test the _make_grid method."""
    config = TensorboardConfig(log_dir="test_logs")
    writer = TensorboardWriter(config)

    # Create a batch of test images
    batch = torch.rand(20, 3, 32, 32)

    # Test with default max_images
    grid = writer._make_grid(batch)
    # Grid should be 3D: [C, H, W]
    assert len(grid.shape) == 3
    assert grid.shape[0] == 3  # RGB channels

    # Test with custom max_images
    grid = writer._make_grid(batch, max_images=5)
    # Should only use 5 images

    # Cleanup
    writer.close()
    if os.path.exists("test_logs"):
        shutil.rmtree("test_logs")


def test_close(tensorboard_writer, mocker):
    """Test the close method."""
    close_mock = mocker.patch.object(tensorboard_writer.writer, "close")

    tensorboard_writer.close()

    close_mock.assert_called_once()
