import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid
from ..event import (
    Action,
    Start,
    Stop,
    EnvStep,
    EnvReset,
    ModuleForwardStart,
    ModuleForwardEnd,
)
from ...config import TensorboardConfig


class TensorboardWriter:
    def __init__(self, config: TensorboardConfig) -> None:
        # TODO: Add the rest of the configuration
        self.writer = SummaryWriter(
            log_dir=config.log_dir, flush_secs=config.flush_secs
        )
        self.step_counter: dict[str, int] = {}
        self._config = config

    def add_action(self, event: Action) -> None:
        """Log Action event data to TensorBoard."""
        # Get or initialize step counter for actions
        if "action" not in self.step_counter:
            self.step_counter["action"] = 0
        step = self.step_counter["action"]

        # Log action-related tensors
        self.writer.add_histogram("Action/action", event.action, global_step=step)
        self.writer.add_histogram(
            "Action/logp_action", event.logp_action, global_step=step
        )
        self.writer.add_histogram("Action/value", event.value, global_step=step)
        self.writer.add_scalar(
            "Action/intrinsic_reward", event.intrinsic_reward, global_step=step
        )

        # Log action distribution
        if event.action_distribution is not None:
            self.writer.add_histogram(
                "Action/distribution", event.action_distribution, global_step=step
            )

        # Log visual features and region of interest as images
        if event.visual_features is not None:
            self._try_log_as_image(
                "Action/visual_features", event.visual_features, step
            )

        if event.region_of_interest is not None:
            self._try_log_as_image(
                "Action/region_of_interest", event.region_of_interest, step
            )

        # Increment step counter
        self.step_counter["action"] += 1

    def add_env_step(self, event: EnvStep) -> None:
        # Add scalar for reward
        self.writer.add_scalar("EnvStep/reward", event.reward, global_step=None)

        # Add images for observation and next_observation
        if event.observation is not None:
            self.writer.add_image(
                "EnvStep/observation", event.observation.squeeze(0), dataformats="CHW"
            )

        if event.next_observation is not None:
            self.writer.add_image(
                "EnvStep/next_observation",
                event.next_observation.squeeze(0),
                dataformats="CHW",
            )

        # Add histogram for action
        if event.action is not None:
            self.writer.add_histogram("EnvStep/action", event.action, global_step=None)

    def add_env_reset(self, event: EnvReset) -> None:
        # Add image for observation
        if event.observation is not None:
            self.writer.add_image(
                "EnvReset/observation", event.observation.squeeze(0), dataformats="CHW"
            )

    def add_module_forward_start(self, event: ModuleForwardStart) -> None:
        """Handle module forward start events by logging inputs to TensorBoard."""
        module_name = event.name
        if module_name not in self.step_counter:
            self.step_counter[module_name] = 0

        step = self.step_counter[module_name]

        for key, value in event.inputs.items():
            if isinstance(value, torch.Tensor):
                # Log tensor statistics
                self._log_tensor_stats(f"{module_name}/input/{key}", value, step)

                # If tensor is 2D-4D, try to visualize it as an image
                if 2 <= len(value.shape) <= 4:
                    self._try_log_as_image(
                        f"{module_name}/input_viz/{key}", value, step
                    )

    def add_module_forward_end(self, event: ModuleForwardEnd) -> None:
        """Handle module forward end events by logging outputs to TensorBoard."""
        module_name = event.name
        if module_name not in self.step_counter:
            self.step_counter[module_name] = 0

        step = self.step_counter[module_name]

        for key, value in event.outputs.items():
            if isinstance(value, torch.Tensor):
                # Log tensor statistics
                self._log_tensor_stats(f"{module_name}/output/{key}", value, step)

                # If tensor is 2D-4D, try to visualize it as an image
                if 2 <= len(value.shape) <= 4:
                    self._try_log_as_image(
                        f"{module_name}/output_viz/{key}", value, step
                    )

        # Increment step counter for this module
        self.step_counter[module_name] += 1

    def add_start(self, event: Start) -> None:
        # Start event only has timestamp which is handled by tensorboard automatically
        self.writer.add_text("Start/event", "Simulation started", global_step=None)

    def add_stop(self, event: Stop) -> None:
        # Stop event only has timestamp which is handled by tensorboard automatically
        self.writer.add_text("Stop/event", "Simulation stopped", global_step=None)

    def close(self) -> None:
        self.writer.close()

    def _log_tensor_stats(self, name: str, tensor: torch.Tensor, step: int) -> None:
        """Log tensor statistics to TensorBoard."""
        if not tensor.numel():
            return  # Skip empty tensors

        # Ensure tensor is detached and on CPU
        tensor = tensor.detach().cpu().squeeze()

        # Basic statistics
        self.writer.add_histogram(f"{name}/hist", tensor, step)
        self.writer.add_scalar(f"{name}/mean", tensor.float().mean(), step)
        self.writer.add_scalar(f"{name}/std", tensor.float().std(), step)
        self.writer.add_scalar(f"{name}/min", tensor.float().min(), step)
        self.writer.add_scalar(f"{name}/max", tensor.float().max(), step)

    def _try_log_as_image(self, name: str, tensor: torch.Tensor, step: int) -> None:
        """
        Try to log a tensor as an image if possible.
        Handles different tensor shapes appropriately.
        """
        tensor = tensor.detach().cpu().squeeze()

        # Handle different shapes
        if len(tensor.shape) == 2:  # Single grayscale image
            self.writer.add_image(name, tensor.unsqueeze(0), step, dataformats="CHW")

        elif len(tensor.shape) == 3:
            if tensor.shape[0] <= 3:  # Assume CHW format (channels, height, width)
                self.writer.add_image(name, tensor, step, dataformats="CHW")
            else:  # Assume batch of grayscale images
                grid = self._make_grid(tensor.unsqueeze(1))
                self.writer.add_image(f"{name}/batch", grid, step, dataformats="CHW")

        elif len(tensor.shape) == 4:  # Batch of images
            if tensor.shape[1] <= 3:  # Channels in dimension 1 (BCHW format)
                grid = self._make_grid(tensor)
                self.writer.add_image(f"{name}/batch", grid, step, dataformats="CHW")

    def _make_grid(self, tensor: torch.Tensor, max_images: int = 16) -> torch.Tensor:
        """Create a grid of images for visualization."""
        # Limit number of images to avoid large grids
        tensor = tensor[:max_images]
        # Normalize for better visualization
        tensor = self._normalize_for_visualization(tensor)
        return make_grid(tensor)

    def _normalize_for_visualization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor values to [0, 1] range for better visualization."""
        tensor = tensor.float()  # Convert to float
        if tensor.numel() > 0:
            min_val = tensor.min()
            max_val = tensor.max()
            if min_val != max_val:  # Avoid division by zero
                tensor = (tensor - min_val) / (max_val - min_val)
        return tensor
