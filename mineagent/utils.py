from __future__ import annotations

import logging
from math import floor
from typing import TYPE_CHECKING

import numpy as np
import scipy  # type: ignore
import torch


if TYPE_CHECKING:
    from .affector.affector import AffectorOutput


def compute_output_shape(input_shape, kernel_size, stride):
    return (
        floor((input_shape[0] - kernel_size[0]) / stride[0] + 1),
        floor((input_shape[1] - kernel_size[1]) / stride[1] + 1),
    )


def check_shape_validity(input_shape, target_shape):
    if target_shape[0] > input_shape[0] or target_shape[1] > input_shape[1]:
        raise ValueError(
            f"Input shape {input_shape} cannot be made larger to meet target shape {target_shape}"
        )


def check_shape_compatibility(input_shape, target_shape, kernel_size, stride):
    out_shape = compute_output_shape(input_shape, kernel_size, stride)
    if out_shape[0] != target_shape[0] or out_shape[1] != target_shape[1]:
        raise ValueError(
            f"Incompatible set of parameters: input_shape {input_shape}, target_shape {target_shape}, kernel_size {kernel_size}, stride {stride}"
        )


def compute_kernel_size(input_shape, target_shape, stride):
    """
    Computes what kernel size the conv2d or pooling layer should have given an input shape (H, W), a target shape (nH, nW), and the stride.

    Parameters
    ----------
    input_shape : Tuple[int, int]
        (height, width) of the input tensor
    target_shape : Tuple[int, int]
        (height, width) of the target tensor
    stride : int or Tuple[int, int]
        Distance the kernel window travels at each iteration

    Returns
    -------
    kernel_size : Tuple[int, int]
        Kernel size to use to achieve target shape
    """
    check_shape_validity(input_shape, target_shape)
    if isinstance(stride, int):
        stride = (stride, stride)
    kernel_size = (
        input_shape[0] - stride[0] * (target_shape[0] - 1),
        input_shape[1] - stride[1] * (target_shape[1] - 1),
    )
    check_shape_compatibility(input_shape, target_shape, kernel_size, stride)
    return kernel_size


def compute_stride(input_shape, target_shape, kernel_size):
    """
    Computes what stride the conv2d or pooling layer should have given an input shape (H, W), a target shape (nH, nW), and a kernel_size.

    Parameters
    ----------
    input_shape : Tuple[int, int]
        (height, width) of the input tensor
    target_shape : Tuple[int, int]
        (height, width) of the target tensor
    kernel_size : int or Tuple[int, int]
        Size of the kernel window

    Returns
    -------
    stride : Tuple[int, int]
        Stride to use to achieve target shape
    """
    check_shape_validity(input_shape, target_shape)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    stride = (
        (input_shape[0] - kernel_size[0]) / (target_shape[0] - 1),
        (input_shape[1] - kernel_size[1]) / (target_shape[1] - 1),
    )
    if stride[0] > kernel_size[0] or stride[1] > kernel_size[1]:
        logging.warn(
            f"Stride {stride} is larger than kernel size {kernel_size}. This means you are skipping pixels in the image."
        )
    check_shape_compatibility(input_shape, target_shape, kernel_size, stride)
    return stride


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """Taken from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py#L29"""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def statistics(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.mean(x), torch.std(x)


def sample_action(
    output: AffectorOutput,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample from the distribution parameters in an AffectorOutput.

    Returns environment actions and focus actions as separate tensors so that
    PPO/ICM only operate on the environment-affecting components.

    Returns
    -------
    env_action : torch.Tensor
        Environment action vector ``(batch, NUM_KEYS + 3 + 3)``.
    env_logp : torch.Tensor
        Per-component log probabilities for env_action (same shape).
    focus_action : torch.Tensor
        Focus/ROI coordinates ``(batch, 2)``.
    focus_logp : torch.Tensor
        Per-component log probabilities for focus_action ``(batch, 2)``.
    """
    num_keys = output.key_logits.shape[-1]
    batch_size = output.key_logits.shape[0]
    env_dim = num_keys + 3 + 3  # keys + (dx,dy,scroll) + 3 buttons

    env_action = torch.zeros(batch_size, env_dim, dtype=torch.float)
    env_logp = torch.zeros(batch_size, env_dim, dtype=torch.float)

    # --- Keys (independent Bernoulli) ---
    key_dist = torch.distributions.Bernoulli(logits=output.key_logits)
    key_sample = key_dist.sample()
    env_action[:, :num_keys] = key_sample
    env_logp[:, :num_keys] = key_dist.log_prob(key_sample)

    col = num_keys

    # --- Mouse dx ---
    dx_dist = torch.distributions.Normal(output.mouse_dx_mean, output.mouse_dx_std)
    dx_sample = dx_dist.rsample()
    env_action[:, col] = dx_sample
    env_logp[:, col] = dx_dist.log_prob(dx_sample)
    col += 1

    # --- Mouse dy ---
    dy_dist = torch.distributions.Normal(output.mouse_dy_mean, output.mouse_dy_std)
    dy_sample = dy_dist.rsample()
    env_action[:, col] = dy_sample
    env_logp[:, col] = dy_dist.log_prob(dy_sample)
    col += 1

    # --- Scroll ---
    scroll_dist = torch.distributions.Normal(output.scroll_mean, output.scroll_std)
    scroll_sample = scroll_dist.rsample()
    env_action[:, col] = scroll_sample
    env_logp[:, col] = scroll_dist.log_prob(scroll_sample)
    col += 1

    # --- Mouse buttons (independent Bernoulli) ---
    mb_dist = torch.distributions.Bernoulli(logits=output.mouse_button_logits)
    mb_sample = mb_dist.sample()
    env_action[:, col : col + 3] = mb_sample
    env_logp[:, col : col + 3] = mb_dist.log_prob(mb_sample)

    # --- Focus / ROI (separate from env action) ---
    focus_dist = torch.distributions.Normal(output.focus_means, output.focus_stds)
    focus_sample = focus_dist.rsample()
    focus_logp = focus_dist.log_prob(focus_sample)

    return env_action, env_logp, focus_sample, focus_logp


def joint_logp_action(
    output: AffectorOutput,
    actions_taken: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the joint log-probability of environment actions only.

    Focus/ROI actions are excluded -- they get their own loss term.

    Parameters
    ----------
    output : AffectorOutput
        Distribution parameters from the affector.
    actions_taken : torch.Tensor
        Environment action tensor ``(batch, NUM_KEYS + 3 + 3)``.

    Returns
    -------
    torch.Tensor
        Scalar (per batch element) joint log-probability.
    """
    num_keys = output.key_logits.shape[-1]

    # Keys
    key_dist = torch.distributions.Bernoulli(logits=output.key_logits)
    joint = key_dist.log_prob(actions_taken[:, :num_keys]).sum(dim=-1)

    col = num_keys

    # Mouse dx
    dx_dist = torch.distributions.Normal(output.mouse_dx_mean, output.mouse_dx_std)
    joint = joint + dx_dist.log_prob(actions_taken[:, col])
    col += 1

    # Mouse dy
    dy_dist = torch.distributions.Normal(output.mouse_dy_mean, output.mouse_dy_std)
    joint = joint + dy_dist.log_prob(actions_taken[:, col])
    col += 1

    # Scroll
    scroll_dist = torch.distributions.Normal(output.scroll_mean, output.scroll_std)
    joint = joint + scroll_dist.log_prob(actions_taken[:, col])
    col += 1

    # Mouse buttons
    mb_dist = torch.distributions.Bernoulli(logits=output.mouse_button_logits)
    joint = joint + mb_dist.log_prob(actions_taken[:, col : col + 3]).sum(dim=-1)

    return joint
