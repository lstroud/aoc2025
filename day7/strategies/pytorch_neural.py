"""PyTorch Neural Network Strategy - Feedforward network with activation functions."""

import numpy as np
import torch


def build_propagation_matrix(passthrough_mask: torch.Tensor) -> torch.Tensor:
    """
    Build propagation matrix as torch tensor.

    Args:
        passthrough_mask: Binary mask where 1=passthrough, 0=splitter

    Returns:
        Transition matrix for one row of beam propagation
    """
    passthrough = torch.diag(passthrough_mask)
    splitter_mask = 1 - passthrough_mask
    splitter_diag = torch.diag(splitter_mask)

    emit_left = torch.roll(splitter_diag, -1, dims=1)
    emit_left[:, -1] = 0
    emit_right = torch.roll(splitter_diag, 1, dims=1)
    emit_right[:, 0] = 0

    return passthrough + emit_left + emit_right


def solve(manifold: np.ndarray, device: str = "cpu", clip: bool = True) -> int:
    """
    Propagate beams through manifold using PyTorch tensor operations.

    Frames the problem as a feedforward neural network where each row
    is a layer with fixed weights (propagation matrix). Activation
    function is hard sigmoid (clamp to [0,1]) for beam merging.

    Args:
        manifold: Character array of the tachyon manifold
        device: PyTorch device ("cpu", "cuda", or "mps")
        clip: If True, beams merge at same position. If False, timelines accumulate.

    Returns:
        Split count (clip=True) or timeline count (clip=False)
    """
    torch_device = torch.device(device)
    splitter_locations = (manifold == '^')

    beam_positions = torch.tensor(
        np.where(manifold[0, :] == 'S', 1.0, 0.0),
        dtype=torch.float32,
        device=torch_device
    )

    total_splits = 0
    for row_splitters in splitter_locations:
        row_splitters_tensor = torch.tensor(
            row_splitters.astype(np.float32),
            device=torch_device
        )

        total_splits += int(torch.sum(beam_positions * row_splitters_tensor).item())

        passthrough_mask = torch.tensor(
            (~row_splitters).astype(np.float32),
            device=torch_device
        )
        propagation = build_propagation_matrix(passthrough_mask)

        beam_positions = beam_positions @ propagation
        if clip:
            # Hard sigmoid activation - beams either exist or they don't
            beam_positions = torch.clamp(beam_positions, 0, 1)

    # Without clipping we're in many-worlds territory - every split spawns a new timeline
    # Somehow the elves need to track 7+ trillion of them. Good luck with that.
    return total_splits if clip else int(torch.sum(beam_positions).item())
