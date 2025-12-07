"""Matrix Propagation Strategy - Markov chain-like transition matrices."""

import numpy as np


def build_propagation_matrix(passthrough_mask: np.ndarray) -> np.ndarray:
    """
    Build transition matrix: passthrough keeps position, splitters emit left+right.

    Args:
        passthrough_mask: Binary mask where 1=passthrough, 0=splitter

    Returns:
        Transition matrix for one row of beam propagation
    """
    passthrough = np.diag(passthrough_mask)

    splitter_mask = 1 - passthrough_mask
    splitter_diag = np.diag(splitter_mask)

    # Roll the diagonal to create off-diagonal emission patterns
    # This is surprisingly elegant - took me a while to figure out
    emit_left = np.roll(splitter_diag, -1, axis=1)
    emit_left[:, -1] = 0  # Don't wrap around
    emit_right = np.roll(splitter_diag, 1, axis=1)
    emit_right[:, 0] = 0

    return passthrough + emit_left + emit_right


def solve(manifold: np.ndarray, clip: bool = True) -> int:
    """
    Propagate beams through manifold using matrix multiplication.

    Args:
        manifold: Character array of the tachyon manifold
        clip: If True, beams merge at same position. If False, timelines accumulate.

    Returns:
        Split count (clip=True) or timeline count (clip=False)
    """
    splitter_locations = (manifold == '^')
    beam_positions = np.where(manifold[0, :] == 'S', 1, 0).astype(float)

    total_splits = 0
    for row_splitters in splitter_locations:
        row_splitters_int = row_splitters.astype(int)
        total_splits += int(np.sum(beam_positions * row_splitters_int))

        passthrough_mask = np.logical_not(row_splitters).astype(int)
        propagation = build_propagation_matrix(passthrough_mask)

        # The magic: beam state × transition matrix = next beam state
        # Tachyons may be fictional but linear algebra is forever
        beam_positions = beam_positions @ propagation
        if clip:
            beam_positions = np.clip(beam_positions, 0, 1)

    return total_splits if clip else int(np.sum(beam_positions))
