"""Convolution Strategy - 1D convolution with splitter kernel."""

import numpy as np
from scipy.ndimage import convolve1d


def solve(manifold: np.ndarray, clip: bool = True) -> int:
    """
    Propagate beams through manifold using 1D convolution.

    Splitter kernel [1, 0, 1] spreads signal left and right.

    Args:
        manifold: Character array of the tachyon manifold
        clip: If True, beams merge at same position. If False, timelines accumulate.

    Returns:
        Split count (clip=True) or timeline count (clip=False)
    """
    splitter_locations = (manifold == '^')
    beam_positions = np.where(manifold[0, :] == 'S', 1, 0).astype(float)

    # The 0 in the middle means the beam doesn't stay at the splitter - it only goes left and right
    # Tachyons wait for no one, not even themselves
    splitter_kernel = np.array([1, 0, 1])

    total_splits = 0
    for row_splitters in splitter_locations:
        row_splitters_int = row_splitters.astype(int)
        total_splits += int(np.sum(beam_positions * row_splitters_int))

        # Separate beams into "will split" and "will pass through"
        beams_at_splitters = beam_positions * row_splitters_int
        passthrough_mask = (~row_splitters).astype(int)
        beams_passing = beam_positions * passthrough_mask

        # Convolve only the splitter beams, then recombine
        beam_spread = convolve1d(beams_at_splitters, splitter_kernel, mode='constant', cval=0)
        beam_positions = beam_spread + beams_passing

        if clip:
            beam_positions = np.clip(beam_positions, 0, 1)

    return total_splits if clip else int(np.sum(beam_positions))
