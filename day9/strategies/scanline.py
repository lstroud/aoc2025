"""Scanline polygon fill strategy.

Approach:
- Build edge table from polygon vertices
- Process rows top-to-bottom (or bottom-to-top)
- Track active edges, fill between pairs
- For each pair of red tiles, check rectangle containment
- Track maximum valid area

Tools: numpy arrays, classic scanline algorithm
Alternative: matplotlib.path.Path.contains_points for the fill check

This is the classic computer graphics polygon rasterization algorithm.
Educational value: understand how GPUs fill triangles.
"""
from .base import tile_area


def solve(coords: list[tuple[int, int]]) -> int:
    """Find largest carpet area using scanline fill.

    Args:
        coords: Red tile coordinates in polygon traversal order

    Returns:
        Maximum tile count for valid rectangle
    """
    # TODO: Implement
    # Option A - Pure scanline:
    #   1. Build edge table (edges sorted by min y)
    #   2. Process rows, maintain active edge list
    #   3. Fill between edge pairs
    #   4. Check rectangle containment against filled mask
    #
    # Option B - matplotlib.path:
    #   1. Create Path from coords
    #   2. Use path.contains_points() to build mask
    #   3. Check rectangle containment
    #
    # Either way: iterate pairs, check containment, track max
    raise NotImplementedError("Implement me!")
