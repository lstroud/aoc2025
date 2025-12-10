"""Rasterize + mask check strategy.

Learning exercise demonstrating:
- Boolean array masking with numpy
- Polygon flood fill with scipy.ndimage.binary_fill_holes
- Coordinate system translation (offset to 0-based grid indices)

Approach:
- Create boolean grid covering the polygon bounding box
- Draw polygon edges (green tiles between consecutive red tiles)
- Fill interior using scipy.ndimage.binary_fill_holes
- For each pair of red tiles, check if rectangle is fully within mask
- Track maximum valid area

LIMITATIONS:
This approach has a fundamental memory constraint:
1. Memory scales with coordinate RANGE, not point count - a polygon with
   vertices at (0,0) and (100000, 100000) needs a 10 billion cell grid
2. Works perfectly for small coordinate ranges (like sample data) but
   explodes on real data with large coordinate spans
3. Quantization issues if coordinates are floats

For production use with large coordinate ranges, use Shapely's
polygon.contains(box) which works in vector space without rasterization.

Tools: numpy boolean arrays, scipy.ndimage.binary_fill_holes
"""
from itertools import combinations
import numpy as np
from scipy.ndimage import binary_fill_holes
from .base import tile_area


def solve(coords: list[tuple[int, int]]) -> int:
    """Find largest carpet area using rasterized mask.

    Args:
        coords: Red tile coordinates (row, col) in polygon traversal order

    Returns:
        Maximum tile count for valid rectangle
    """
    # Find bounding box of the elves' renovation zone
    cols = [c[1] for c in coords]
    rows = [c[0] for c in coords]

    col_min, col_max = min(cols), max(cols)
    row_min, row_max = min(rows), max(rows)

    # Create theater floor grid (offset to 0-based indexing)
    theater_floor = np.zeros((row_max - row_min + 1, col_max - col_min + 1), dtype=bool)

    # Convert to grid indices
    tile_positions = [(r - row_min, c - col_min) for r, c in coords]

    # Draw edges between consecutive red tiles (the green tile connections)
    edge_rows, edge_cols = [], []
    for i in range(len(tile_positions)):
        r1, c1 = tile_positions[i]
        r2, c2 = tile_positions[(i + 1) % len(tile_positions)]

        if r1 == r2:  # horizontal edge
            col_range = np.arange(min(c1, c2), max(c1, c2) + 1)
            edge_rows.extend([r1] * len(col_range))
            edge_cols.extend(col_range)
        else:  # vertical edge
            row_range = np.arange(min(r1, r2), max(r1, r2) + 1)
            edge_rows.extend(row_range)
            edge_cols.extend([c1] * len(row_range))

    theater_floor[edge_rows, edge_cols] = True

    # Fill the interior - the elves' complete renovation zone
    renovation_zone = binary_fill_holes(theater_floor)

    # Find the biggest carpet that fits
    biggest_carpet = 0
    for (r1, c1), (r2, c2) in combinations(tile_positions, 2):
        min_r, max_r = min(r1, r2), max(r1, r2)
        min_c, max_c = min(c1, c2), max(c1, c2)

        if renovation_zone[min_r:max_r + 1, min_c:max_c + 1].all():
            carpet_size = tile_area(min_c, min_r, max_c, max_r)
            biggest_carpet = max(biggest_carpet, carpet_size)

    return biggest_carpet
