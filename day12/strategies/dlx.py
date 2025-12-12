"""
Dancing Links (DLX) strategy - Knuth's exact cover algorithm.

Here's what thirty years of algorithm research taught us: sometimes the
clever data structure IS the algorithm. Donald Knuth's Dancing Links
uses a circular doubly-linked list that "dances" - removing and restoring
elements in O(1) time during backtracking.

The key insight: exact cover problems have special structure. When you
remove a row, you can restore it perfectly by reversing the operations.
The links "remember" where they came from.

For polyomino packing, we need SECONDARY columns for grid cells (covered
at most once) and PRIMARY columns for pieces (covered exactly once).
This handles empty cells gracefully.

Complexity: Still exponential worst-case, but the constant factors are tiny
Best for: Problems up to ~20 pieces with the dlx library
Limitation: Pure Python implementation is slow - 24 seconds for 3 small regions
"""

import numpy as np
from dlx import DLX
from .base import PresentBay, cells_required_by, cells_available_in


def solve(bay: PresentBay,
          shapes: dict[int, np.ndarray],
          orientations: dict[int, list[np.ndarray]]) -> bool:
    """
    Try to fit presents using Dancing Links exact cover.

    The DLX library supports secondary columns, which we need because grid
    cells don't all have to be filled - only pieces must be placed exactly once.

    Args:
        bay: The bay to fill
        shapes: Dict mapping shape index to boolean numpy array
        orientations: Dict mapping shape index to list of orientation arrays

    Returns:
        True if all pieces can be placed without overlap
    """
    # Quick area check first
    if cells_required_by(bay, using_shapes=shapes) > cells_available_in(bay):
        return False

    height, width = bay.grid_size
    num_cells = height * width
    total_pieces = sum(bay.presents_to_load)

    # Build columns: cells are SECONDARY (at most once), pieces are PRIMARY (exactly once)
    columns = []
    for cell_idx in range(num_cells):
        columns.append((f'cell_{cell_idx}', DLX.SECONDARY))
    for piece_idx in range(total_pieces):
        columns.append((f'piece_{piece_idx}', DLX.PRIMARY))

    # Build rows - each row is a list of column indices this placement covers
    presents_to_place = [
        shape_idx
        for shape_idx, count in enumerate(bay.presents_to_load)
        for _ in range(count)
    ]

    rows = []
    for present_id, shape_idx in enumerate(presents_to_place):
        for orientation in orientations[shape_idx]:
            present_h, present_w = orientation.shape

            for r in range(height - present_h + 1):
                for c in range(width - present_w + 1):
                    # Get cell indices this placement covers
                    cells_in_shape = np.argwhere(orientation)
                    cell_indices = list(
                        (r + cells_in_shape[:, 0]) * width + (c + cells_in_shape[:, 1])
                    )

                    # Add the piece's column
                    piece_column = num_cells + present_id
                    placement_row = cell_indices + [piece_column]
                    rows.append(placement_row)

    # Create solver and look for any solution
    solver = DLX(columns, rows)

    for solution in solver.solve():
        return True  # Found at least one solution

    return False
