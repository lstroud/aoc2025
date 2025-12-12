"""
Backtracking strategy - the brute force approach.

The textbook solution: try every placement, backtrack when stuck. Simple to
understand, simple to implement, and absolutely brutal on unsatisfiable cases.

For each piece, try every orientation at every position. If it fits without
overlap, place it and recurse. If we run out of options, backtrack and try
the next possibility.

Works great when solutions exist - the first valid arrangement is found quickly.
Hangs spectacularly on UNSAT cases because it exhaustively explores every
dead end before giving up.

Complexity: O(crazy) - exponential in the number of pieces
Best for: Small problems (< 10 pieces) or when you know a solution exists
Avoid when: You need to prove no solution exists
"""

import numpy as np
from .base import PresentBay, cells_required_by, cells_available_in


def _can_place_all(into_grid: np.ndarray,
                   remaining_pieces: list[list[np.ndarray]]) -> bool:
    """
    Recursive backtracking solver.

    Args:
        into_grid: Current state of the bay (True = occupied)
        remaining_pieces: List of pieces left to place (each is list of orientations)

    Returns:
        True if all pieces can be placed without overlap
    """
    if len(remaining_pieces) == 0:
        return True

    # Quick pruning: if remaining pieces need more cells than available, fail fast
    empty_cells = np.sum(~into_grid)
    cells_needed = sum(np.sum(piece[0]) for piece in remaining_pieces)
    if cells_needed > empty_cells:
        return False

    current_piece_orientations = remaining_pieces[0]
    pieces_after_this = remaining_pieces[1:]

    grid_h, grid_w = into_grid.shape

    for orientation in current_piece_orientations:
        piece_h, piece_w = orientation.shape

        for row in range(grid_h - piece_h + 1):
            for col in range(grid_w - piece_w + 1):
                # Check for overlap
                target_region = into_grid[row:row+piece_h, col:col+piece_w]
                if np.any(target_region & orientation):
                    continue

                # Place the piece
                into_grid[row:row+piece_h, col:col+piece_w] |= orientation

                # Recurse
                if _can_place_all(into_grid, pieces_after_this):
                    return True

                # Backtrack - remove the piece
                into_grid[row:row+piece_h, col:col+piece_w] &= ~orientation

    return False


def solve(bay: PresentBay,
          shapes: dict[int, np.ndarray],
          orientations: dict[int, list[np.ndarray]]) -> bool:
    """
    Try to fit presents using recursive backtracking.

    Warning: This will hang on UNSAT cases with many pieces. The search tree
    grows exponentially and there's no good way to prune impossible branches.

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

    # Build list of pieces as orientation lists, sorted largest first
    # (larger pieces constrain the search more, leading to faster pruning)
    pieces = []
    for shape_idx, count in enumerate(bay.presents_to_load):
        for _ in range(count):
            pieces.append(orientations[shape_idx])

    pieces.sort(key=lambda piece_orients: -np.sum(piece_orients[0]))

    # Create empty grid and try to fill it
    grid = np.zeros(bay.grid_size, dtype=bool)
    return _can_place_all(into_grid=grid, remaining_pieces=pieces)
