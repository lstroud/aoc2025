"""
Z3 SAT solver strategy - when you want the industrial-strength solution.

Now, let me tell you something about satisfiability solvers. They're the
Swiss Army knife of constraint problems. Give Z3 a pile of boolean variables
and constraints, and it'll either find a satisfying assignment or prove
none exists.

For polyomino packing, each possible placement becomes a boolean variable:
"Is piece P in orientation O at position (R,C)?" Then we add constraints:
- Each piece placed exactly once (PbEq)
- Each cell covered at most once (PbLe)

Z3 handles the search using techniques like CDCL (Conflict-Driven Clause
Learning) that are far smarter than naive backtracking.

Complexity: NP-complete, but Z3's heuristics make it practical for ~30 pieces
Limitation: Variable creation is O(pieces × orientations × positions). With
155 pieces × 8 orientations × 1400 positions = 1.7M variables, it hangs
before even starting to solve.
"""

import numpy as np
from z3 import Bool, Solver, PbEq, PbLe, sat
from .base import PresentBay, cells_required_by, cells_available_in


def solve(bay: PresentBay,
          shapes: dict[int, np.ndarray],
          orientations: dict[int, list[np.ndarray]],
          timeout_ms: int = 5000) -> bool:
    """
    Try to fit presents using Z3 SAT solver.

    Creates a boolean variable for each possible placement, then constrains:
    - Each piece placed exactly once
    - Each cell covered at most once

    Args:
        bay: The bay to fill
        shapes: Dict mapping shape index to boolean numpy array
        orientations: Dict mapping shape index to list of orientation arrays
        timeout_ms: Solver timeout in milliseconds (default 5 seconds)

    Returns:
        True if Z3 finds a valid packing, False otherwise (including timeout)
    """
    # Quick area check first
    if cells_required_by(bay, using_shapes=shapes) > cells_available_in(bay):
        return False

    height, width = bay.grid_size

    # Flatten counts to list of individual present instances
    presents_to_place = [
        shape_idx
        for shape_idx, count in enumerate(bay.presents_to_load)
        for _ in range(count)
    ]

    # For each present instance, track all possible placements
    possible_placements = {p: [] for p in range(len(presents_to_place))}

    # For each cell, track which placements would cover it
    cell_occupancy = {}

    # Generate all possible placements as Z3 boolean variables
    for present_id, shape_idx in enumerate(presents_to_place):
        for orient_id, orientation in enumerate(orientations[shape_idx]):
            present_h, present_w = orientation.shape

            for r in range(height - present_h + 1):
                for c in range(width - present_w + 1):
                    # Create a variable for this specific placement
                    var = Bool(f"present{present_id}_orient{orient_id}_row{r}_col{c}")
                    possible_placements[present_id].append(var)

                    # Track which cells this placement would cover
                    cells_in_shape = np.argwhere(orientation)
                    for dr, dc in cells_in_shape:
                        cell = (r + dr, c + dc)
                        if cell not in cell_occupancy:
                            cell_occupancy[cell] = []
                        cell_occupancy[cell].append(var)

    # Build the solver with constraints
    solver = Solver()
    solver.set("timeout", timeout_ms)

    # Constraint 1: Each present must be placed exactly once
    for present_id in range(len(presents_to_place)):
        placements_for_present = possible_placements[present_id]
        solver.add(PbEq([(v, 1) for v in placements_for_present], 1))

    # Constraint 2: Each cell can have at most one present
    for cell, overlapping_placements in cell_occupancy.items():
        if len(overlapping_placements) > 1:
            solver.add(PbLe([(v, 1) for v in overlapping_placements], 1))

    return solver.check() == sat
