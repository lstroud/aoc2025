"""Present packing strategies for the Christmas Tree Farm puzzle."""

from .base import (
    PresentBay,
    PackingManifest,
    load_manifest,
    get_all_orientations,
    precompute_orientations,
    cells_required_by,
    cells_available_in,
)
from .area_check import solve as area_check
from .backtrack import solve as backtrack
from .dlx import solve as dlx
from .z3_sat import solve as z3_sat
from .hybrid import solve as hybrid

__all__ = [
    # Data structures
    'PresentBay',
    'PackingManifest',

    # Utilities
    'load_manifest',
    'get_all_orientations',
    'precompute_orientations',
    'cells_required_by',
    'cells_available_in',

    # Strategies (swap import in puzzle121.py to try different ones)
    'hybrid',       # THE WORKING SOLUTION - area check + Z3 for small cases
    'area_check',   # Fast but wrong on sample bay 3
    'z3_sat',       # Correct but hangs on data.dat
    'dlx',          # Correct but slow
    'backtrack',    # Hangs on UNSAT cases
]
