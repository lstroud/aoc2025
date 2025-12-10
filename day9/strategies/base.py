"""Shared utilities for movie theater carpet strategies."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_tile_locations(file_name: str) -> list[tuple[int, int]]:
    """Load red tile coordinates from file.

    Args:
        file_name: Name of data file (relative to day9/)

    Returns:
        List of (col, row) coordinate tuples in polygon traversal order
    """
    day9_dir = Path(__file__).parent.parent
    df = pd.read_csv(day9_dir / file_name, header=None)
    return list(zip(df[0], df[1]))


def tile_area(x1: int, y1: int, x2: int, y2: int) -> int:
    """Calculate tile count for rectangle with opposite corners.

    Uses discrete tile formula: (|Δx| + 1) × (|Δy| + 1)
    """
    return (abs(x2 - x1) + 1) * (abs(y2 - y1) + 1)
