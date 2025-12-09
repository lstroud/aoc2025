"""Shared utilities for junction box strategies."""
from pathlib import Path
import numpy as np


def load_junction_boxes(file_name: str) -> np.ndarray:
    """Load junction box 3D coordinates from file.

    Args:
        file_name: Name of data file (relative to day8/)

    Returns:
        n×3 array of integer coordinates
    """
    day8_dir = Path(__file__).parent.parent
    return np.loadtxt(day8_dir / file_name, delimiter=',', dtype=int)
