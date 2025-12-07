"""Base utilities for tachyon beam strategies."""

from pathlib import Path
import numpy as np


def load_tachyon_manifold(file_path: str) -> np.ndarray:
    """Load tachyon manifold from file as character array."""
    current_dir = Path(__file__).parent.parent
    full_path = current_dir / file_path
    with open(full_path) as f:
        lines = [list(line.strip()) for line in f]
    return np.array(lines, dtype='U1')
