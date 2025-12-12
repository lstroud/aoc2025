"""
Shared utilities for present packing strategies.

The elves need to load presents into bays under Christmas trees. This module
provides the data structures and loading functions that all strategies share.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np


@dataclass
class PresentBay:
    """A bay under a Christmas tree where presents get loaded."""
    grid_size: tuple[int, int]    # (height, width) of the bay's cell grid
    presents_to_load: np.ndarray  # how many of each present shape to load


class PackingManifest:
    """
    The elves' packing manifest: present shapes and bay assignments.

    Like any good warehouse operation, the elves have very specific present
    shapes and very specific requirements for what goes into each bay.
    """
    SHAPE_PATTERN = re.compile(r'^(\d+):\n([\s\S]*)')
    BAY_PATTERN = re.compile(r'^(\d+)x(\d+):\s+(.+)$')

    def __init__(self, from_blocks: list[str]):
        self.present_shapes: dict[int, np.ndarray] = {}
        self.bays: list[PresentBay] = []

        for block in from_blocks:
            if self._looks_like_shape(block):
                idx, shape = self._parse_present_shape(block)
                self.present_shapes[idx] = shape
            else:
                self.bays = self._parse_bays(block)

    def _looks_like_shape(self, block: str) -> bool:
        return bool(re.match(self.SHAPE_PATTERN, block))

    def _parse_present_shape(self, from_block: str) -> tuple[int, np.ndarray]:
        """Parse a present shape definition like '4:\\n###\\n#..\\n###'"""
        match = re.match(self.SHAPE_PATTERN, from_block)
        if not match:
            raise ValueError(f"This doesn't look like a present shape: {from_block}")

        shape_id = int(match.group(1))
        pattern_text = match.group(2)

        rows = [list(line) for line in pattern_text.splitlines() if line]
        grid = np.array(rows)
        shape = (grid == "#").astype(bool)

        return shape_id, shape

    def _parse_bays(self, from_block: str) -> list[PresentBay]:
        """Parse all bay specs like '12x5: 1 0 1 0 2 2'"""
        return [self._parse_single_bay(line) for line in from_block.split('\n')]

    def _parse_single_bay(self, from_line: str) -> PresentBay:
        """
        Parse '12x5: 1 0 1 0 2 2' into grid size and present counts.

        The format is WIDTHxHEIGHT but we store as (height, width) because
        that's how numpy arrays work. I'll spare you the story of how long
        that particular bug took to find.
        """
        match = re.match(self.BAY_PATTERN, from_line)
        width, height = int(match.group(1)), int(match.group(2))
        counts = np.array(match.group(3).split(), dtype=int)
        return PresentBay(grid_size=(height, width), presents_to_load=counts)


def load_manifest(from_file: str) -> PackingManifest:
    """Load the packing manifest from a file."""
    filepath = Path(__file__).parent.parent / from_file
    with open(filepath, "r", encoding="utf-8") as f:
        blocks = f.read().strip().split("\n\n")
    return PackingManifest(from_blocks=blocks)


def get_all_orientations(of_shape: np.ndarray) -> list[np.ndarray]:
    """
    Elves can rotate and flip presents to make them fit.

    Four rotations times two (original plus mirror) gives eight possible
    orientations. Some shapes are symmetric, so we dedupe to avoid the
    solver doing the same work twice.
    """
    orientations = []

    for k in range(4):
        orientations.append(np.rot90(of_shape, k))

    flipped = np.flip(of_shape, axis=1)
    for k in range(4):
        orientations.append(np.rot90(flipped, k))

    unique = {}
    for arr in orientations:
        key = (arr.shape, arr.tobytes())
        if key not in unique:
            unique[key] = arr

    return list(unique.values())


def precompute_orientations(for_shapes: dict[int, np.ndarray]) -> dict[int, list[np.ndarray]]:
    """Pre-calculate all orientations for each present shape."""
    return {idx: get_all_orientations(of_shape=shape)
            for idx, shape in for_shapes.items()}


def cells_required_by(bay: PresentBay, using_shapes: dict[int, np.ndarray]) -> int:
    """Count total cells needed by all presents in a bay."""
    cells_per_present_type = np.array([
        np.sum(using_shapes[i]) for i in range(len(bay.presents_to_load))
    ])
    return int(np.dot(bay.presents_to_load, cells_per_present_type))


def cells_available_in(bay: PresentBay) -> int:
    """Count total cells available in a bay."""
    height, width = bay.grid_size
    return height * width
