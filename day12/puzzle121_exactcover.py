from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import exact_cover as ec
from exact_cover.error import NoSolution
from rich.console import Console
from rich.panel import Panel

console = Console()

@dataclass
class RegionSpec:
    shape: tuple[int, int]
    piece_count: np.ndarray

class ShapeRegistry:

    SHAPE_REGEX = re.compile(r'^(\d+):\n([\s\S]*)')
    REGION_BLOCK_REGEX = re.compile(r'^(\d+x\d+):+')
    REGION_SPEC_REGEX = re.compile(r'^(\d+)x(\d+):\s+(.+)$')
    def __init__(self, blocks: list[str]):
        self.shape_idx: dict[int, np.ndarray] = {}
        self.regions: list[RegionSpec] = []
        for block in blocks:
            if self._is_shape_block(block):
                s = self._parse_shape_block(block)
                self.shape_idx[s[0]] = s[1]
            else:
                self.regions = self._parse_region_block(block)
                

    def _is_shape_block(self, block: str) -> bool:
        return re.match(self.SHAPE_REGEX, block)
    
    def _is_region_block(self, block:str) -> bool:
        return re.match(self.REGION_BLOCK_REGEX, block)
    
    def _parse_shape_block(self, block: str) -> tuple[int, np.ndarray]:
        m = re.match(self.SHAPE_REGEX, block)
        if m:
            idx = int(m.group(1))
            pattern_str = m.group(2)
            pattern_arr = np.array([list(line) for line in pattern_str.splitlines() if line])
            mask = (pattern_arr == "#").astype(bool)
            return (idx, mask)
        else:
            raise ValueError("Not a Shape:", block)
        
    def _parse_region_block(self, block: str) -> list[RegionSpec]:
        return [self._parse_region_spec(r) for r in block.split('\n')]

    def _parse_region_spec(self, line: str) -> RegionSpec:
        m = re.match(self.REGION_SPEC_REGEX, line)
        shape = (int(m.group(2)), int(m.group(1)))
        mask = np.array(m.group(3).split(), dtype=int)
        return RegionSpec(shape, mask)
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "\n".join([
            str(self.shape_idx),
            str(self.regions)
        ])



def parse_shape_registry(file_name: str) -> ShapeRegistry:
    """
    Parse the puzzle input file.

    Returns:
        shapes: dict mapping shape index to boolean numpy array
        regions: list of (width, height, piece_counts) tuples
    """
    shape_file = Path(__file__).parent / file_name
    with open(shape_file, "r", encoding="utf-8") as f:
        blocks = f.read().strip().split("\n\n")
    return ShapeRegistry(blocks)


def get_orientations(shape: np.ndarray) -> list[np.ndarray]:
    """
    Generate all unique orientations of a shape (rotations + flips).

    Use np.rot90(shape, k) for k=0,1,2,3 rotations
    Use np.flip(shape, axis=1) for horizontal mirror
    Deduplicate symmetric shapes using tobytes() or tuple conversion

    Returns:
        List of unique orientation arrays
    """
    orientations = []
    # Generate 4 rotations of original
    for k in range(4):
        orientations.append(np.rot90(shape, k))
    # Generate 4 rotations of flipped
    for k in range(4):
         orientations.append(np.rot90(np.flip(shape, axis=1), k))
    # Dedupe
    seen = {}
    for arr in orientations:
        key = (arr.shape, arr.tobytes())
        if key not in seen:
            seen[key] = arr
    return list(seen.values())


def can_fit_dlx(grid_shape: tuple[int, int], piece_counts: np.ndarray,
                 shape_orientations: dict[int, list[np.ndarray]]) -> bool:
    """
    Use exact_cover (DLX algorithm) to determine if pieces fit in grid.

    Matrix structure:
    - Columns 0 to (height*width - 1): grid cells
    - Columns (height*width) to end: piece slots (one per piece instance)

    Each row represents one placement: (piece_instance, orientation, row, col)
    Row has True in: cells covered by piece + that piece's slot column

    Args:
        grid_shape: (height, width) of the region
        piece_counts: array where piece_counts[i] = number of shape i needed
        shape_orientations: dict mapping shape index to list of orientation arrays

    Returns:
        True if all pieces can be placed without overlap
    """
    height, width = grid_shape
    num_cells = height * width

    # Count total pieces needed
    total_pieces = sum(piece_counts)
    num_cols = num_cells + total_pieces

    # TODO: Build list of rows
    flat_piece_counts = [i for i, v in enumerate(piece_counts) for _ in range(v)]
    # For each piece instance (piece_idx, instance within that type):
    rows = []
    for p, shape_idx in enumerate(flat_piece_counts):
        for orientation in shape_orientations[shape_idx]:
    #     For each valid (r, c) position:
            piece_h, piece_w = orientation.shape
            for r in range(height - piece_h + 1):
                for c in range(width - piece_w + 1):
                    row = np.zeros(num_cols, dtype=bool)
                    cells = np.argwhere(orientation)
                    absolute_cells = (r + cells[:, 0]) * width + (c + cells[:, 1])
                    row[absolute_cells] = True
                    row[num_cells + p] = True
                    rows.append(row)
    
    matrix = np.array(rows, dtype=bool)
    print(f"flat_piece_counts: {flat_piece_counts}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Pieces: {total_pieces}, Cells: {num_cells}")
    try:
        solution = ec.get_exact_cover(matrix)
        return True
    except:
        return False


def solve(file_name: str) -> int:
    """Count how many regions can fit all their required pieces."""
    shape_registry = parse_shape_registry(file_name)

    # Precompute orientations for each shape type
    shape_orientations = {idx: get_orientations(shape)
                          for idx, shape in shape_registry.shape_idx.items()}

    count = 0
    for i, r in enumerate(shape_registry.regions):
        if can_fit_dlx(r.shape, r.piece_count, shape_orientations):
            count += 1

    return count


if __name__ == "__main__":
    # Test on sample
    sample_answer = solve("sample.dat")
    console.print(f"Sample: {sample_answer} (expected 2)")

    # Real data
    # answer = solve("data.dat")
    # console.print(Panel(f"{answer}", title="Day 12 Part 1", border_style="red"))
