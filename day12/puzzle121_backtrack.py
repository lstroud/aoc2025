from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
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


def can_fit(grid: np.ndarray, pieces: list[np.ndarray]) -> bool:
    """
    Recursive backtracking solver.

    Args:
        grid: Current state of the region (True = occupied)
        pieces: Remaining pieces to place (each is list of orientations)

    Returns:
        True if all pieces can be placed without overlap
    """
    if len(pieces) == 0:
        return True
    
    empty_cells = np.sum(~grid)
    needed_cells = sum(np.sum(p[0]) for p in pieces)
    if needed_cells > empty_cells:
        return False

    current_piece_orientations = pieces[0]
    remaining_pieces = pieces[1:]

    for orientation in current_piece_orientations:
        piece_h, piece_w = orientation.shape
        grid_h, grid_w = grid.shape
        for r in range(grid_h - piece_h + 1):
            for c in range(grid_w - piece_w + 1):
                if np.any(grid[r:r+piece_h, c:c+piece_w] & orientation):
                    continue

                grid[r:r+piece_h, c:c+piece_w] |= orientation
                if can_fit(grid, remaining_pieces):
                    return True
                grid[r:r+piece_h, c:c+piece_w] &= ~orientation

    return False

def solve(file_name: str) -> int:
    """
    Count how many regions can fit all their required pieces.
    """
    # shapes, regions = parse_input(file_name)
    shape_registry = parse_shape_registry(file_name)
    for r in shape_registry.regions:
      print(f"Region: {r.shape}, pieces: {r.piece_count}, total pieces: {sum(r.piece_count)}")
    # # Precompute orientations for each shape
    shape_orientations = {idx: get_orientations(shape) for idx, shape in shape_registry.shape_idx.items()}
    # print(shape_orientations)

    count = 0
    for i, r in enumerate(shape_registry.regions):
        # Build list of pieces (as orientation lists) from counts
        print(f"Solving region {i}: {r.shape}")
        pieces = []
        for idx, qty in enumerate(r.piece_count):
            for _ in range(qty):
                pieces.append(shape_orientations[idx])

        pieces.sort(key=lambda p: -np.sum(p[0]))
        # Try to fit all pieces
        grid = np.zeros(r.shape, dtype=bool)
        if i == 0:
            print(f"Grid shape: {grid.shape}")
            print(f"Number of pieces: {len(pieces)}")
            print(f"First piece orientations: {len(pieces[0])}")
            for j, ori in enumerate(pieces[0]):
                print(f"  Orientation {j} shape: {ori.shape}")
                print(ori.astype(int))
        if can_fit(grid, pieces):
            print(f"Region {i}: FITS")
            count += 1
        else:
          print(f"Region {i}: doesn't fit")
    return count


if __name__ == "__main__":
    # Test on sample
    sample_answer = solve("sample.dat")
    console.print(f"Sample: {sample_answer} (expected 2)")

    # Real data
    # answer = solve("data.dat")
    # console.print(Panel(f"{answer}", title="Day 12 Part 1", border_style="red"))
