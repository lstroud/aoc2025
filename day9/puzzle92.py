from pathlib import Path
import numpy as np
import pandas as pd
from shapely import Polygon, box
from itertools import combinations
from rich.console import Console
from rich.panel import Panel


def load_tile_locations(file_path: str) -> np.ndarray:
    """Load red tile coordinates from the movie theater floor."""
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    return pd.read_csv(full_path, header=None)


def count_carpet_tiles(carpet: Polygon) -> int:
    """Count tiles in carpet. Shapely thinks in continuous space, elves think in tiles."""
    xs = carpet.bounds[0::2]
    ys = carpet.bounds[1::2]
    return int((np.abs(xs[0] - xs[1]) + 1) * (np.abs(ys[0] - ys[1]) + 1))


# Load the red tile markers the elves left behind
red_tiles = load_tile_locations('data.dat')
tile_corners = list(zip(red_tiles[0], red_tiles[1]))

# The elves' renovation zone - only red and green tiles allowed
# (they really should have checked the supply closet first)
renovation_zone = Polygon(tile_corners)

if not renovation_zone.is_valid:
    raise ValueError("The elves drew an impossible floor plan. Again.")

# Find the biggest carpet that fits in the renovation zone
# with red tiles at opposite corners (elf union rules, don't ask)
biggest_carpet = 0
best_carpet_spot = None

for corner1, corner2 in combinations(tile_corners, 2):
    carpet_candidate = box(corner1[0], corner1[1], corner2[0], corner2[1])

    if renovation_zone.contains(carpet_candidate):
        carpet_size = count_carpet_tiles(carpet_candidate)
        if carpet_size > biggest_carpet:
            biggest_carpet = carpet_size
            best_carpet_spot = carpet_candidate

console = Console()
console.print(Panel(
    f"🎄 Biggest carpet: [bold red]{biggest_carpet}[/bold red] tiles 🎅\n"
    f"[dim]Location: {best_carpet_spot.bounds if best_carpet_spot else 'None'}[/dim]",
    title="[green]Movie Theater Renovation[/green]",
    border_style="red",
    expand=False
))