from pathlib import Path

import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel


def load_tile_locations(file_path: str) -> np.ndarray:
    """Load red tile coordinates from the movie theater floor."""
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    return pd.read_csv(full_path, header=None)


red_tiles = load_tile_locations('data.dat')
# 32-bit overflows for the data
tile_cols = red_tiles[0].to_numpy(dtype='int64', copy=False)
tile_rows = red_tiles[1].to_numpy(dtype='int64', copy=False)

width_spans = np.abs(tile_cols[:, None] - tile_cols)
height_spans = np.abs(tile_rows[:, None] - tile_rows)

carpet_sizes = (width_spans + 1) * (height_spans + 1)
biggest_carpet = carpet_sizes.max()

console = Console()
console.print(Panel(
    f"🎄 Biggest carpet area: [bold red]{biggest_carpet}[/bold red] tiles 🎅",
    title="[green]Movie Theater Results[/green]",
    expand=False,
    border_style="red"
))