"""
Advent of Code 2025 - Day 4, Part 1: Warehouse Accessibility

Problem: Find items (@) that are accessible from the edge - items with
fewer than 4 neighboring items (not fully surrounded).

Uses convolution to count neighbors for each cell efficiently.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.ndimage import convolve
from rich.console import Console
from rich.text import Text
from rich.padding import Padding
from rich.panel import Panel
from rich.console import Group

NEIGHBOR_KERNEL = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])


def load_inventory_grid(file_path: str) -> np.ndarray:
    """Load warehouse grid from file as character array."""
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    df = pd.read_csv(full_path, header=None)
    grid = df[0].apply(list).to_list()
    return np.array(grid)


def calculate_accessible_positions(grid: np.ndarray) -> np.ndarray:
    """Find positions that are items (@) with fewer than 4 neighbors."""
    binary = (grid == '@').astype(int)
    neighbor_counts = convolve(binary, NEIGHBOR_KERNEL, mode='constant', cval=0)
    return (binary == 1) & (neighbor_counts < 4)


def display_grid(console: Console, grid: np.ndarray, accessible: np.ndarray):
    """Display grid with accessible items highlighted."""
    binary = (grid == '@').astype(int)
    display = np.where(accessible, 'x', np.where(binary == 1, '@', '.'))
    rows = []
    for row in display:
        text = Text()
        for char in row:
            style = {"x": "green", "@": "red", ".": "dim"}.get(char, "")
            text.append(char, style=style)
        rows.append(text)
    console.print(
        Padding(
            Panel(Group(*rows), title="Grid", border_style="blue", expand=False, padding=(1, 1, 1, 1)),
            (0, 0, 0, 4))
        )


if __name__ == "__main__":
    grid = load_inventory_grid('data.dat')
    accessible = calculate_accessible_positions(grid)

    console = Console()
    console.print(Padding("\n[bold]Puzzle Summary[/bold]", (0, 0, 1, 4)))
    display_grid(console, grid, accessible)
    console.print(Padding(f"[bold red]Accessible Count[/bold red]: {int(accessible.sum())}\n", (0, 0, 0, 4)))
