"""
Wave Picking Simulation for Warehouse Inventory

Simulates the process of picking items from a warehouse grid where items can only
be accessed if they have fewer than a threshold number of neighbors. Each wave
picks all currently accessible items, which may expose new items for subsequent waves.

Uses 2D convolution to efficiently calculate neighbor counts across the entire grid.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.ndimage import convolve
from rich.console import Console
from rich.text import Text
from rich.padding import Padding
from rich.panel import Panel
from rich.console import Group


# Constants
ACCESSIBILITY_THRESHOLD = 4  # Items with fewer than this many neighbors are accessible
NEIGHBOR_KERNEL = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])  # 8-directional neighbor kernel for convolution

GRID_CELL_STYLES = {
    "x": "green",   # Picked items
    "@": "red",     # Remaining items
    ".": "dim"      # Empty positions
}

@dataclass
class WavePickResult:
    """Result of a single wave pick operation."""
    items_picked_count: int
    picked_mask: np.ndarray
    updated_grid: np.ndarray

class DisplayVerbosity(Enum):
    """Controls how much output is displayed during simulation."""
    FULL = "full"         # Header + all wave panels + final result
    BOOKENDS = "bookends" # Header + first/last wave panels + summary
    SUMMARY = "summary"   # Header + summary panel only

@dataclass
class WaveDisplayData:
    """Data needed to display a wave result after simulation completes."""
    wave_number: int
    items_picked_count: int
    remaining_accessible_count: int
    picked_mask: np.ndarray
    grid_before_pick: np.ndarray

@dataclass
class SimulationResult:
    """Complete results from running a wave picking simulation."""
    total_items_picked: int
    wave_picks: list[int]
    wave_data: list[WaveDisplayData]

def load_inventory_grid(file_path: str) -> np.ndarray:
    """
    Load warehouse inventory from file as a binary grid.

    Reads the layout file and converts '@' characters to 1 (item present)
    and all other characters to 0 (empty position).

    Args:
        file_path: Path to the inventory data file (relative to this module)

    Returns:
        Binary numpy array where 1 = item present, 0 = empty
    """
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    df = pd.read_csv(full_path, header=None)
    grid = df[0].apply(list).to_list()
    grid = np.array(grid)
    binary_grid = (grid == '@').astype(int)
    return binary_grid

def calculate_accessible_positions(grid: np.ndarray) -> np.ndarray:
    """
    Calculate which positions in the grid are currently accessible for picking.

    A position is accessible if it contains an item and has fewer than
    ACCESSIBILITY_THRESHOLD neighbors.

    Args:
        grid: Binary grid where 1 = item present, 0 = empty

    Returns:
        Boolean mask where True = position is accessible
    """
    neighbor_counts = convolve(grid, NEIGHBOR_KERNEL, mode='constant', cval=0)
    is_accessible = (grid == 1) & (neighbor_counts < ACCESSIBILITY_THRESHOLD)
    return is_accessible

def render_grid_rows(picked_mask: np.ndarray, grid_before_pick: np.ndarray) -> list[Text]:
    """
    Convert grid state to styled Text rows for Rich console display.

    Args:
        picked_mask: Boolean mask where True = item was picked this wave
        grid_before_pick: Grid state before picking (1 = item, 0 = empty)

    Returns:
        List of Rich Text objects, one per row, with color-coded characters
    """
    display_grid = np.where(picked_mask, 'x', np.where(grid_before_pick == 1, '@', '.'))
    styled_rows = []
    for row in display_grid:
        row_text = Text()
        for cell_char in row:
            cell_style = GRID_CELL_STYLES.get(cell_char, "")
            row_text.append(cell_char, style=cell_style)
        styled_rows.append(row_text)
    return styled_rows

def display_wave_result(console: Console,
                        wave_number: int,
                        items_picked_count: int,
                        remaining_accessible_count: int,
                        picked_mask: np.ndarray,
                        grid_before_pick: np.ndarray):
    """
    Display the results of a single wave pick operation.

    Args:
        console: Rich console for output
        wave_number: Current wave number (1-indexed)
        items_picked_count: Number of items picked in this wave
        remaining_accessible_count: Number of items accessible for next wave
        picked_mask: Boolean mask of items picked this wave
        grid_before_pick: Grid state before this wave's picks
    """
    styled_rows = render_grid_rows(picked_mask, grid_before_pick)
    console.print(
        Padding(
            Panel(Group(*styled_rows,
                        Text(f"\nPicked     : {items_picked_count}\n", style="yellow"),
                        Text(f"Accessible : {remaining_accessible_count}\n", style="cyan")
                        ),
                  title=f"Wave #{wave_number}",
                  border_style="blue",
                  expand=False,
                  padding=(1, 1, 1, 1)),
            (0, 0, 0, 4))
        )
    


def perform_wave_pick(current_grid: np.ndarray) -> WavePickResult:
    """
    Execute a single wave of picking accessible items from the grid.

    Identifies all currently accessible positions and removes them from the grid.
    An item is accessible if it has fewer than ACCESSIBILITY_THRESHOLD neighbors.

    Args:
        current_grid: Binary grid where 1 = item present, 0 = empty

    Returns:
        WavePickResult containing items picked count, picked mask, and updated grid
    """
    neighbor_counts = convolve(current_grid, NEIGHBOR_KERNEL, mode='constant', cval=0)
    is_accessible = (current_grid == 1) & (neighbor_counts < ACCESSIBILITY_THRESHOLD)

    if is_accessible.sum() == 0:
        return WavePickResult(
            items_picked_count=0,
            picked_mask=np.zeros_like(current_grid, dtype=bool),
            updated_grid=current_grid
        )

    items_picked_count = is_accessible.sum()
    updated_grid = current_grid * ~is_accessible
    picked_mask = current_grid & ~updated_grid

    return WavePickResult(
        items_picked_count=items_picked_count,
        picked_mask=picked_mask,
        updated_grid=updated_grid
    )

def execute_simulation(file_path: str) -> SimulationResult:
    """
    Execute the wave picking simulation without any display output.

    Args:
        file_path: Path to the inventory data file

    Returns:
        SimulationResult containing all simulation data
    """
    inventory_grid = load_inventory_grid(file_path)
    total_items_picked = 0
    wave_number = 1
    wave_picks: list[int] = []
    wave_data: list[WaveDisplayData] = []

    while True:
        result = perform_wave_pick(inventory_grid)
        total_items_picked += result.items_picked_count
        wave_picks.append(result.items_picked_count)

        accessible_positions = calculate_accessible_positions(result.updated_grid)
        remaining_accessible_count = accessible_positions.sum()

        wave_data.append(WaveDisplayData(
            wave_number=wave_number,
            items_picked_count=result.items_picked_count,
            remaining_accessible_count=remaining_accessible_count,
            picked_mask=result.picked_mask.copy(),
            grid_before_pick=inventory_grid.copy()
        ))

        if remaining_accessible_count == 0:
            break

        inventory_grid = result.updated_grid
        wave_number += 1

    return SimulationResult(
        total_items_picked=total_items_picked,
        wave_picks=wave_picks,
        wave_data=wave_data
    )

def display_summary_panel(console: Console,
                          wave_picks: list[int],
                          total_items_picked: int):
    """
    Display a compact summary panel of the simulation results.

    Args:
        console: Rich console for output
        wave_picks: List of items picked per wave
        total_items_picked: Total items picked across all waves
    """
    wave_count = len(wave_picks)
    picks_formatted = ", ".join(str(p) for p in wave_picks)

    console.print(Padding(
        Panel(
            Group(
                Text(f"Waves          : {wave_count}\n", style="cyan"),
                Text(f"Picks per wave : {picks_formatted}\n", style="yellow"),
                Text(f"Total picked   : {total_items_picked}\n", style="bold red")
            ),
            title="Summary",
            border_style="blue",
            expand=False,
            padding=(1, 1, 1, 1)
        ),
        (0, 0, 0, 4)
    ))

def display_simulation_results(console: Console,
                               verbosity: DisplayVerbosity,
                               result: SimulationResult):
    """
    Display simulation results based on verbosity level.

    Args:
        console: Rich console for output
        verbosity: Controls output detail level
        result: Complete simulation results
    """
    console.print(Padding("\n[bold]Puzzle Summary[/bold]", (0, 0, 1, 4)))

    if verbosity == DisplayVerbosity.FULL:
        for wave in result.wave_data:
            display_wave_result(console, wave.wave_number,
                                wave.items_picked_count,
                                wave.remaining_accessible_count,
                                wave.picked_mask,
                                wave.grid_before_pick)
        console.print(Padding(
            f"[bold red]Total Picked[/bold red]: {result.total_items_picked}\n",
            (0, 0, 0, 4)
        ))
    elif verbosity == DisplayVerbosity.BOOKENDS and len(result.wave_data) > 0:
        first_wave = result.wave_data[0]
        last_wave = result.wave_data[-1]
        display_wave_result(console, first_wave.wave_number,
                            first_wave.items_picked_count,
                            first_wave.remaining_accessible_count,
                            first_wave.picked_mask,
                            first_wave.grid_before_pick)
        if last_wave.wave_number != first_wave.wave_number:
            display_wave_result(console, last_wave.wave_number,
                                last_wave.items_picked_count,
                                last_wave.remaining_accessible_count,
                                last_wave.picked_mask,
                                last_wave.grid_before_pick)
        display_summary_panel(console, result.wave_picks, result.total_items_picked)
    else:
        display_summary_panel(console, result.wave_picks, result.total_items_picked)

if __name__ == "__main__":
    console = Console()
    result = execute_simulation('data.dat')
    display_simulation_results(console, DisplayVerbosity.BOOKENDS, result)



