from pathlib import Path
import pandas as pd
import numpy as np
from scipy.ndimage import convolve
from rich.console import Console
from rich.rule import Rule
from rich.text import Text
from rich.padding import Padding
from rich.panel import Panel
from rich.console import Group

def read_layout(file_path: str) -> np.ndarray:
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    df = pd.read_csv(full_path, header=None)
    grid = df[0].apply(list).to_list()
    return np.array(grid)

grid = read_layout('data.dat')
binary = (grid == '@').astype(int)

kernel = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

neighbor_counts = convolve(binary, kernel, mode='constant', cval=0)
accessible = (binary == 1) & (neighbor_counts < 4)
accessible_count = accessible.sum()

# Display Results
console = Console()
console.print(Padding("\n[bold]Puzzle Summary[/bold]", (0, 0, 1, 4)))
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
# console.print(Rule(style="blue"))
console.print(Padding(f"[bold red]Accessible Count[/bold red]: {int(accessible_count)}\n", (0, 0, 0, 4)))


