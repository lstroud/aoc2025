from pathlib import Path
import re

import numpy as np
from pandas import IntervalIndex
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.console import Group
from rich.text import Text



class Section:
    def __init__(self, data: str):
        self.data = data
        self.lines = self._parse(data)

    def is_range_section(self):
        pattern = re.compile(r'^\d+-\d+$')
        return all(pattern.match(l) for l in self.lines)
    
    def is_value_section(self):
        return all(l.isdigit() for l in self.lines)
    
    def get_as_range_tuples(self):
        if not self.is_range_section():
            raise ValueError("Not a Range Section")
        return [tuple(map(int, r.split('-'))) for r in self.lines]
    
    @property
    def values(self):
        return np.array([int(v) for v in self.lines])

    def _parse(self, data:str):
        return data.strip().split('\n')
    
    def __str__(self):
        return "\n".join(self.lines)
    
    def __repr__(self):
        return "\n".join(self.lines)


def read_input(file_path: str) -> list[Section]:
    current_dir = Path(__file__).parent
    full_path = current_dir / file_path
    sections = []
    with open(full_path) as f:
        sections = [Section(s) for s in f.read().split('\n\n')]
    return sections


def parse_sections(sections: list[Section]) -> tuple[IntervalIndex, np.ndarray]:
    """Extract intervals and values from parsed sections."""
    range_tuples = []
    values_list = []
    for s in sections:
        if s.is_range_section():
            range_tuples.extend(s.get_as_range_tuples())
        else:
            values_list.append(s.values)
    idx = IntervalIndex.from_tuples(range_tuples, closed='both')
    values = np.concatenate(values_list)
    return idx, values


def find_interval_matches(idx: IntervalIndex, values: np.ndarray) -> np.ndarray:
    """Vectorized interval containment check using broadcasting."""
    left = np.array([iv.left for iv in idx])
    right = np.array([iv.right for iv in idx])
    return (values[:, None] >= left) & (values[:, None] <= right)


def display_results(console: Console, matching_values: np.ndarray, total: int):
    """Display puzzle results using rich formatting."""
    values_text = Text(", ".join(str(v) for v in matching_values), style="green")

    console.print(Padding("\n[bold]Puzzle Summary[/bold]", (0, 0, 1, 4)))
    console.print(Padding(
        Panel(
            Group(
                Text(f"Matching values: ", style="cyan"),
                values_text,
                Text(f"\n\nCount: {len(matching_values)}", style="yellow"),
                Text(f"\nTotal: {total}", style="bold red"),
            ),
            title="Results",
            border_style="blue",
            expand=False,
            padding=(1, 1, 1, 1)
        ),
        (0, 0, 0, 4)
    ))


if __name__ == "__main__":
    sections = read_input('data.dat')
    idx, values = parse_sections(sections)

    has_match = find_interval_matches(idx, values).any(axis=1)
    matching_values = values[has_match]

    console = Console()
    display_results(console, matching_values, matching_values.sum())