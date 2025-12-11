from dataclasses import dataclass
import functools
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.panel import Panel

console = Console()


def load_wiring_diagram(file_name: str) -> dict[str: list[str]]:
    """Load machine specs from what remains of the manual."""
    manual_path = Path(__file__).parent / file_name
    with open(manual_path) as f:
        return {k: (v.split()) for k, v in (line.split(":") for line in f)}



def solve(file_name: str) -> int:
    """Count all paths from 'you' to 'out' in the device graph."""
    data = load_wiring_diagram(file_name)

    @functools.cache
    def count_paths(output_name: str) -> int:
        if output_name == "out":
            return 1
        else:
            outputs = data.get(output_name, None)
            if outputs is None:
                return 0
            return sum([count_paths(p) for p in outputs])


    paths = count_paths("you")
    return paths

if __name__ == "__main__":
    sample = solve("sample.dat")
    print(f"Sample: {sample} (expected 5)")

    answer = solve("data.dat")
    console.print(Panel(f"[bold green]{answer}[/]", title="Day 11 Part 1"))
