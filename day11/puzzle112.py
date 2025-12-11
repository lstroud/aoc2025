import functools
from pathlib import Path
from functools import cache
from rich.console import Console
from rich.panel import Panel

console = Console()


def load_wiring_diagram(file_name: str) -> dict[str: list[str]]:
    """Load machine specs from what remains of the manual."""
    manual_path = Path(__file__).parent / file_name
    with open(manual_path) as f:
        return {k: (v.split()) for k, v in (line.split(":") for line in f)}


def solve(file_name: str) -> int:
    """Count paths from 'svr' to 'out' that visit both 'dac' and 'fft'."""
    data = load_wiring_diagram(file_name)

    @functools.cache
    def count_paths(output_name: str, seen_dac: bool, seen_fft: bool) -> int:
        if output_name == "out" and seen_dac and seen_fft:
            return 1
        else:
            outputs = data.get(output_name, None)
            if outputs is None:
                return 0
            return sum([count_paths(p, seen_dac or p == "dac", seen_fft or p == "fft") for p in outputs])

    paths = count_paths("svr", False, False)
    return paths

if __name__ == "__main__":
    sample = solve("sample2.dat")
    print(f"Sample: {sample} (expected 2)")

    answer = solve("data.dat")
    console.print(Panel(f"[bold green]{answer}[/]", title="Day 11 Part 2"))
