"""
Day 7 Part 2: Quantum Tachyon Splitting - Count timelines

Many-worlds interpretation: each splitter creates two timelines.
Timelines don't merge - accumulate counts (no clipping).
"""

from rich.console import Console
from rich.padding import Padding

from strategies import (
    load_tachyon_manifold,
    matrix_propagation,
    convolution,
    pytorch_neural,
)
from benchmark import run_benchmarks


if __name__ == "__main__":
    console = Console()
    console.print(Padding("\n[bold]Day 7 Part 2: Quantum Tachyon Splitting[/bold]", (0, 0, 1, 4)))

    manifold = load_tachyon_manifold('data.dat')

    run_benchmarks(
        console,
        manifold,
        strategies={
            "Matrix Propagation": lambda m: matrix_propagation(m, clip=False),
            "Convolution": lambda m: convolution(m, clip=False),
            "PyTorch (CPU)": lambda m: pytorch_neural(m, device="cpu", clip=False),
        },
        pytorch_strategy=lambda m, device: pytorch_neural(m, device=device, clip=False),
        result_label="Timelines"
    )
