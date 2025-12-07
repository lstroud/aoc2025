"""
Day 7 Part 1: Tachyon Beam Splitting - Count total splits

Simulates tachyon beam propagation through a manifold.
Beams merge when they converge (clip to binary).
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
    console.print(Padding("\n[bold]Day 7 Part 1: Tachyon Beam Splitting[/bold]", (0, 0, 1, 4)))

    manifold = load_tachyon_manifold('data.dat')

    run_benchmarks(
        console,
        manifold,
        strategies={
            "Matrix Propagation": lambda m: matrix_propagation(m, clip=True),
            "Convolution": lambda m: convolution(m, clip=True),
            "PyTorch (CPU)": lambda m: pytorch_neural(m, device="cpu", clip=True),
        },
        pytorch_strategy=lambda m, device: pytorch_neural(m, device=device, clip=True),
        result_label="Total Splits"
    )
