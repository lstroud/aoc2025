"""Benchmarking and display utilities for junction box strategies."""

import time
import tracemalloc
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.padding import Padding


def benchmark(name: str, func, coords: np.ndarray, k: int) -> dict:
    """Run a strategy and measure time and memory (CPU and GPU)."""
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    tracemalloc.start()
    start = time.perf_counter()

    result = func(coords, k)

    if cuda_available:
        torch.cuda.synchronize()
    elif mps_available:
        torch.mps.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gpu_memory_kb = 0
    if cuda_available:
        gpu_memory_kb = torch.cuda.max_memory_allocated() / 1024
    elif mps_available:
        gpu_memory_kb = torch.mps.driver_allocated_memory() / 1024

    return {
        'name': name,
        'result': result,
        'time_ms': elapsed_ms,
        'memory_kb': peak_memory / 1024,
        'gpu_memory_kb': gpu_memory_kb,
    }


def get_gpu_info() -> tuple[str | None, str | None]:
    """Return (gpu_type, gpu_name) or (None, None) if no GPU."""
    if torch.cuda.is_available():
        return "CUDA", torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        return "MPS", "Apple Silicon"
    return None, None


def get_gpu_baseline() -> float:
    """Get current GPU memory baseline in KB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024
    elif torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 1024
    return 0


def display_results(
    console: Console,
    results: list[dict],
    n_boxes: int,
    k: int,
    gpu_type: str | None,
):
    """Display benchmark results in a table."""
    table = Table(title=f"Strategy Comparison ({n_boxes:,} boxes, k={k:,})")
    table.add_column("Strategy", style="cyan")
    table.add_column("Answer", justify="right", style="green")
    table.add_column("Time (ms)", justify="right", style="magenta")
    table.add_column("Memory (KB)", justify="right", style="blue")
    if gpu_type:
        table.add_column(f"{gpu_type} (KB)", justify="right", style="yellow")

    for r in results:
        row = [
            r['name'],
            f"{r['result']:,}",
            f"{r['time_ms']:.3f}",
            f"{r['memory_kb']:.1f}",
        ]
        if gpu_type:
            row.append(f"{r['gpu_memory_kb']:.1f}")
        table.add_row(*row)

    console.print(Padding(table, (1, 0, 1, 4)))


def run_benchmarks(
    console: Console,
    coords: np.ndarray,
    strategies: dict,
    k: int = 1000,
    pytorch_strategy=None,
):
    """Run all benchmarks and display results."""
    gpu_type, gpu_name = get_gpu_info()

    console.print(Padding(
        f"[cyan]Junction boxes:[/cyan] {len(coords):,}",
        (1, 0, 0, 4)
    ))
    console.print(Padding(
        f"[cyan]Connection attempts:[/cyan] {k:,}",
        (0, 0, 0, 4)
    ))
    if gpu_type:
        console.print(Padding(
            f"[green]GPU available:[/green] {gpu_name} ({gpu_type})",
            (0, 0, 1, 4)
        ))
    else:
        console.print(Padding(
            "[yellow]GPU:[/yellow] Not available (using CPU)",
            (0, 0, 1, 4)
        ))

    gpu_baseline_kb = get_gpu_baseline()
    results = [benchmark(name, func, coords, k) for name, func in strategies.items()]

    # Run PyTorch GPU strategy if provided and GPU available
    if pytorch_strategy and gpu_type:
        device = "mps" if gpu_type == "MPS" else "cuda"
        results.append(benchmark(
            f"PyTorch ({gpu_type})",
            lambda c, k: pytorch_strategy(c, k, device),
            coords,
            k
        ))

    # Subtract baseline GPU memory
    for r in results:
        r['gpu_memory_kb'] = max(0, r['gpu_memory_kb'] - gpu_baseline_kb)

    display_results(console, results, len(coords), k, gpu_type)
