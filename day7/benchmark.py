"""Benchmarking and display utilities for tachyon beam strategies."""

import time
import tracemalloc
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.padding import Padding


def benchmark(name: str, func, manifold: np.ndarray) -> dict:
    """Run a strategy and measure time and memory (CPU and GPU)."""
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    tracemalloc.start()
    start = time.perf_counter()

    result = func(manifold)

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
        'gpu_memory_kb': gpu_memory_kb
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
    manifold_shape: tuple,
    gpu_type: str | None,
    result_label: str = "Result"
):
    """Display benchmark results in a table."""
    height, width = manifold_shape
    table = Table(title=f"Strategy Comparison ({height} x {width} manifold)")
    table.add_column("Strategy", style="cyan")
    table.add_column(result_label, justify="right", style="green")
    table.add_column("Time (ms)", justify="right", style="magenta")
    table.add_column("Memory (KB)", justify="right", style="blue")
    if gpu_type:
        table.add_column(f"{gpu_type} (KB)", justify="right", style="yellow")

    for r in results:
        row = [r['name'], f"{r['result']:,}", f"{r['time_ms']:.3f}", f"{r['memory_kb']:.1f}"]
        if gpu_type:
            row.append(f"{r['gpu_memory_kb']:.1f}")
        table.add_row(*row)

    console.print(Padding(table, (1, 0, 1, 4)))


def run_benchmarks(
    console: Console,
    manifold: np.ndarray,
    strategies: dict,
    pytorch_strategy=None,
    result_label: str = "Result"
):
    """Run all benchmarks and display results."""
    gpu_type, gpu_name = get_gpu_info()

    console.print(Padding(f"[cyan]Manifold size:[/cyan] {manifold.shape[0]} rows x {manifold.shape[1]} columns", (0, 0, 1, 4)))
    if gpu_type:
        console.print(Padding(f"[green]GPU available:[/green] {gpu_name} ({gpu_type})", (0, 0, 1, 4)))
    else:
        console.print(Padding("[yellow]GPU:[/yellow] Not available (using CPU)", (0, 0, 1, 4)))

    gpu_baseline_kb = get_gpu_baseline()
    results = [benchmark(name, func, manifold) for name, func in strategies.items()]

    if pytorch_strategy and gpu_type:
        device = "mps" if gpu_type == "MPS" else "cuda"
        results.append(benchmark(f"PyTorch ({gpu_type})", lambda m: pytorch_strategy(m, device), manifold))

    for r in results:
        r['gpu_memory_kb'] = max(0, r['gpu_memory_kb'] - gpu_baseline_kb)

    display_results(console, results, manifold.shape, gpu_type, result_label)
