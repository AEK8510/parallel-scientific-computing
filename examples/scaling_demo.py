"""Run a simple process-count scaling study."""

from __future__ import annotations

import os

import numpy as np

from parallel_scientific_computing import (
    benchmark_callable,
    multiprocessing_kernel,
    parallel_efficiency,
    speedup,
)


def main() -> None:
    values = np.linspace(-50.0, 50.0, 1_000_000, dtype=np.float64)
    max_workers = max(1, min(8, os.cpu_count() or 1))
    worker_counts = [count for count in (1, 2, 4, 8) if count <= max_workers]

    results = []
    for workers in worker_counts:
        result = benchmark_callable(
            f"workers={workers}",
            lambda workers=workers: multiprocessing_kernel(values, workers=workers),
            repeats=3,
            warmup=0,
        )
        results.append((workers, result))

    baseline = results[0][1].mean_seconds
    print(f"{'Workers':>8} {'Mean [s]':>12} {'Speedup':>10} {'Efficiency':>12}")
    for workers, result in results:
        value = speedup(baseline, result.mean_seconds)
        efficiency = parallel_efficiency(value, workers)
        print(
            f"{workers:>8d} {result.mean_seconds:>12.6f} "
            f"{value:>10.2f} {efficiency:>12.2%}"
        )


if __name__ == "__main__":
    main()
