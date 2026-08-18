"""Compare sequential, NumPy, and multiprocessing implementations."""

from __future__ import annotations

import os

import numpy as np

from parallel_scientific_computing import (
    benchmark_callable,
    multiprocessing_kernel,
    sequential_kernel,
    speedup,
    vectorized_kernel,
)


def main() -> None:
    values = np.linspace(-25.0, 25.0, 500_000, dtype=np.float64)
    workers = max(1, min(4, os.cpu_count() or 1))

    sequential = benchmark_callable(
        "sequential",
        lambda: sequential_kernel(values),
        repeats=3,
        warmup=1,
    )
    numpy_result = benchmark_callable(
        "numpy",
        lambda: vectorized_kernel(values),
        repeats=5,
        warmup=1,
    )
    multiprocessing = benchmark_callable(
        f"multiprocessing-{workers}",
        lambda: multiprocessing_kernel(values, workers=workers),
        repeats=3,
        warmup=0,
    )

    assert np.allclose(sequential.output, numpy_result.output)
    assert np.allclose(sequential.output, multiprocessing.output)

    print(f"{'Strategy':<22} {'Mean [s]':>12} {'Speedup':>10}")
    for result in (sequential, numpy_result, multiprocessing):
        value = speedup(sequential.mean_seconds, result.mean_seconds)
        print(f"{result.name:<22} {result.mean_seconds:>12.6f} {value:>10.2f}")


if __name__ == "__main__":
    main()
