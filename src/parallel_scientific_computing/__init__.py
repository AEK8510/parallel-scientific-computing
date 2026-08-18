"""Utilities for benchmarking small scientific computing workloads."""

from .benchmarking import BenchmarkResult, benchmark_callable
from .kernels import sequential_kernel, vectorized_kernel
from .metrics import parallel_efficiency, speedup
from .parallel import multiprocessing_kernel

__all__ = [
    "BenchmarkResult",
    "benchmark_callable",
    "multiprocessing_kernel",
    "parallel_efficiency",
    "sequential_kernel",
    "speedup",
    "vectorized_kernel",
]
