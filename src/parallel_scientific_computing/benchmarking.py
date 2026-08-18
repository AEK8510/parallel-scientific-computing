"""Small benchmarking helpers for reproducible timing experiments."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from time import perf_counter
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkResult(Generic[T]):
    """Timing summary for a callable."""

    name: str
    timings: tuple[float, ...]
    output: T

    @property
    def mean_seconds(self) -> float:
        """Return the arithmetic mean runtime."""
        return mean(self.timings)

    @property
    def min_seconds(self) -> float:
        """Return the fastest measured runtime."""
        return min(self.timings)

    @property
    def std_seconds(self) -> float:
        """Return sample standard deviation, or zero for one repeat."""
        return stdev(self.timings) if len(self.timings) > 1 else 0.0


def benchmark_callable(
    name: str,
    function: Callable[[], T],
    *,
    repeats: int = 3,
    warmup: int = 1,
) -> BenchmarkResult[T]:
    """Benchmark a zero-argument callable.

    Warm-up executions are excluded from measured timings.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")

    output: T
    for _ in range(warmup):
        output = function()

    timings: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        output = function()
        timings.append(perf_counter() - start)

    return BenchmarkResult(name=name, timings=tuple(timings), output=output)
