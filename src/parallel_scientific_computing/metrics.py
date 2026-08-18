"""Performance metrics commonly used in parallel-computing studies."""


def speedup(baseline_seconds: float, parallel_seconds: float) -> float:
    """Compute runtime speedup relative to a baseline implementation."""
    if baseline_seconds <= 0.0 or parallel_seconds <= 0.0:
        raise ValueError("timings must be strictly positive")
    return baseline_seconds / parallel_seconds


def parallel_efficiency(speedup_value: float, workers: int) -> float:
    """Compute parallel efficiency as speedup divided by worker count."""
    if speedup_value < 0.0:
        raise ValueError("speedup cannot be negative")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    return speedup_value / workers
