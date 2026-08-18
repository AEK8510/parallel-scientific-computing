"""Process-based parallel implementation of the numerical workload."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.typing import NDArray

from .kernels import vectorized_kernel


def _evaluate_chunk(
    values: NDArray[np.float64],
    scale: float,
) -> NDArray[np.float64]:
    """Evaluate one independent chunk in a worker process."""
    return vectorized_kernel(values, scale=scale)


def multiprocessing_kernel(
    values: NDArray[np.float64],
    workers: int = 2,
    scale: float = 25.0,
) -> NDArray[np.float64]:
    """Evaluate the workload across multiple Python processes.

    Args:
        values: One-dimensional numerical input.
        workers: Number of worker processes. Must be at least one.
        scale: Positive decay scale used by the kernel.

    Returns:
        Output values in the same order as the input.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if array.size == 0:
        return np.array([], dtype=np.float64)
    if workers == 1:
        return vectorized_kernel(array, scale=scale)

    chunks = [chunk for chunk in np.array_split(array, workers) if chunk.size]
    with ProcessPoolExecutor(max_workers=min(workers, len(chunks))) as executor:
        futures = [executor.submit(_evaluate_chunk, chunk, scale) for chunk in chunks]
        parts = [future.result() for future in futures]
    return np.concatenate(parts)
