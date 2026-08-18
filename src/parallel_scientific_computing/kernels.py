"""Numerical kernels used by the benchmark examples."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def _validate_scale(scale: float) -> None:
    if scale <= 0.0:
        raise ValueError("scale must be strictly positive")


def scalar_function(x: float, scale: float = 25.0) -> float:
    """Evaluate the scalar reference function.

    Args:
        x: Input value.
        scale: Positive decay scale.

    Returns:
        Evaluated scalar value.
    """
    _validate_scale(scale)
    return math.sin(x) * math.exp(-(x * x) / scale) + math.log1p(x * x)


def sequential_kernel(
    values: NDArray[np.float64],
    scale: float = 25.0,
) -> NDArray[np.float64]:
    """Evaluate the workload with an explicit Python loop."""
    _validate_scale(scale)
    result = np.empty_like(values, dtype=np.float64)
    for index, value in enumerate(values):
        result[index] = math.sin(float(value)) * math.exp(
            -(float(value) ** 2) / scale
        ) + math.log1p(float(value) ** 2)
    return result


def vectorized_kernel(
    values: NDArray[np.float64],
    scale: float = 25.0,
) -> NDArray[np.float64]:
    """Evaluate the same workload with vectorized NumPy operations."""
    _validate_scale(scale)
    array = np.asarray(values, dtype=np.float64)
    return np.sin(array) * np.exp(-(array**2) / scale) + np.log1p(array**2)
