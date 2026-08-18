import numpy as np
import pytest

from parallel_scientific_computing.kernels import (
    scalar_function,
    sequential_kernel,
    vectorized_kernel,
)


def test_scalar_and_vectorized_agree() -> None:
    values = np.array([-2.0, 0.0, 3.5], dtype=np.float64)
    expected = np.array([scalar_function(float(value)) for value in values])
    assert np.allclose(vectorized_kernel(values), expected)


def test_sequential_and_vectorized_agree() -> None:
    values = np.linspace(-10.0, 10.0, 101)
    assert np.allclose(sequential_kernel(values), vectorized_kernel(values))


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_invalid_scale_is_rejected(scale: float) -> None:
    with pytest.raises(ValueError):
        vectorized_kernel(np.array([1.0]), scale=scale)
