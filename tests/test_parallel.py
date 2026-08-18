import numpy as np
import pytest

from parallel_scientific_computing.kernels import vectorized_kernel
from parallel_scientific_computing.parallel import multiprocessing_kernel


def test_parallel_matches_vectorized() -> None:
    values = np.linspace(-4.0, 4.0, 257)
    expected = vectorized_kernel(values)
    actual = multiprocessing_kernel(values, workers=2)
    assert np.allclose(actual, expected)


def test_empty_input() -> None:
    result = multiprocessing_kernel(np.array([], dtype=np.float64), workers=2)
    assert result.size == 0


def test_one_worker_path() -> None:
    values = np.array([1.0, 2.0, 3.0])
    assert np.allclose(
        multiprocessing_kernel(values, workers=1),
        vectorized_kernel(values),
    )


def test_invalid_worker_count() -> None:
    with pytest.raises(ValueError):
        multiprocessing_kernel(np.array([1.0]), workers=0)


def test_only_one_dimensional_inputs_are_supported() -> None:
    with pytest.raises(ValueError):
        multiprocessing_kernel(np.ones((2, 2)), workers=2)
