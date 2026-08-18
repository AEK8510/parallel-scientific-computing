import pytest

from parallel_scientific_computing.metrics import parallel_efficiency, speedup


def test_speedup() -> None:
    assert speedup(10.0, 2.5) == pytest.approx(4.0)


def test_efficiency() -> None:
    assert parallel_efficiency(3.2, 4) == pytest.approx(0.8)


@pytest.mark.parametrize(
    "baseline, parallel",
    [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)],
)
def test_speedup_requires_positive_timings(
    baseline: float,
    parallel: float,
) -> None:
    with pytest.raises(ValueError):
        speedup(baseline, parallel)


def test_efficiency_requires_workers() -> None:
    with pytest.raises(ValueError):
        parallel_efficiency(1.0, 0)
