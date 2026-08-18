import pytest

from parallel_scientific_computing.benchmarking import benchmark_callable


def test_benchmark_result_contains_requested_repeats() -> None:
    result = benchmark_callable("answer", lambda: 42, repeats=2, warmup=0)
    assert result.output == 42
    assert len(result.timings) == 2
    assert result.mean_seconds >= 0.0


def test_invalid_repeat_count() -> None:
    with pytest.raises(ValueError):
        benchmark_callable("bad", lambda: None, repeats=0)


def test_invalid_warmup_count() -> None:
    with pytest.raises(ValueError):
        benchmark_callable("bad", lambda: None, warmup=-1)
