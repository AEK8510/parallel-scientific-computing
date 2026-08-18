# Parallel Scientific Computing

A compact, reproducible Python project demonstrating how the same scientific workload can be implemented and benchmarked with sequential Python, NumPy vectorization, and process-based parallelism.

The repository is designed as a public engineering portfolio project focused on scientific computing, performance analysis, and HPC-oriented software practices.

## What this project demonstrates

- clear separation between numerical kernels and benchmarking code;
- sequential and vectorized implementations of the same workload;
- process-based parallel execution with `multiprocessing`;
- reproducible timing experiments;
- speedup and parallel-efficiency calculations;
- automated tests and continuous integration;
- practical discussion of when parallelism helps — and when overhead dominates.

## Scientific workload

The example workload evaluates a nonlinear scalar function over a one-dimensional numerical domain:

```text
f(x) = sin(x) * exp(-x² / scale) + log1p(x²)
```

It is deliberately simple enough to inspect while still being useful for comparing execution strategies.

## Implementations

| Strategy | Description |
|---|---|
| Sequential Python | Element-by-element evaluation in a Python loop |
| NumPy vectorized | Array-based execution using NumPy ufuncs |
| Multiprocessing | Domain split into chunks evaluated by worker processes |

## Repository structure

```text
parallel-scientific-computing/
├── src/parallel_scientific_computing/
│   ├── __init__.py
│   ├── kernels.py
│   ├── parallel.py
│   ├── benchmarking.py
│   └── metrics.py
├── tests/
├── examples/
│   ├── benchmark_demo.py
│   └── scaling_demo.py
├── .github/workflows/tests.yml
├── pyproject.toml
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Run the benchmark demo

```bash
python examples/benchmark_demo.py
```

Example output:

```text
Strategy             Mean [s]    Speedup
sequential            0.420       1.00
numpy                  0.018      23.33
multiprocessing-4      0.150       2.80
```

Actual timings depend on CPU, operating system, Python version, process startup cost, and problem size. The project intentionally reports measured results instead of embedding fixed performance claims.

## Run a scaling study

```bash
python examples/scaling_demo.py
```

This measures the multiprocessing implementation with several worker counts and reports:

- elapsed time;
- speedup relative to the one-worker baseline;
- parallel efficiency.

## Performance interpretation

A central HPC lesson is that more workers do not automatically imply faster execution. For small problems, process creation, serialization, memory traffic, and scheduling can cost more than the work itself. As the workload grows, parallel execution can become worthwhile.

NumPy may outperform Python multiprocessing for this particular kernel because vectorized ufuncs execute optimized native loops with very little Python overhead. That is an important result rather than a failure: choosing the right computational strategy matters more than parallelizing by default.

## Quality practices

The repository includes:

- type hints;
- small, testable functions;
- input validation;
- deterministic numerical comparisons;
- `pytest` tests;
- `ruff` linting;
- GitHub Actions CI on multiple Python versions.

## Scope

This is a self-contained public demonstration project. It does not contain confidential employer code, proprietary research implementations, or restricted datasets.

## Author

**Ahmed El Kerim, PhD**  
Research Engineer — Scientific Computing | HPC | AI/ML | Engineering Simulation
