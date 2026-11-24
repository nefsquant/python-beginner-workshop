# Python for quantitative finance

A one-hour introduction to Python, with additional examples for practice.
Each notebook starts with small, non-financial examples before applying the operations to finance.
All financial data is synthetic; random examples use fixed seeds.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync
uv run jupyter lab
```

uv installs Python 3.14 if needed. Open the notebooks in `lessons/` in order and
press **Shift+Enter** to run a cell. In VS Code, select the Python environment in `.venv`.

## Notebooks

| Notebook | Start with | Apply to |
| --- | --- | --- |
| [1. Python and returns](lessons/01_python_and_returns.ipynb) | Values, collections, conditions, loops and functions | Profit, compounding and fees |
| [2. NumPy and simulation](lessons/02_numpy_and_simulation.ipynb) | Arrays, masks, summaries, axes, broadcasting and random samples | Returns, risk and simulated paths |
| [3. pandas, Polars and backtesting](lessons/03_pandas_polars_and_backtesting.ipynb) | Selection, new columns, missing values, groups and rolling windows | Price histories, signals and trading costs |
| [4. SciPy and native code](lessons/04_scipy_and_native_code.ipynb) | Distributions, scalar optimisation and constrained colour mixing | Portfolio allocation |

Optional sections cover joins, CSV files, lazy queries, performance comparisons and the C++/Rust bridge.

Each notebook has a matching `.py` file that can also run as a script:

```bash
uv run python lessons/02_numpy_and_simulation.py
```

The price generator and Python bindings are in `src/quant_workshop/`.

## C++ and Rust

The native examples are optional. To build either one with an installed compiler:

```bash
uv run python native/build.py cpp
uv run python native/build.py rust
```

C++ builds on macOS and Linux; on Windows, use WSL or the Rust example.
Rust needs `rustc` and a platform linker. Notebook 4 uses NumPy if neither library is built.
