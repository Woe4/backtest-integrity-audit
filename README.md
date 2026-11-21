# Backtest Integrity Audit

A Python project for auditing backtest integrity.

## Getting started

Create and activate the local environment from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project in editable mode:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

The package source lives in `src/backtest_integrity_audit/`.

## Generate sample backtests

Install the project and generate a seeded Gaussian noise sweep plus SPY and QQQ
moving-average sweeps:

```powershell
python -m pip install -e .
backtest-samples
```

The default sweep compares all combinations of fast windows 5, 10, and 20 with
slow windows 30, 50, and 100. Each Parquet return matrix has dates in rows and
one strategy per column. Signals are lagged by one period to avoid same-day
lookahead. The highest compounded total return is printed as the selected
strategy. Generated matrices and cached adjusted ETF prices are written under
`data/cache/`; reruns reuse cached ETF prices. Use `--refresh` to download them
again. You can change the noise length and ETF date range with `--periods`,
`--start`, and `--end`.

## Calculate a Deflated Sharpe Ratio

Pass a T x N return matrix to the DSR function. By default, it selects the
strategy with the highest compounded total return and treats its N columns as
the tested trials:

```python
import pandas as pd
from backtest_integrity_audit import deflated_sharpe_ratio

returns = pd.read_parquet("data/cache/noise_returns.parquet")
result = deflated_sharpe_ratio(returns)
print(result)
```

`result.dsr` estimates the probability that the selected strategy's true
Sharpe exceeds the expected maximum Sharpe from the search. The Sharpe and
benchmark are per-period, not annualized. The benchmark uses the sample mean
and standard deviation of the tested strategies' Sharpe estimates with the
extreme-value approximation for the expected maximum. By default, the nominal
column count is used as the independent-trial count; pass `n_trials=` to use an
effective independent-trial estimate when variants are correlated. The DSR
sampling adjustment uses the selected return series' skewness, excess
kurtosis, and length.
