"""Generate return matrices for moving-average parameter sweep examples."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "cache"
FAST_WINDOWS = (5, 10, 20)
SLOW_WINDOWS = (30, 50, 100)
ETF_TICKERS = ("SPY", "QQQ")


def strategy_columns() -> list[str]:
    return [f"ma_{fast}_{slow}" for fast, slow in product(FAST_WINDOWS, SLOW_WINDOWS)]


def moving_average_sweep(prices: pd.Series) -> pd.DataFrame:
    """Return a T x N matrix of daily strategy returns, with lagged signals."""
    prices = prices.dropna().astype(float)
    asset_returns = prices.pct_change().fillna(0.0)
    matrix: dict[str, pd.Series] = {}
    for fast, slow in product(FAST_WINDOWS, SLOW_WINDOWS):
        fast_ma = prices.rolling(fast, min_periods=fast).mean()
        slow_ma = prices.rolling(slow, min_periods=slow).mean()
        position = fast_ma.gt(slow_ma).astype(float).shift(1).fillna(0.0)
        matrix[f"ma_{fast}_{slow}"] = position * asset_returns
    return pd.DataFrame(matrix, index=prices.index).rename_axis("date")


def best_strategy(returns: pd.DataFrame) -> tuple[str, float]:
    """Select the column with the highest compounded total return."""
    total_returns = (1.0 + returns).prod(axis=0) - 1.0
    name = str(total_returns.idxmax())
    return name, float(total_returns[name])


def _noise_prices(periods: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    # Zero-drift Gaussian log returns create a noise-only price path.
    log_returns = rng.normal(loc=0.0, scale=0.01, size=periods)
    index = pd.bdate_range("2015-01-01", periods=periods, name="date")
    return pd.Series(100.0 * np.exp(np.cumsum(log_returns)), index=index, name="price")


def _etf_prices(ticker: str, start: str, end: str, refresh: bool) -> pd.Series:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / f"{ticker.lower()}_adjusted_close.parquet"
    if cache_path.exists() and not refresh:
        frame = pd.read_parquet(cache_path)
        return frame["adjusted_close"].rename(ticker)

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Install project dependencies with `python -m pip install -e .`.") from exc

    history = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if history.empty:
        raise RuntimeError(f"No price history returned for {ticker}; check the ticker and date range.")
    close = history["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.rename("adjusted_close").dropna()
    close.to_frame().rename_axis("date").to_parquet(cache_path)
    return close.rename(ticker)


def _save_matrix(name: str, matrix: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}_returns.parquet"
    matrix.to_parquet(path)
    selected, total = best_strategy(matrix)
    print(f"{name}: {len(matrix)} periods x {len(matrix.columns)} strategies")
    print(f"  best by compounded total return: {selected} ({total:.2%})")
    print(f"  saved {path.relative_to(ROOT)}")
    return path


def generate_samples(
    *, periods: int = 2520, seed: int = 7, start: str = "2015-01-01",
    end: str | None = None, refresh: bool = False,
) -> list[Path]:
    """Create seeded noise and ETF T x N return matrices in Parquet format."""
    noise = _noise_prices(periods, seed)
    paths = [_save_matrix("noise", moving_average_sweep(noise))]
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    for ticker in ETF_TICKERS:
        prices = _etf_prices(ticker, start, end, refresh)
        paths.append(_save_matrix(ticker.lower(), moving_average_sweep(prices)))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=int, default=2520, help="Business-day periods in the noise sample")
    parser.add_argument("--seed", type=int, default=7, help="Noise generator seed")
    parser.add_argument("--start", default="2015-01-01", help="ETF history start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="ETF history end date (exclusive; defaults to today)")
    parser.add_argument("--refresh", action="store_true", help="Redownload ETF history instead of using cached Parquet")
    args = parser.parse_args()
    generate_samples(periods=args.periods, seed=args.seed, start=args.start, end=args.end, refresh=args.refresh)


if __name__ == "__main__":
    main()
