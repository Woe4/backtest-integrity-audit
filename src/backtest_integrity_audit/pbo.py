"""Probability of Backtest Overfitting via combinatorially symmetric CV."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Literal

import numpy as np
import pandas as pd

PerformanceMetric = Literal["sharpe", "total_return"]


@dataclass(frozen=True)
class PBOResult:
    """CSCV PBO estimate plus per-split winners and out-of-sample ranks."""

    pbo: float
    splits: pd.DataFrame
    n_blocks: int
    n_splits: int
    n_strategies: int
    metric: PerformanceMetric


def _score(returns: pd.DataFrame, metric: PerformanceMetric, risk_free_rate: float) -> pd.Series:
    if metric == "total_return":
        return (1.0 + returns).prod(axis=0) - 1.0
    excess = returns - risk_free_rate
    means = excess.mean(axis=0)
    stds = excess.std(axis=0, ddof=1)
    # A flat return series has no defined Sharpe; assigning zero lets it take
    # a neutral rank without allowing division-by-zero to break a split.
    return means.div(stds.where(stds.ne(0.0))).fillna(0.0)


def cscv_pbo(
    returns: pd.DataFrame,
    *,
    n_blocks: int = 16,
    metric: PerformanceMetric = "sharpe",
    risk_free_rate: float = 0.0,
) -> PBOResult:
    """Estimate PBO by enumerating every half-block train/test partition.

    Rows are split into ``n_blocks`` contiguous, near-equal blocks. Every
    combination of half the blocks is used in sample and its complement out
    of sample. The in-sample winner is ranked among all strategies out of
    sample, with rank 1 as worst and ties receiving average ranks. PBO is the
    share of splits where relative rank ``rank / (N + 1)`` is below 0.5.
    """
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty T x N pandas DataFrame.")
    if returns.columns.has_duplicates:
        raise ValueError("Strategy columns must have unique names.")
    if returns.shape[1] < 2:
        raise ValueError("At least two strategy columns are required.")
    if isinstance(n_blocks, bool) or not isinstance(n_blocks, int):
        raise TypeError("n_blocks must be an integer.")
    if n_blocks < 4 or n_blocks % 2:
        raise ValueError("n_blocks must be an even number of at least 4 (typically 8–16).")
    if n_blocks > len(returns):
        raise ValueError("n_blocks cannot exceed the number of observations.")
    if metric not in ("sharpe", "total_return"):
        raise ValueError("metric must be 'sharpe' or 'total_return'.")
    if not isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite and expressed per period.")

    numeric = returns.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if numeric.isna().any().any() or not np.isfinite(values).all():
        raise ValueError("returns must contain only finite numeric values; handle missing data first.")
    if (numeric <= -1.0).any().any():
        raise ValueError("Simple returns must be greater than -100%.")

    blocks = np.array_split(np.arange(len(numeric)), n_blocks)
    half = n_blocks // 2
    records: list[dict[str, object]] = []
    below_median = 0
    all_blocks = set(range(n_blocks))

    for train_tuple in combinations(range(n_blocks), half):
        train_blocks = set(train_tuple)
        test_blocks = all_blocks - train_blocks
        train_rows = np.sort(np.concatenate([blocks[i] for i in train_tuple]))
        test_rows = np.sort(np.concatenate([blocks[i] for i in sorted(test_blocks)]))
        train_returns = numeric.iloc[train_rows]
        test_returns = numeric.iloc[test_rows]

        in_sample_scores = _score(train_returns, metric, risk_free_rate)
        winner = str(in_sample_scores.idxmax())
        out_sample_scores = _score(test_returns, metric, risk_free_rate)
        ranks = out_sample_scores.rank(method="average", ascending=True)
        rank = float(ranks[winner])
        relative_rank = rank / (numeric.shape[1] + 1.0)
        is_below_median = relative_rank < 0.5
        below_median += int(is_below_median)

        records.append({
            "train_blocks": tuple(sorted(train_blocks)),
            "test_blocks": tuple(sorted(test_blocks)),
            "in_sample_winner": winner,
            "in_sample_score": float(in_sample_scores[winner]),
            "oos_score": float(out_sample_scores[winner]),
            "oos_rank": rank,
            "oos_relative_rank": relative_rank,
            "below_median": is_below_median,
        })

    split_frame = pd.DataFrame.from_records(records)
    return PBOResult(
        pbo=below_median / len(records),
        splits=split_frame,
        n_blocks=n_blocks,
        n_splits=len(records),
        n_strategies=numeric.shape[1],
        metric=metric,
    )
