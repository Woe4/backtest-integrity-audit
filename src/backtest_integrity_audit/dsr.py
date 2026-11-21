"""Deflated Sharpe Ratio calculations for a matrix of strategy returns."""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, isfinite, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

_EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class DSRResult:
    """Summary of a selected strategy's Deflated Sharpe Ratio."""

    selected_strategy: str
    selected_sharpe: float
    expected_max_sharpe: float
    dsr: float
    observations: int
    strategies_tested: int
    effective_trials: int
    skewness: float
    excess_kurtosis: float


def expected_max_sharpe(sharpes: pd.Series, n_trials: int | None = None) -> float:
    """Approximate the expected maximum Sharpe across independent trials.

    The extreme-value approximation is mean(SR) + std(SR) * E[max(Z_1,...,Z_N)].
    All Sharpe estimates and the returned benchmark use the same per-period
    (non-annualized) scale.
    """
    values = pd.to_numeric(sharpes, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("At least one finite trial Sharpe is required.")
    trials = int(n_trials if n_trials is not None else values.size)
    if trials < 1:
        raise ValueError("n_trials must be at least 1.")

    mean = float(np.mean(values))
    if trials == 1:
        return mean
    sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    normal = NormalDist()
    gamma = _EULER_MASCHERONI
    max_z = (1.0 - gamma) * normal.inv_cdf(1.0 - 1.0 / trials)
    max_z += gamma * normal.inv_cdf(1.0 - 1.0 / (trials * np.e))
    return mean + sigma * max_z


def deflated_sharpe_ratio(
    returns: pd.DataFrame,
    selected_strategy: str | None = None,
    *,
    n_trials: int | None = None,
    risk_free_rate: float = 0.0,
) -> DSRResult:
    """Calculate DSR for a selected column of a T x N simple-return matrix.

    By default, selects the column with the highest compounded total return.
    ``n_trials`` is the effective independent-trial count and defaults to N.
    The Sharpe ratio remains per-period, matching the observation count used
    in the sampling-error adjustment.
    """
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty T x N pandas DataFrame.")
    if returns.columns.has_duplicates:
        raise ValueError("Strategy columns must have unique names.")
    if len(returns) < 3:
        raise ValueError("At least three return observations are required.")
    if not isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite and expressed per period.")

    numeric = returns.apply(pd.to_numeric, errors="coerce")
    array = numeric.to_numpy(dtype=float)
    if numeric.isna().any().any() or not np.isfinite(array).all():
        raise ValueError("returns must contain only finite numeric values; handle missing data first.")
    if (numeric <= -1.0).any().any():
        raise ValueError("Simple returns must be greater than -100%.")

    if selected_strategy is None:
        selected_strategy = str(((1.0 + numeric).prod(axis=0) - 1.0).idxmax())
    if selected_strategy not in numeric.columns:
        raise KeyError(f"Unknown strategy column: {selected_strategy!r}")

    excess = numeric - risk_free_rate
    volatilities = excess.std(axis=0, ddof=1)
    trial_sharpes = (excess.mean(axis=0) / volatilities.replace(0.0, np.nan)).dropna()
    if selected_strategy not in trial_sharpes.index:
        raise ValueError("The selected strategy has zero volatility, so its Sharpe ratio is undefined.")

    selected_returns = excess[selected_strategy]
    selected_sharpe = float(trial_sharpes[selected_strategy])
    skewness = float(selected_returns.skew())
    excess_kurtosis = float(selected_returns.kurt())
    if not isfinite(skewness) or not isfinite(excess_kurtosis):
        raise ValueError("Selected returns do not have finite skewness and kurtosis.")

    benchmark = expected_max_sharpe(trial_sharpes, n_trials=n_trials)
    variance_term = (
        1.0 - skewness * selected_sharpe
        + ((excess_kurtosis + 2.0) / 4.0) * selected_sharpe**2
    )
    if not isfinite(variance_term) or variance_term <= 0.0:
        raise ValueError("The skew/kurtosis adjustment is non-positive; DSR is undefined for these inputs.")

    z_score = (selected_sharpe - benchmark) * sqrt(len(numeric) - 1) / sqrt(variance_term)
    probability = 0.5 * erfc(-z_score / sqrt(2.0))
    effective_trials = int(n_trials if n_trials is not None else numeric.shape[1])
    if effective_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    return DSRResult(
        selected_strategy=selected_strategy,
        selected_sharpe=selected_sharpe,
        expected_max_sharpe=benchmark,
        dsr=probability,
        observations=len(numeric),
        strategies_tested=numeric.shape[1],
        effective_trials=effective_trials,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )
