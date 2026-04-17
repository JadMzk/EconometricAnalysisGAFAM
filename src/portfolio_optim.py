"""Fonctions d'optimisation de portefeuille (modele de risque + allocations)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def ledoit_wolf_covariance(returns_window: np.ndarray) -> np.ndarray:
    """Estime la matrice de covariance via shrinkage de Ledoit-Wolf."""
    if returns_window.ndim != 2:
        raise ValueError("`returns_window` must be a 2D array.")
    if returns_window.shape[0] < 2:
        raise ValueError("At least two observations are required for covariance.")

    lw = LedoitWolf().fit(returns_window)
    return lw.covariance_


def min_variance_weights(
    covariance: np.ndarray,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Calcule les poids long-only de variance minimale avec somme(w)=1."""
    n_assets = covariance.shape[0]
    cov_reg = covariance + 1e-8 * np.eye(n_assets)

    def objective(weights: np.ndarray) -> float:
        return float(weights @ cov_reg @ weights)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    box = tuple(bounds for _ in range(n_assets))
    w0 = np.full(n_assets, 1.0 / n_assets)

    try:
        result = minimize(
            objective,
            x0=w0,
            method="SLSQP",
            constraints=constraints,
            bounds=box,
        )
        if not result.success:
            raise ValueError(result.message)
        weights = result.x
    except Exception:
        weights = w0

    weights = np.clip(weights, bounds[0], bounds[1])
    total = weights.sum()
    return weights / total if total > 0 else w0


def max_sharpe_weights(
    mu: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Calcule les poids long-only de Sharpe maximal avec somme(w)=1.

    L'optimiseur minimise l'oppose du ratio de Sharpe.
    """
    n_assets = covariance.shape[0]
    cov_reg = covariance + 1e-8 * np.eye(n_assets)

    def objective(weights: np.ndarray) -> float:
        port_ret = float(weights @ mu)
        port_vol = float(np.sqrt(np.clip(weights @ cov_reg @ weights, 1e-12, None)))
        return -((port_ret - risk_free_rate) / port_vol)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    box = tuple(bounds for _ in range(n_assets))
    w0 = np.full(n_assets, 1.0 / n_assets)

    try:
        result = minimize(
            objective,
            x0=w0,
            method="SLSQP",
            constraints=constraints,
            bounds=box,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            raise ValueError(result.message)
        weights = result.x
    except Exception:
        weights = w0

    weights = np.clip(weights, bounds[0], bounds[1])
    total = weights.sum()
    return weights / total if total > 0 else w0


def backtest_min_variance(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    assets: list[str],
    oos_start: str,
    oos_end: str,
    rolling_window: int = 60,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Execute un backtest out-of-sample d'un portefeuille Min Variance.

    Les poids sont calcules au temps t sur une fenetre glissante passee
    et appliques au rendement realise en t+1.
    """
    next_assets = asset_returns.shift(-1)
    next_bench = benchmark_returns.shift(-1)

    port_ret: list[float] = []
    bench_ret: list[float] = []
    dates: list[pd.Timestamp] = []
    weights_hist: list[np.ndarray] = []

    for current_date in asset_returns.index:
        if current_date < pd.Timestamp(oos_start) or current_date > pd.Timestamp(oos_end):
            continue

        window = asset_returns.loc[:current_date, assets].tail(rolling_window)
        if len(window) < rolling_window:
            continue

        sigma_t = ledoit_wolf_covariance(window.values)
        w_t = min_variance_weights(sigma_t, bounds=bounds)

        r_next = next_assets.loc[current_date, assets]
        b_next = next_bench.loc[current_date]
        if r_next.isna().any() or pd.isna(b_next):
            continue

        port_ret.append(float(np.dot(w_t, r_next.values)))
        bench_ret.append(float(b_next))
        dates.append(current_date)
        weights_hist.append(w_t)

    port_series = pd.Series(port_ret, index=dates, name="Min Variance")
    bench_series = pd.Series(bench_ret, index=dates, name="Benchmark")
    weights_df = pd.DataFrame(weights_hist, index=dates, columns=assets)
    return port_series, bench_series, weights_df


def backtest_max_sharpe(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    mu_predictions: pd.DataFrame,
    risk_free_daily: pd.Series,
    assets: list[str],
    rolling_window: int = 60,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Execute un backtest OOS d'un portefeuille dynamique Max Sharpe.

    A chaque date t, on utilise:
    - mu predit par ML,
    - covariance Ledoit-Wolf sur la fenetre passee,
    puis on applique les poids au rendement realise en t+1.
    """
    next_assets = asset_returns.shift(-1)
    next_bench = benchmark_returns.shift(-1)

    port_ret: list[float] = []
    bench_ret: list[float] = []
    dates: list[pd.Timestamp] = []
    weights_hist: list[np.ndarray] = []

    for current_date in mu_predictions.index:
        window = asset_returns.loc[:current_date, assets].tail(rolling_window)
        if len(window) < rolling_window:
            continue

        sigma_t = ledoit_wolf_covariance(window.values)
        mu_t = mu_predictions.loc[current_date, assets].values
        rf_t = float(risk_free_daily.loc[current_date])
        w_t = max_sharpe_weights(mu_t, sigma_t, rf_t, bounds=bounds)

        r_next = next_assets.loc[current_date, assets]
        b_next = next_bench.loc[current_date]
        if r_next.isna().any() or pd.isna(b_next):
            continue

        port_ret.append(float(np.dot(w_t, r_next.values)))
        bench_ret.append(float(b_next))
        dates.append(current_date)
        weights_hist.append(w_t)

    port_series = pd.Series(port_ret, index=dates, name="Max Sharpe ML")
    bench_series = pd.Series(bench_ret, index=dates, name="Benchmark")
    weights_df = pd.DataFrame(weights_hist, index=dates, columns=assets)
    return port_series, bench_series, weights_df
