"""Metriques de performance et fonctions de visualisation pour le backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wealth_from_log_returns(log_returns: pd.Series, base: float = 100.0) -> pd.Series:
    """Transforme des log-rendements en indice de richesse."""
    return base * np.exp(log_returns.cumsum())


def max_drawdown(log_returns: pd.Series) -> float:
    """Calcule le drawdown maximum a partir de log-rendements."""
    wealth = wealth_from_log_returns(log_returns)
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def jensen_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """Calcule l'alpha de Jensen annualise dans un cadre CAPM."""
    aligned = pd.concat(
        [portfolio_returns, benchmark_returns, risk_free_returns], axis=1
    ).dropna()
    aligned.columns = ["rp", "rb", "rf"]

    excess_port = aligned["rp"] - aligned["rf"]
    excess_bench = aligned["rb"] - aligned["rf"]

    var_bench = np.var(excess_bench.values)
    if var_bench <= 1e-12:
        return np.nan

    beta = np.cov(excess_port.values, excess_bench.values, ddof=1)[0, 1] / var_bench
    alpha_daily = excess_port.mean() - beta * excess_bench.mean()
    return float(alpha_daily * trading_days)


def performance_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_returns: pd.Series,
    trading_days: int = 252,
) -> pd.Series:
    """Calcule rendement, volatilite, Sharpe, MDD et alpha de Jensen annualises."""
    aligned = pd.concat(
        [portfolio_returns, benchmark_returns, risk_free_returns], axis=1
    ).dropna()
    aligned.columns = ["rp", "rb", "rf"]

    rp = aligned["rp"]
    rb = aligned["rb"]
    rf = aligned["rf"]

    ann_return = float(np.exp(rp.mean() * trading_days) - 1.0)
    ann_vol = float(rp.std(ddof=1) * np.sqrt(trading_days))
    sharpe = float(((rp - rf).mean() / rp.std(ddof=1)) * np.sqrt(trading_days)) if rp.std(ddof=1) > 0 else np.nan
    mdd = max_drawdown(rp)
    alpha = jensen_alpha(rp, rb, rf, trading_days)

    return pd.Series(
        {
            "Rendement annuel": ann_return,
            "Volatilité annuelle": ann_vol,
            "Ratio de Sharpe": sharpe,
            "Max Drawdown": mdd,
            "Alpha de Jensen": alpha,
        }
    )


def format_metrics_for_display(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Met a l'echelle les metriques en pourcentage pour affichage."""
    display_df = metrics_df.copy()
    percentage_rows = [
        "Rendement annuel",
        "Volatilité annuelle",
        "Max Drawdown",
        "Alpha de Jensen",
    ]
    for row in percentage_rows:
        if row in display_df.index:
            display_df.loc[row] = display_df.loc[row] * 100.0
    return display_df


def plot_cumulative_wealth(
    returns_map: dict[str, pd.Series],
    title: str,
    figsize: tuple[int, int] = (11, 6),
) -> None:
    """Trace les courbes de richesse cumulee a partir d'un dictionnaire de rendements."""
    plt.figure(figsize=figsize)
    for label, ret in returns_map.items():
        wealth = wealth_from_log_returns(ret)
        plt.plot(wealth.index, wealth, label=label, linewidth=2)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Valeur cumulée (base 100)")
    plt.legend()
    plt.tight_layout()
    plt.show()
