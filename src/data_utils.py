"""Fonctions utilitaires pour le telechargement et le pretraitement des donnees."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


def download_adj_close(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Telecharge les prix ajustes de cloture pour une liste de tickers.

    Parametres
    ----------
    tickers:
        Iterable de symboles Yahoo Finance.
    start:
        Date de debut au format ISO (YYYY-MM-DD).
    end:
        Date de fin au format ISO (YYYY-MM-DD), exclue par yfinance.

    Retour
    ------
    pd.DataFrame
        Prix ajustes de cloture indexes par date.
    """
    tickers = list(tickers)
    if not tickers:
        raise ValueError("`tickers` must contain at least one symbol.")

    panel = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if panel.empty:
        raise RuntimeError("No data returned by yfinance.")

    adj_close = panel["Adj Close"].dropna(how="all")
    if adj_close.empty:
        raise RuntimeError("Adjusted close data is empty after cleanup.")

    return adj_close


def compute_log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calcule les rendements logarithmiques a partir de prix."""
    return np.log(prices / prices.shift(1)).dropna()


def split_in_sample_oos(
    index: pd.DatetimeIndex,
    in_sample_end: str,
    oos_start: str,
    oos_end: str,
) -> tuple[pd.Series, pd.Series]:
    """Cree des masques booleens pour les periodes in-sample et out-of-sample."""
    in_mask = index <= pd.Timestamp(in_sample_end)
    oos_mask = (index >= pd.Timestamp(oos_start)) & (index <= pd.Timestamp(oos_end))
    return in_mask, oos_mask


def get_daily_risk_free_rate(
    start: str,
    end: str,
    trading_days: int = 252,
    rf_tickers: tuple[str, ...] = ("^IRX", "^TNX", "^FVX"),
) -> pd.Series:
    """Recupere un taux sans risque annuel proxy puis le convertit en taux journalier.

    La fonction teste chaque ticker de `rf_tickers` jusqu'a trouver des donnees.
    """
    rf_yield: pd.Series | None = None

    for ticker in rf_tickers:
        try:
            data = yf.download(
                [ticker],
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                candidate = data["Adj Close"][ticker].dropna()
            else:
                candidate = data["Adj Close"].dropna()

            if not candidate.empty:
                rf_yield = candidate
                break
        except Exception:
            continue

    if rf_yield is None:
        raise RuntimeError(
            f"Could not download any risk-free yield from tickers {rf_tickers}."
        )

    rf_annual = rf_yield / 100.0
    rf_daily = (1.0 + rf_annual) ** (1.0 / trading_days) - 1.0
    return rf_daily


def align_series_on_common_index(*series_or_frames: pd.Series | pd.DataFrame) -> tuple:
    """Aligne plusieurs objets pandas sur leur intersection de dates commune."""
    if len(series_or_frames) < 2:
        raise ValueError("Provide at least two series/dataframes to align.")

    common_index = series_or_frames[0].index
    for item in series_or_frames[1:]:
        common_index = common_index.intersection(item.index)

    return tuple(item.loc[common_index] for item in series_or_frames)
