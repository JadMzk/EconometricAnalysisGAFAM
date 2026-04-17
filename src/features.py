"""Fonctions de feature engineering pour la prevision de rendements."""

from __future__ import annotations

import pandas as pd


def build_features_target(
    returns: pd.Series,
    lags: tuple[int, ...] = (1, 3, 5),
    moving_averages: tuple[int, ...] = (10, 20),
    volatility_window: int = 20,
    target_horizon: int = 1,
) -> pd.DataFrame:
    """Construit les variables explicatives et la cible a horizon futur.

    Parametres
    ----------
    returns:
        Serie de log-rendements pour un actif.
    lags:
        Retards utilises comme variables autoregressives.
    moving_averages:
        Fenetres de moyennes mobiles (en jours).
    volatility_window:
        Fenetre de volatilite glissante (ecart-type).
    target_horizon:
        Horizon de prediction en jours. Par defaut, t+1.

    Retour
    ------
    pd.DataFrame
        Jeu de donnees contenant les features et la colonne `target`.
    """
    if returns.empty:
        raise ValueError("Input returns series is empty.")

    df = pd.DataFrame(index=returns.index)

    for lag in lags:
        df[f"lag_{lag}"] = returns.shift(lag)

    for window in moving_averages:
        df[f"ma_{window}"] = returns.rolling(window).mean()

    df[f"vol_{volatility_window}"] = returns.rolling(volatility_window).std()
    df["target"] = returns.shift(-target_horizon)

    return df.dropna()


def split_features_target(
    dataset: pd.DataFrame,
    target_col: str = "target",
) -> tuple[pd.DataFrame, pd.Series]:
    """Separe un dataframe de features en objets X et y."""
    if target_col not in dataset.columns:
        raise KeyError(f"Target column `{target_col}` not found.")

    x = dataset.drop(columns=[target_col])
    y = dataset[target_col]
    return x, y
