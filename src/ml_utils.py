"""Fonctions utilitaires pour l'apprentissage supervise avec XGBoost."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.data_utils import split_in_sample_oos
from src.features import build_features_target, split_features_target


def fit_predict_xgb_for_asset(
    returns: pd.Series,
    asset_name: str,
    in_sample_end: str,
    oos_start: str,
    oos_end: str,
    n_splits: int = 5,
    xgb_params: dict | None = None,
) -> tuple[pd.Series, pd.Series, dict]:
    """Entraine un modele XGBoost pour un actif et predit mu en OOS."""
    dataset = build_features_target(returns)
    in_mask, oos_mask = split_in_sample_oos(dataset.index, in_sample_end, oos_start, oos_end)
    train_df = dataset.loc[in_mask]
    oos_df = dataset.loc[oos_mask]

    if len(train_df) < 200:
        raise RuntimeError(f"Pas assez d'observations in-sample pour {asset_name}.")
    if oos_df.empty:
        raise RuntimeError(f"Aucune observation OOS pour {asset_name}.")

    x_train, y_train = split_features_target(train_df)
    x_oos, _ = split_features_target(oos_df)

    params = {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    }
    if xgb_params:
        params.update(xgb_params)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_folds: list[float] = []

    for train_idx, val_idx in tscv.split(x_train):
        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_model = XGBRegressor(**params)
        fold_model.fit(x_tr, y_tr)
        pred_val = fold_model.predict(x_val)
        rmse_folds.append(float(np.sqrt(mean_squared_error(y_val, pred_val))))

    final_model = XGBRegressor(**params)
    final_model.fit(x_train, y_train)

    mu_oos = pd.Series(final_model.predict(x_oos), index=x_oos.index, name=asset_name)
    feature_importance = pd.Series(
        final_model.feature_importances_,
        index=x_train.columns,
        name=asset_name,
    )
    cv_stats = {
        "asset": asset_name,
        "cv_rmse_mean": float(np.mean(rmse_folds)),
        "cv_rmse_std": float(np.std(rmse_folds)),
    }
    return mu_oos, feature_importance, cv_stats


def fit_predict_xgb_multi_asset(
    returns_df: pd.DataFrame,
    assets: list[str],
    in_sample_end: str,
    oos_start: str,
    oos_end: str,
    n_splits: int = 5,
    xgb_params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Entraine un modele par actif et retourne mu, importances et stats CV."""
    mu_list: list[pd.Series] = []
    imp_list: list[pd.Series] = []
    cv_rows: list[dict] = []

    for asset in assets:
        mu_oos, importance, cv_stats = fit_predict_xgb_for_asset(
            returns=returns_df[asset],
            asset_name=asset,
            in_sample_end=in_sample_end,
            oos_start=oos_start,
            oos_end=oos_end,
            n_splits=n_splits,
            xgb_params=xgb_params,
        )
        mu_list.append(mu_oos)
        imp_list.append(importance)
        cv_rows.append(cv_stats)

    mu_df = pd.concat(mu_list, axis=1).dropna(how="any")
    imp_df = pd.concat(imp_list, axis=1)
    cv_df = pd.DataFrame(cv_rows).set_index("asset").sort_index()
    return mu_df, imp_df, cv_df
