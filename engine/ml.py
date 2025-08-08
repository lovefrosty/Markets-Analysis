
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def _features(prices: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=prices.index)
    df["close"] = prices
    df["ret1"] = prices.pct_change(1)
    df["ret5"] = prices.pct_change(5)
    df["ret20"] = prices.pct_change(20)
    df["roll_mean20"] = prices.rolling(20).mean() / prices - 1.0
    df["roll_vol20"] = prices.pct_change().rolling(20).std()
    # RSI(14)
    delta = prices.diff()
    up = (delta.clip(lower=0)).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / (down + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df = df.dropna()
    return df

def train_models(prices: pd.Series, horizon: int = 20) -> Dict[str, Any]:
    # target: forward return over horizon
    df = _features(prices)
    y = prices.shift(-horizon) / prices - 1.0
    y = y.reindex(df.index)
    df = df.drop(columns=["close"])
    df = df.dropna(); y = y.loc[df.index].dropna()
    df = df.loc[y.index];  # align

    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=3)
    rf_preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in tscv.split(df):
        rf.fit(df.iloc[train_idx], y.iloc[train_idx])
        rf_preds[test_idx] = rf.predict(df.iloc[test_idx])
    rf_metrics = {
        "r2": float(r2_score(y, rf_preds)),
        "mae": float(mean_absolute_error(y, rf_preds)),
        "rmse": float(mean_squared_error(y, rf_preds, squared=False))
    }
    # fit on all
    rf.fit(df, y)

    xgb_metrics = None; xgb_model = None
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
        xgb_preds = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in tscv.split(df):
            xgb.fit(df.iloc[train_idx], y.iloc[train_idx])
            xgb_preds[test_idx] = xgb.predict(df.iloc[test_idx])
        xgb_metrics = {
            "r2": float(r2_score(y, xgb_preds)),
            "mae": float(mean_absolute_error(y, xgb_preds)),
            "rmse": float(mean_squared_error(y, xgb_preds, squared=False))
        }
        xgb_model = xgb
    # Current prediction
    last_feats = df.iloc[[-1]]
    rf_next = float(rf.predict(last_feats)[0])
    xgb_next = float(xgb_model.predict(last_feats)[0]) if xgb_model is not None else None

    return {
        "rf_metrics": rf_metrics,
        "xgb_metrics": xgb_metrics,
        "rf_next_return": rf_next,
        "xgb_next_return": xgb_next
    }
