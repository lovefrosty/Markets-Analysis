
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pmdarima import auto_arima

@dataclass
class ForecastOutputs:
    regression_r2: float
    arima_order: Optional[Tuple[int, int, int]]
    arima_forecast: pd.Series

def factor_regression(prices: pd.Series, spy_prices: pd.Series, rf_daily: float) -> Dict[str, Any]:
    """Regress asset daily returns vs SPY daily returns; returns beta/alpha/R2."""
    ret = prices.pct_change().dropna()
    mkt = spy_prices.pct_change().reindex_like(ret).dropna()
    df = pd.DataFrame({"ret": ret, "mkt": mkt})
    rf_d = rf_daily / 252.0
    df["excess"] = df["ret"] - rf_d
    X = df[["mkt"]]
    y = df["excess"]
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    r2 = r2_score(y, pred)
    beta = float(model.coef_[0])
    alpha = float(model.intercept_)
    return {"beta": beta, "alpha": alpha, "r2": r2}

def arima_forecast(prices: pd.Series, horizon_days: int = 30) -> ForecastOutputs:
    """Auto ARIMA on daily close; returns forecast Series."""
    y = prices.asfreq("B").ffill()
    model = auto_arima(y, seasonal=False, suppress_warnings=True, stepwise=True, error_action="ignore")
    order = model.order if hasattr(model, "order") else None
    future = model.predict(n_periods=horizon_days)
    idx = pd.bdate_range(start=y.index[-1] + pd.tseries.offsets.BDay(1), periods=horizon_days)
    fc = pd.Series(future, index=idx)
    return ForecastOutputs(regression_r2=np.nan, arima_order=order, arima_forecast=fc)
