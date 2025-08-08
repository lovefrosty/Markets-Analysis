
import os, math, requests
import numpy as np
import pandas as pd
import yfinance as yf

FRED_API = os.getenv("FRED_API_KEY", None)
SEC_UA = os.getenv("SEC_USER_AGENT", "you@example.com EquityResearchLab/1.0")

def fred_series(series_id: str, start: str = "2000-01-01") -> pd.Series:
    """Pull a FRED series if API key provided. Returns Series indexed by date (UTC)."""
    if not FRED_API:
        raise RuntimeError("FRED_API_KEY not set")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": FRED_API, "file_type": "json", "observation_start": start}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()["observations"]
    s = pd.Series({pd.to_datetime(d["date"]): float(d["value"]) if d["value"] != "." else np.nan for d in data})
    return s.dropna()

def risk_free_rate() -> float:
    """Return annualized rf as decimal. Prefers FRED DGS10, else Yahoo ^TNX/100."""
    try:
        s = fred_series("DGS10", "2010-01-01")
        rf = s.dropna().iloc[-1] / 100.0
        return float(rf)
    except Exception:
        tnx = yf.Ticker("^TNX").history(period="5d")["Close"]
        if len(tnx):
            return float(tnx.iloc[-1] / 100.0)
        return 0.04  # fallback

def load_prices(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Historical prices with Adj Close."""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No price data for {ticker}")
    return df

def load_index_prices(ticker="SPY", period="5y", interval="1d") -> pd.DataFrame:
    return load_prices(ticker, period, interval)

def get_info(ticker: str) -> dict:
    tk = yf.Ticker(ticker)
    info = {}
    try:
        info = tk.get_info()
    except Exception:
        try:
            info = tk.info
        except Exception:
            info = {}
    return info or {}

def get_statements(ticker: str) -> dict:
    tk = yf.Ticker(ticker)
    out = {}
    try:
        out["income"] = tk.financials
    except Exception:
        out["income"] = pd.DataFrame()
    try:
        out["balance"] = tk.balance_sheet
    except Exception:
        out["balance"] = pd.DataFrame()
    try:
        out["cashflow"] = tk.cashflow
    except Exception:
        out["cashflow"] = pd.DataFrame()
    return out

def get_options_chain(ticker: str) -> dict:
    tk = yf.Ticker(ticker)
    chain = {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expirations": []}
    try:
        exps = tk.options or []
        chain["expirations"] = exps
        if exps:
            opt = tk.option_chain(exps[0])
            chain["calls"] = opt.calls
            chain["puts"]  = opt.puts
    except Exception:
        pass
    return chain

def humanize_dollars(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "N/A"
    absx = abs(x)
    sign = "-" if x < 0 else ""
    if absx >= 1_000_000_000_000:
        return f"{sign}{absx/1_000_000_000_000:.2f}T"
    if absx >= 1_000_000_000:
        return f"{sign}{absx/1_000_000_000:.2f}B"
    if absx >= 1_000_000:
        return f"{sign}{absx/1_000_000:.2f}M"
    if absx >= 1_000:
        return f"{sign}{absx/1_000:.2f}K"
    return f"{sign}{absx:.2f}"
