
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import yfinance as yf

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def sp500_constituents() -> pd.DataFrame:
    try:
        tables = pd.read_html(WIKI_SP500)
        # first table is usually constituents
        df = tables[0]
        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def pick_peers_by_industry(sector: str, industry: str, max_n: int = 12) -> List[str]:
    df = sp500_constituents()
    if df.empty:
        return []
    # The wiki table has 'GICS Sector' and 'GICS Sub-Industry'
    # Our normalized columns:
    # 'gics_sector', 'gics_sub-industry', 'symbol'
    gics_sec = 'gics_sector' if 'gics_sector' in df.columns else None
    gics_sub = 'gics_sub-industry' if 'gics_sub-industry' in df.columns else None
    symcol = 'symbol' if 'symbol' in df.columns else None
    if not (gics_sec and gics_sub and symcol):
        return []

    mask = (df[gics_sec].astype(str).str.lower()==str(sector).lower())
    # fall back if industry not cleanly matching
    if isinstance(industry, str) and len(industry):
        mask &= df[gics_sub].astype(str).str.contains(industry.split()[0], case=False, na=False)
    peers = df.loc[mask, symcol].head(max_n).tolist()
    return [s.replace('.','-') for s in peers if isinstance(s, str)]

def basic_multiples_for(ticker: str) -> Dict[str, Any]:
    tk = yf.Ticker(ticker)
    info = {}
    try:
        info = tk.get_info()
    except Exception:
        try:
            info = tk.info
        except Exception:
            info = {}
    price = None
    try:
        h = tk.history(period="1d")["Close"]
        if len(h): price = float(h.iloc[-1])
    except Exception:
        pass

    fin = {}
    try:
        fin["income"] = tk.financials
        fin["balance"] = tk.balance_sheet
        fin["cashflow"] = tk.cashflow
    except Exception:
        fin["income"] = fin["balance"] = fin["cashflow"] = pd.DataFrame()

    # crude multiples
    pe = info.get("trailingPE") or info.get("forwardPE")
    ev_ebitda = info.get("enterpriseToEbitda")
    ps = info.get("priceToSalesTrailing12Months")
    pfcf = None  # not always provided

    return {
        "ticker": ticker, "price": price, "sector": info.get("sector"), "industry": info.get("industry"),
        "pe": pe, "ev_ebitda": ev_ebitda, "ev_sales": ps, "p_fcf": pfcf, "market_cap": info.get("marketCap")
    }
