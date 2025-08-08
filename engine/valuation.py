
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any
from .data_sources import humanize_dollars

@dataclass
class DCFInputs:
    wacc: float = 0.09
    g_terminal: float = 0.025
    horizon_years: int = 5
    op_margin_target: float = 0.18
    reinvestment_rate: float = 0.4  # Capex+ΔWC as % of NOPAT growth
    tax_rate: float = 0.21
    shares_out: Optional[float] = None

@dataclass
class DCFResult:
    fair_value_per_share: float
    enterprise_value: float
    equity_value: float
    wacc: float
    g_terminal: float
    upside_pct: float

def compute_basic_multiples(price: float, shares_out: float, ttm_income: pd.DataFrame,
                            ttm_balance: pd.DataFrame, ttm_cashflow: pd.DataFrame) -> Dict[str, Any]:
    """Return common multiples using ttm data where possible."""
    def get_latest(df, key):
        try:
            val = df.loc[key].iloc[0]
            if pd.isna(val): return None
            return float(val)
        except Exception:
            return None

    market_cap = price * shares_out if shares_out else None
    ebit = get_latest(ttm_income, "Ebit") or get_latest(ttm_income, "EBIT") or None
    ebitda = get_latest(ttm_income, "Ebitda") or get_latest(ttm_income, "EBITDA") or None
    net_income = get_latest(ttm_income, "Net Income") or get_latest(ttm_income, "NetIncome") or None
    total_debt = (get_latest(ttm_balance, "Long Term Debt") or 0.0) + (get_latest(ttm_balance, "Short Long Term Debt") or 0.0) + (get_latest(ttm_balance, "Short Term Debt") or 0.0)
    cash = get_latest(ttm_balance, "Cash") or get_latest(ttm_balance, "Cash And Cash Equivalents") or 0.0
    enterprise_value = (market_cap or 0.0) + total_debt - cash

    free_cf = get_latest(ttm_cashflow, "Free Cash Flow") or None
    revenue = get_latest(ttm_income, "Total Revenue") or get_latest(ttm_income, "TotalRevenue") or None

    pe = round((market_cap / net_income), 2) if market_cap and net_income and net_income != 0 else None
    ev_ebitda = round((enterprise_value / ebitda), 2) if enterprise_value and ebitda and ebitda != 0 else None
    ev_sales = round((enterprise_value / revenue), 2) if enterprise_value and revenue and revenue != 0 else None
    p_fcf = round((market_cap / free_cf), 2) if market_cap and free_cf and free_cf != 0 else None

    return {
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "pe": pe, "ev_ebitda": ev_ebitda, "ev_sales": ev_sales, "p_fcf": p_fcf
    }

def estimate_fcff_ttm(income: pd.DataFrame, cashflow: pd.DataFrame, balance: pd.DataFrame) -> float:
    """Rough FCFF estimate using EBIT(1-T) + D&A - Capex - ΔNWC (ttm). Falls back to CFO - Capex."""
    def latest(df, key):
        try:
            return float(df.loc[key].iloc[0])
        except Exception:
            return None

    ebit = latest(income, "Ebit") or latest(income, "EBIT")
    tax_exp = latest(income, "Income Tax Expense") or latest(income, "IncomeTaxExpense")
    pretax = latest(income, "Income Before Tax") or latest(income, "Pretax Income") or None
    tax_rate = (tax_exp / pretax) if tax_exp and pretax and pretax != 0 else 0.21

    da = latest(cashflow, "Depreciation") or latest(cashflow, "Depreciation & Amortization") or latest(cashflow, "Depreciation Amortization Depletion")
    capex = latest(cashflow, "Capital Expenditures") or 0.0

    if ebit is not None and da is not None:
        nopat = ebit * (1 - tax_rate)
        try:
            ca = float(balance.loc["Total Current Assets"].iloc[0])
            cl = float(balance.loc["Total Current Liabilities"].iloc[0])
            ca_prev = float(balance.loc["Total Current Assets"].iloc[1])
            cl_prev = float(balance.loc["Total Current Liabilities"].iloc[1])
            d_nwc = (ca - cl) - (ca_prev - cl_prev)
        except Exception:
            d_nwc = 0.0
        fcff = nopat + da - (capex or 0.0) - d_nwc
        return float(fcff)
    cfo = latest(cashflow, "Total Cash From Operating Activities") or latest(cashflow, "Operating Cash Flow")
    if cfo is not None:
        return float(cfo - (capex or 0.0))
    return np.nan

def dcf(value_drivers: Dict[str, float], inputs: DCFInputs) -> DCFResult:
    """Simple FCFF DCF with fade to terminal margin; project FCFF from revenue and margin assumptions."""
    rev0 = value_drivers.get("revenue_ttm", 0.0)
    op_margin0 = value_drivers.get("op_margin_ttm", 0.15)
    shares_out = inputs.shares_out or value_drivers.get("shares_out", None)
    net_debt = value_drivers.get("net_debt", 0.0)

    g = value_drivers.get("revenue_cagr_5y", 0.06)
    op_margin_target = inputs.op_margin_target
    wacc = inputs.wacc
    g_term = inputs.g_terminal
    tax_rate = inputs.tax_rate
    reinvest_rate = inputs.reinvestment_rate

    years = list(range(1, inputs.horizon_years + 1))
    op_margins = np.linspace(op_margin0, op_margin_target, len(years))
    revenues = [rev0 * (1 + g) ** t for t in years]
    nopat = [rev * mar * (1 - tax_rate) for rev, mar in zip(revenues, op_margins)]
    nopat_with0 = [value_drivers.get("revenue_ttm", rev0) * op_margin0 * (1 - tax_rate)] + nopat
    growth = [max(nopat_with0[i+1] - nopat_with0[i], 0.0) for i in range(len(nopat))]
    reinvest = [reinvest_rate * gth for gth in growth]
    fcff = [n - r for n, r in zip(nopat, reinvest)]

    disc = [(1 + wacc) ** t for t in years]
    pv_fcff = sum(f/d for f, d in zip(fcff, disc))
    tv = fcff[-1] * (1 + g_term) / (wacc - g_term) if wacc > g_term else np.nan
    pv_tv = tv / disc[-1] if not np.isnan(tv) else np.nan
    ev = pv_fcff + (pv_tv if not np.isnan(pv_tv) else 0.0)
    eq = ev - net_debt
    fv_per_share = (eq / shares_out) if shares_out else np.nan
    last_price = value_drivers.get("last_price", np.nan)
    upside = (fv_per_share / last_price - 1.0) if shares_out and last_price and last_price>0 else np.nan

    return DCFResult(fair_value_per_share=fv_per_share, enterprise_value=ev,
                     equity_value=eq, wacc=wacc, g_terminal=g_term, upside_pct=upside)

def reverse_dcf(price_per_share: float, value_drivers: Dict[str, float], inputs: DCFInputs) -> Dict[str, float]:
    """Solve for implied revenue CAGR and steady-state margin that match market price, simple heuristic grid search."""
    shares = inputs.shares_out or value_drivers.get("shares_out", None)
    if not shares or shares<=0:
        return {"implied_revenue_cagr": np.nan, "implied_op_margin": np.nan}
    target_equity = price_per_share * shares
    net_debt = value_drivers.get("net_debt", 0.0)

    grid_g = np.linspace(0.00, 0.20, 41)
    grid_margin = np.linspace(0.05, 0.35, 31)
    best = (1e18, np.nan, np.nan)
    for g in grid_g:
        for m in grid_margin:
            test_inputs = DCFInputs(**{**inputs.__dict__, "op_margin_target": float(m)})
            dcf_res = dcf({**value_drivers, "revenue_cagr_5y": float(g)}, test_inputs)
            eq = dcf_res.equity_value
            err = abs((eq - target_equity))
            if err < best[0]:
                best = (err, g, m)
    return {"implied_revenue_cagr": best[1], "implied_op_margin": best[2]}
