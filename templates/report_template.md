# {{ ticker }} — Automated Equity Research Note
_Date generated: {{ as_of }}_

## Snapshot
- Last close: ${{ snapshot.last_price | round(2) }}
- Market cap: ${{ snapshot.market_cap_human }}
- Sector / Industry: {{ snapshot.sector or 'N/A' }} / {{ snapshot.industry or 'N/A' }}
- Beta (vs SPY): {{ risk.beta | round(2) }}
- Risk-free (10Y): {{ (100*risk.rf) | round(2) }}%
- Expected return (CAPM): {{ (100*risk.expected_return) | round(2) }}%
- 30d realized vol (ann.): {{ (100*risk.realized_vol) | round(2) }}%
- Implied move (ATM straddle, next expiry): ±{{ (100*options.expected_move) | round(2) }}%

## Fundamental Trends
- Revenue (ttm): ${{ fundamentals.revenue_ttm_human }}
- Operating margin (ttm): {{ (100*fundamentals.op_margin_ttm) | round(1) }}%
- FCFF (ttm est.): ${{ fundamentals.fcff_ttm_human }}
- Net debt: ${{ fundamentals.net_debt_human }}
- Interest coverage (EBIT / Interest): {{ fundamentals.interest_coverage if fundamentals.interest_coverage is not none else 'N/A' }}

### Commentary
{{ fundamentals.commentary }}

## Valuation
**Multiples (ttm where available)**
- P/E: {{ valuation.pe or 'N/A' }}
- EV/EBITDA: {{ valuation.ev_ebitda or 'N/A' }}
- EV/Sales: {{ valuation.ev_sales or 'N/A' }}
- P/FCF: {{ valuation.p_fcf or 'N/A' }}

**DCF (base case)**
- WACC: {{ (100*valuation.dcf.wacc) | round(2) }}%
- Terminal growth: {{ (100*valuation.dcf.g_terminal) | round(2) }}%
- Intrinsic value / share: ${{ valuation.dcf.fair_value_per_share | round(2) }}
- Upside vs last close: {{ (100*valuation.dcf.upside_pct) | round(1) }}%

**Reverse DCF**
- Implied revenue CAGR (yrs 1–5): {{ (100*valuation.reverse_dcf.implied_revenue_cagr) | round(2) }}%
- Implied steady-state op margin: {{ (100*valuation.reverse_dcf.implied_op_margin) | round(1) }}%

## Options & Risk
- Expected 1-week move (ATM straddle): ±{{ (100*options.expected_move) | round(2) }}%
- Parametric VaR (95%, 1-day): {{ (100*risk.var_95_pct) | round(2) }}%
- Historical VaR (95%, 1-day): {{ (100*risk.hist_var_95_pct) | round(2) }}%
- Return variance (daily): {{ risk.variance | round(6) }}
- Standard deviation (daily): {{ risk.std_dev | round(4) }}

## Forecasts
- ARIMA next-{{ horizon_days }}d price path: see dashboard
- Linear factor regression (vs SPY): R² = {{ forecasting.regression_r2 | round(3) }}

## Financing & Ratios
- Debt / Equity: {{ ratios.debt_to_equity or 'N/A' }}
- Net debt / EBITDA: {{ ratios.net_debt_to_ebitda or 'N/A' }}
- Current ratio: {{ ratios.current_ratio or 'N/A' }}
- FCF / Debt: {{ ratios.fcf_to_debt or 'N/A' }}

---

_Notes:_ Data via Yahoo Finance/FRED/SEC (where available). Estimates are algorithmic and educational.


## Risk-Neutral Density (Nearest Expiry)
- Time to expiry (yrs): {{ rnd.time_to_expiry | round(4) if rnd else 'N/A' }}
- Mode strike (approx): {{ rnd.mode_strike if rnd else 'N/A' }}
- 10–90% strike interval (prob mass): {{ rnd.p10_strike }} – {{ rnd.p90_strike }}

## Sector Comps (Peer Multiples)
Median among peers (if available):
- P/E: {{ comps.median_pe if comps else 'N/A' }}
- EV/EBITDA: {{ comps.median_ev_ebitda if comps else 'N/A' }}
- EV/Sales: {{ comps.median_ev_sales if comps else 'N/A' }}

## Machine Learning Targets
- RandomForest next-{{ horizon_days }}d return: {{ (100*ml.rf_next_return) | round(2) }}% (R²={{ ml.rf_r2 | round(3) }})
- XGBoost next-{{ horizon_days }}d return: {{ (100*ml.xgb_next_return) | round(2) if ml.xgb_next_return is not none else 'N/A' }}% (R²={{ ml.xgb_r2 | round(3) if ml.xgb_r2 is not none else 'N/A' }})
