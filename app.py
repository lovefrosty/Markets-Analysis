
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from engine.data_sources import load_prices, load_index_prices, get_info, get_statements, get_options_chain, risk_free_rate, humanize_dollars
from engine.valuation import compute_basic_multiples, estimate_fcff_ttm, dcf, DCFInputs, reverse_dcf
from engine.forecasting import factor_regression, arima_forecast
from engine.report import render_report
from engine.rnd import breeden_litzenberger_from_chain
from engine.comps import pick_peers_by_industry, basic_multiples_for
from engine.ml import train_models
from engine.pdf import render_pdf

st.set_page_config(page_title="Equity Research Lab", layout="wide")
st.title("📊 Equity Research Lab")
st.write("Enter a U.S. equity ticker to generate a dashboard + research note.")

with st.sidebar:
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    period = st.selectbox("History Period", options=["1y","3y","5y","max"], index=2)
    interval = st.selectbox("Interval", options=["1d","1wk"], index=0)
    st.markdown("---")
    st.markdown("### Valuation Assumptions")
    wacc = st.number_input("WACC", min_value=0.0, max_value=0.5, value=0.09, step=0.005, format="%.3f")
    gterm = st.number_input("Terminal Growth", min_value=0.0, max_value=0.05, value=0.025, step=0.001, format="%.3f")
    op_margin_target = st.number_input("Target Operating Margin", min_value=0.0, max_value=0.5, value=0.18, step=0.01, format="%.2f")
    reinvest_rate = st.number_input("Reinvestment Rate (of NOPAT growth)", min_value=0.0, max_value=1.0, value=0.40, step=0.05, format="%.2f")
    horizon = st.slider("DCF Horizon (years)", 3, 10, 5)
    st.markdown("---")
    mc_paths = st.number_input("Monte Carlo Paths", min_value=100, max_value=50000, value=5000, step=100)
    horizon_days = st.slider("MC / Forecast Horizon (trading days)", 10, 252, 60)
    st.markdown("---")
    st.markdown("### Comps Settings")
    auto_comps = st.checkbox("Auto-pick S&P 500 peers by industry", value=True)
    manual_peers = st.text_input("Manual peer tickers (comma-separated)", value="")
    st.markdown("---")
    st.markdown("### ML Targets")
    ml_horizon = st.slider("ML prediction horizon (trading days)", 5, 60, 20)

if not ticker:
    st.stop()

# Load data
try:
    px = load_prices(ticker, period=period, interval=interval)
    last_price = float(px["Adj Close"].dropna().iloc[-1])
    info = get_info(ticker)
    stmts = get_statements(ticker)
    spy = load_index_prices("SPY", period=period, interval=interval)
    rf = risk_free_rate()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Shares outstanding heuristic
shares_out = None
if "sharesOutstanding" in info:
    shares_out = float(info.get("sharesOutstanding"))
elif "marketCap" in info and last_price > 0:
    shares_out = float(info["marketCap"])/last_price

# Snapshot
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last Price", f"${last_price:,.2f}")
c2.metric("Market Cap", humanize_dollars(info.get("marketCap")) if info.get("marketCap") else "N/A")
c3.metric("Sector", info.get("sector","N/A"))
c4.metric("Industry", info.get("industry","N/A"))
st.markdown("---")

# Risk / Beta
r = px["Adj Close"].pct_change().dropna()
r_spy = spy["Adj Close"].pct_change().reindex_like(r).dropna()
if len(r) and len(r_spy):
    beta = float(np.cov(r, r_spy)[0,1] / np.var(r_spy))
    realized_vol = float(r.std() * np.sqrt(252))
    expected_return = rf + beta * 0.055
else:
    beta, realized_vol, expected_return = np.nan, np.nan, np.nan

# Options expected move (ATM straddle)
chain = get_options_chain(ticker)
exp_move = np.nan
if chain["expirations"] and not chain["calls"].empty and not chain["puts"].empty:
    calls = chain["calls"].copy(); puts = chain["puts"].copy()
    calls["dist"] = (calls["strike"] - last_price).abs()
    k = float(calls.sort_values("dist").iloc[0]["strike"])
    put_row = puts.iloc[(puts["strike"]-k).abs().argsort()].iloc[0]
    call_row = calls.sort_values("dist").iloc[0]
    call_mid = (float(call_row.get("bid", np.nan)) + float(call_row.get("ask", np.nan))) / 2.0
    put_mid  = (float(put_row.get("bid", np.nan)) + float(put_row.get("ask", np.nan))) / 2.0
    if not np.isnan(call_mid) and not np.isnan(put_mid):
        exp_move = (call_mid + put_mid) / last_price

# Monte Carlo
mu = expected_return if not np.isnan(expected_return) else (r.mean() * 252.0)
sigma = realized_vol if not np.isnan(realized_vol) else (r.std() * np.sqrt(252))
dt_step = 1/252.0
paths = int(mc_paths)
steps = int(horizon_days)
if sigma and sigma>0 and steps>0 and paths>0:
    Z = np.random.standard_normal((steps, paths))
    increments = (mu - 0.5*sigma**2) * dt_step + sigma * np.sqrt(dt_step) * Z
    log_paths = np.vstack([np.zeros((1, paths)), increments]).cumsum(axis=0)
    S = last_price * np.exp(log_paths)
    mc_prices = pd.DataFrame(S, index=pd.RangeIndex(0, steps+1, name="t"))
    var_95_pct = 1.65 * sigma/np.sqrt(252)
    hist_var_95 = -np.percentile(r, 5) if len(r) else np.nan
else:
    mc_prices = pd.DataFrame(); var_95_pct = np.nan; hist_var_95 = np.nan

# Multiples & FCFF
multiples = compute_basic_multiples(last_price, shares_out or np.nan, stmts["income"], stmts["balance"], stmts["cashflow"])
fcff_ttm = estimate_fcff_ttm(stmts["income"], stmts["cashflow"], stmts["balance"])

# DCF & Reverse DCF
rev_ttm = None; op_margin_ttm = None
try:
    rev_ttm = float(stmts["income"].loc["Total Revenue"].iloc[0])
except Exception: pass
try:
    op_income = float(stmts["income"].loc["Operating Income"].iloc[0])
    op_margin_ttm = (op_income / rev_ttm) if rev_ttm and rev_ttm>0 else None
except Exception: pass

try:
    rev_hist = stmts["income"].loc["Total Revenue"].dropna()
    if len(rev_hist) >= 2:
        r0 = float(rev_hist.iloc[-1]); r5 = float(rev_hist.iloc[0]); n = len(rev_hist)-1
        rev_cagr_5y = (r0/r5)**(1/n) - 1.0 if r5>0 and n>0 else 0.06
    else:
        rev_cagr_5y = 0.06
except Exception:
    rev_cagr_5y = 0.06

net_debt = 0.0
try:
    lt_debt = float(stmts["balance"].loc.get("Long Term Debt", [0])[0])
    st_debt = float(stmts["balance"].loc.get("Short Long Term Debt", [0])[0])
    cash = float(stmts["balance"].loc.get("Cash", [0])[0])
    net_debt = lt_debt + st_debt - cash
except Exception: pass

drivers = dict(
    revenue_ttm = rev_ttm or 0.0,
    op_margin_ttm = op_margin_ttm or 0.15,
    revenue_cagr_5y = rev_cagr_5y,
    shares_out = shares_out or np.nan,
    net_debt = net_debt,
    last_price = last_price
)
dcf_inputs = DCFInputs(wacc=wacc, g_terminal=gterm, horizon_years=horizon, op_margin_target=op_margin_target, reinvestment_rate=reinvest_rate, shares_out=shares_out)
dcf_res = dcf(drivers, dcf_inputs)
rdcf = reverse_dcf(last_price, drivers, dcf_inputs)

# Ratios
def safe_div(a, b): return (a / b) if (a is not None and b and b!=0) else None
debt = (stmts["balance"].loc["Long Term Debt"].iloc[0] if "Long Term Debt" in stmts["balance"].index else 0.0) +        (stmts["balance"].loc["Short Long Term Debt"].iloc[0] if "Short Long Term Debt" in stmts["balance"].index else 0.0) +        (stmts["balance"].loc["Short Term Debt"].iloc[0] if "Short Term Debt" in stmts["balance"].index else 0.0)
equity = stmts["balance"].loc["Total Stockholder Equity"].iloc[0] if "Total Stockholder Equity" in stmts["balance"].index else None
cash = stmts["balance"].loc["Cash"].iloc[0] if "Cash" in stmts["balance"].index else 0.0
ebit = stmts["income"].loc["Ebit"].iloc[0] if "Ebit" in stmts["income"].index else (stmts["income"].loc["EBIT"].iloc[0] if "EBIT" in stmts["income"].index else None)
interest = stmts["income"].loc["Interest Expense"].iloc[0] if "Interest Expense" in stmts["income"].index else None
current_ratio = safe_div(stmts["balance"].loc["Total Current Assets"].iloc[0] if "Total Current Assets" in stmts["balance"].index else None,
                         stmts["balance"].loc["Total Current Liabilities"].iloc[0] if "Total Current Liabilities" in stmts["balance"].index else None)
interest_cov = safe_div(ebit, (abs(interest) if interest is not None else None))
net_debt_to_ebitda = safe_div((debt - cash), (stmts["income"].loc["Ebitda"].iloc[0] if "Ebitda" in stmts["income"].index else None))
ratios = dict(
    debt_to_equity = round(safe_div(debt, equity), 2) if safe_div(debt, equity) is not None else None,
    net_debt_to_ebitda = round(net_debt_to_ebitda, 2) if net_debt_to_ebitda is not None else None,
    current_ratio = round(current_ratio, 2) if current_ratio is not None else None,
    fcf_to_debt = round((fcff_ttm / debt), 2) if debt and debt!=0 and fcff_ttm and not np.isnan(fcff_ttm) else None
)

# TABS
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Valuation", "Options & Risk", "Financials", "Forecasting", "Derivatives & Peers"])

with tab1:
    st.subheader("Price History")
    fig = plt.figure()
    plt.plot(px.index, px["Adj Close"])
    plt.title(f"{ticker} Adj Close"); plt.xlabel("Date"); plt.ylabel("Price")
    st.pyplot(fig)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Beta vs SPY", f"{beta:.2f}" if not np.isnan(beta) else "N/A")
    c2.metric("Realized Vol (ann.)", f"{realized_vol*100:.2f}%" if not np.isnan(realized_vol) else "N/A")
    c3.metric("Risk-free (10Y)", f"{rf*100:.2f}%")
    c4.metric("Expected Return (CAPM)", f"{(expected_return*100):.2f}%" if not np.isnan(expected_return) else "N/A")

with tab2:
    st.subheader("Multiples")
    st.write({
        "P/E": multiples["pe"],
        "EV/EBITDA": multiples["ev_ebitda"],
        "EV/Sales": multiples["ev_sales"],
        "P/FCF": multiples["p_fcf"],
        "EV": humanize_dollars(multiples["enterprise_value"] if multiples["enterprise_value"] else None)
    })
    st.subheader("DCF (Base Case)")
    st.write({
        "WACC": round(dcf_res.wacc*100,2),
        "Terminal Growth": round(dcf_res.g_terminal*100,2),
        "EV ($B)": round(dcf_res.enterprise_value/1e9,2) if dcf_res.enterprise_value==dcf_res.enterprise_value else None,
        "Equity Value ($B)": round(dcf_res.equity_value/1e9,2) if dcf_res.equity_value==dcf_res.equity_value else None,
        "Fair Value / Share": round(dcf_res.fair_value_per_share,2) if dcf_res.fair_value_per_share==dcf_res.fair_value_per_share else None,
        "Upside %": round(dcf_res.upside_pct*100,1) if dcf_res.upside_pct==dcf_res.upside_pct else None
    })
    st.subheader("Reverse DCF")
    st.write({
        "Implied Rev CAGR (1–5y)": round(rdcf["implied_revenue_cagr"]*100,2) if rdcf["implied_revenue_cagr"]==rdcf["implied_revenue_cagr"] else None,
        "Implied Steady-State Op Margin": round(rdcf["implied_op_margin"]*100,1) if rdcf["implied_op_margin"]==rdcf["implied_op_margin"] else None
    })

with tab3:
    st.subheader("Options-backed Expectations")
    st.write({"Nearest-expiry ATM straddle move": (f"±{exp_move*100:.2f}%" if exp_move==exp_move else "N/A")})
    if not mc_prices.empty:
        st.subheader("Monte Carlo Price Paths")
        fig2 = plt.figure()
        sample = mc_prices.iloc[:, :min(100, mc_prices.shape[1])]
        plt.plot(sample.index, sample)
        plt.title(f"{ticker} Monte Carlo ({paths} paths, {steps} days)"); plt.xlabel("Day"); plt.ylabel("Price")
        st.pyplot(fig2)
    st.subheader("Risk Metrics")
    variance = float(r.var()) if len(r) else float("nan")
    std_dev = float(r.std()) if len(r) else float("nan")
    st.write({
        "Daily Variance": round(variance, 6) if variance==variance else None,
        "Daily Std Dev": round(std_dev, 4) if std_dev==std_dev else None,
        "Parametric VaR 95% (1d)": f"{var_95_pct*100:.2f}% " if var_95_pct==var_95_pct else "N/A",
        "Historical VaR 95% (1d)": f"{hist_var_95*100:.2f}%" if hist_var_95==hist_var_95 else "N/A"
    })

with tab4:
    st.subheader("Top line & margins")
    st.write({
        "Revenue (ttm)": humanize_dollars(rev_ttm) if rev_ttm else "N/A",
        "Operating Margin (ttm)": f"{op_margin_ttm*100:.1f}%" if op_margin_ttm else "N/A",
        "FCFF (ttm est.)": humanize_dollars(fcff_ttm) if fcff_ttm==fcff_ttm else "N/A"
    })
    st.subheader("Financing & Liquidity")
    st.write(ratios)

with tab5:
    st.subheader("ARIMA Forecast")
    try:
        fc = arima_forecast(px["Adj Close"], horizon_days)
        fig3 = plt.figure()
        plt.plot(px.index, px["Adj Close"])
        plt.plot(fc.arima_forecast.index, fc.arima_forecast.values)
        plt.title(f"{ticker} ARIMA Forecast"); plt.xlabel("Date"); plt.ylabel("Price")
        st.pyplot(fig3)
    except Exception as e:
        st.info(f"ARIMA failed: {e}")
    st.subheader("Factor Regression (vs SPY)")
    try:
        fac = factor_regression(px["Adj Close"], spy["Adj Close"], rf)
        st.write({k: (round(v,4) if isinstance(v,(int,float)) else v) for k,v in fac.items()})
    except Exception as e:
        st.info(f"Regression failed: {e}")

st.markdown("---")
st.header("📝 Generate Research Note")
# include extended sections (RND, comps, ML) if available above
if st.button("Build Markdown Report"):
    context = {
        "ticker": ticker,
        "as_of": pd.Timestamp.now(tz='US/Eastern').strftime("%Y-%m-%d %H:%M %Z"),
        "snapshot": {
            "last_price": last_price,
            "market_cap_human": humanize_dollars(info.get('marketCap')) if info.get('marketCap') else "N/A",
            "sector": info.get("sector"), "industry": info.get("industry")
        },
        "risk": {
            "beta": beta, "rf": rf, "expected_return": expected_return,
            "realized_vol": realized_vol,
            "var_95_pct": var_95_pct, "hist_var_95_pct": hist_var_95,
            "variance": float(r.var()) if len(r) else None,
            "std_dev": float(r.std()) if len(r) else None
        },
        "options": { "expected_move": float(exp_move) if exp_move==exp_move else 0.0 },
        "fundamentals": {
            "revenue_ttm_human": humanize_dollars(rev_ttm) if rev_ttm else "N/A",
            "op_margin_ttm": op_margin_ttm or 0.0,
            "fcff_ttm_human": humanize_dollars(fcff_ttm) if fcff_ttm==fcff_ttm else "N/A",
            "net_debt_human": humanize_dollars(net_debt),
            "interest_coverage": interest_cov,
            "commentary": "Algorithmic summary based on ttm trends and default assumptions."
        },
        "valuation": {
            "pe": multiples["pe"], "ev_ebitda": multiples["ev_ebitda"], "ev_sales": multiples["ev_sales"], "p_fcf": multiples["p_fcf"],
            "dcf": {
                "wacc": dcf_res.wacc, "g_terminal": dcf_res.g_terminal,
                "fair_value_per_share": dcf_res.fair_value_per_share,
                "upside_pct": dcf_res.upside_pct
            },
            "reverse_dcf": rdcf
        },
        "forecasting": { "regression_r2": fac["r2"] if 'fac' in locals() else np.nan },
            "rnd": rnd_summary if 'rnd_summary' in locals() else {},
            "comps_summary": comps_summary if 'comps_summary' in locals() else {},
            "ml": ml_out if 'ml_out' in locals() else {},
        "ratios": ratios,
        "horizon_days": horizon_days
    }
    md = render_report("templates", context)
    st.download_button("⬇️ Download report.md", data=md, file_name=f"{ticker}_research_note.md", mime="text/markdown")

with tab6:
    st.subheader("Risk-Neutral Density (Breeden–Litzenberger)")
    rnd_summary = {}
    try:
        if chain["expirations"] and not chain["calls"].empty:
            # time to expiry
            import pandas as pd, numpy as np, datetime as dt, math
            exp0 = pd.to_datetime(chain["expirations"][0])
            T = max((exp0 - pd.Timestamp.utcnow()).days/365.25, 1/365.25)
            rnd_est = breeden_litzenberger_from_chain(chain["calls"], last_price, rf, T)
            # display plot
            import matplotlib.pyplot as plt
            fig4 = plt.figure()
            plt.plot(rnd_est.strikes_filtered, rnd_est.pdf_filtered)
            plt.title("Risk-neutral density vs strike (nearest expiry)")
            plt.xlabel("Strike"); plt.ylabel("Density")
            st.pyplot(fig4)
            # summary stats: p10, p90, mode approx
            cdf = np.cumsum(rnd_est.pdf_filtered)
            cdf = cdf/ cdf[-1] if cdf[-1]>0 else cdf
            p10 = np.interp(0.10, cdf, rnd_est.strikes_filtered)
            p90 = np.interp(0.90, cdf, rnd_est.strikes_filtered)
            mode_k = float(rnd_est.strikes_filtered[np.argmax(rnd_est.pdf_filtered)])
            rnd_summary = {"time_to_expiry_years": round(T,4), "mode_strike": round(mode_k,2), "p10_strike": round(float(p10),2), "p90_strike": round(float(p90),2)}
            st.write(rnd_summary)
        else:
            st.info("Options chain unavailable for RND.")
    except Exception as e:
        st.info(f"RND failed: {e}")

    st.subheader("Sector Comps & Peer Multiples")
    comps_table = None; comps_summary = {}
    try:
        peers = []
        if auto_comps:
            peers = pick_peers_by_industry(info.get("sector",""), info.get("industry",""), max_n=12)
            peers = [p for p in peers if p != ticker]
        if manual_peers.strip():
            peers += [t.strip().upper() for t in manual_peers.split(",") if t.strip()]
        peers = sorted(set(peers))
        if peers:
            rows = []
            for p in peers:
                rows.append(basic_multiples_for(p))
            import pandas as pd, numpy as np
            comps_table = pd.DataFrame(rows)
            st.dataframe(comps_table)
            # summary
            def med(col): 
                try: 
                    return round(float(np.nanmedian(pd.to_numeric(comps_table[col], errors="coerce"))),2)
                except Exception: 
                    return None
            comps_summary = {
                "num_peers": len(peers),
                "median_pe": med("pe"),
                "median_ev_ebitda": med("ev_ebitda"),
                "median_ev_sales": med("ev_sales")
            }
            st.write(comps_summary)
        else:
            st.info("No peers found. Try adding manual tickers.")
    except Exception as e:
        st.info(f"Comps failed: {e}")

    st.subheader("Machine Learning Price Targets")
    ml_out = {}
    try:
        ml_res = train_models(px["Adj Close"], horizon=ml_horizon)
        rf_next = ml_res["rf_next_return"]
        xgb_next = ml_res["xgb_next_return"]
        rf_target = float(last_price * (1.0 + rf_next))
        xgb_target = (float(last_price * (1.0 + xgb_next)) if xgb_next is not None else None)
        ml_out = {
            "rf_r2": round(ml_res["rf_metrics"]["r2"],3),
            "rf_mae": round(ml_res["rf_metrics"]["mae"],4),
            "rf_rmse": round(ml_res["rf_metrics"]["rmse"],4),
            "rf_next_return": rf_next,
            "rf_price_target": rf_target,
            "xgb_r2": (round(ml_res["xgb_metrics"]["r2"],3) if ml_res["xgb_metrics"] else None),
            "xgb_mae": (round(ml_res["xgb_metrics"]["mae"],4) if ml_res["xgb_metrics"] else None),
            "xgb_rmse": (round(ml_res["xgb_metrics"]["rmse"],4) if ml_res["xgb_metrics"] else None),
            "xgb_next_return": xgb_next,
            "xgb_price_target": xgb_target
        }
        st.write(ml_out)
    except Exception as e:
        st.info(f"ML targets failed: {e}")


    # PDF export
    st.markdown("#### Or download as PDF")
    try:
        pdf_bytes = render_pdf(context)
        st.download_button("⬇️ Download report.pdf", data=pdf_bytes, file_name=f"{ticker}_research_note.pdf", mime="application/pdf")
    except Exception as e:
        st.info(f"PDF export failed: {e}")
