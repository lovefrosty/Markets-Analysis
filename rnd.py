
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from scipy.signal import savgol_filter
from numpy.typing import ArrayLike

@dataclass
class RNDEstimate:
    expiry: pd.Timestamp
    time_to_expiry: float
    strikes: np.ndarray
    pdf: np.ndarray
    strikes_filtered: np.ndarray
    pdf_filtered: np.ndarray

def _mid(bid, ask):
    import numpy as np
    b = np.array(bid, dtype=float); a = np.array(ask, dtype=float)
    m = (b + a) / 2.0
    # if missing, fall back to lastPrice if present
    return m

def breeden_litzenberger_from_chain(calls: pd.DataFrame, last_price: float, rf: float, T_years: float) -> RNDEstimate:
    """
    Approximate risk-neutral density via Breeden-Litzenberger:
      f(K) = exp(rT) * d^2 C(K)/dK^2
    Using mid call prices across strikes for nearest expiry. Finite-difference + Savitzky-Golay smoothing.
    """
    # Ensure required columns
    for col in ["strike", "bid", "ask"]:
        if col not in calls.columns:
            raise ValueError("Calls chain missing required columns")
    df = calls[["strike","bid","ask"]].dropna().copy()
    df["mid"] = (df["bid"].astype(float) + df["ask"].astype(float)) / 2.0
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values("strike")
    strikes = df["strike"].values
    C = df["mid"].values

    # Smooth call prices to reduce microstructure noise
    if len(C) >= 7:
        C_smooth = savgol_filter(C, 7 if len(C)>=7 else len(C)-(len(C)%2==0), 3, mode="interp")
    else:
        C_smooth = C

    # Finite difference second derivative wrt strike
    # Use non-uniform grid aware second derivative via numpy.gradient twice
    dC_dK = np.gradient(C_smooth, strikes, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK, strikes, edge_order=2)

    pdf = np.exp(rf*T_years) * d2C_dK2
    # enforce non-negativity by clipping
    pdf = np.clip(pdf, 0, None)

    # Normalize to integrate to ~1 over covered strike range (approximation)
    # Use trapezoidal integration
    area = np.trapz(pdf, strikes)
    if area > 0:
        pdf_norm = pdf / area
    else:
        pdf_norm = pdf

    # Light smoothing on pdf
    if len(pdf_norm) >= 7:
        pdf_f = savgol_filter(pdf_norm, 7 if len(pdf_norm)>=7 else len(pdf_norm)-(len(pdf_norm)%2==0), 3, mode="interp")
        pdf_f = np.clip(pdf_f, 0, None)
        # re-normalize
        area2 = np.trapz(pdf_f, strikes)
        if area2 > 0:
            pdf_f = pdf_f / area2
    else:
        pdf_f = pdf_norm

    return RNDEstimate(
        expiry=None,
        time_to_expiry=T_years,
        strikes=strikes,
        pdf=pdf_norm,
        strikes_filtered=strikes,
        pdf_filtered=pdf_f
    )
