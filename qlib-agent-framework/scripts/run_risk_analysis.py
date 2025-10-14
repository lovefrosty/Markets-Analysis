#!/usr/bin/env python3
"""
Quick utility to compute risk metrics from a saved backtest report.

Usage:
    python scripts/run_risk_analysis.py results/strategy_agent_topk_dropout_report_1day.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from qlib.contrib.evaluate import risk_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute risk analysis metrics from a report CSV.")
    parser.add_argument("report_csv", type=Path, help="Path to the strategy report CSV.")
    parser.add_argument("--freq", default="1day", help="Frequency string for risk_analysis (default: 1day).")
    args = parser.parse_args()

    report = pd.read_csv(args.report_csv, index_col=0, parse_dates=True)
    metrics = risk_analysis(report["return"] - report["bench"], freq=args.freq)

    print("Risk analysis summary:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
