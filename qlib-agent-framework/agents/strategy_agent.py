"""Codex strategy implementation agent for QLib workflows."""

from __future__ import annotations

import asyncio
import copy
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from qlib.backtest import backtest
from qlib.contrib.evaluate import indicator_analysis, risk_analysis

from .base_agent import BaseAgent

_DEFAULT_EXECUTOR = {
    "class": "SimulatorExecutor",
    "module_path": "qlib.backtest.executor",
    "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
}


class StrategyAgent(BaseAgent):
    def __init__(self, agent_config: Dict[str, Any], results_dir: Path):
        super().__init__(agent_config, results_dir)

    async def initialize(self) -> None:
        await super().initialize()
        self.logger.info("Strategy agent configured")

    async def implement_strategies(
        self,
        strategies: Dict[str, Dict[str, Any]],
        prediction_df: pd.DataFrame,
        dataset,
        port_analysis_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        if prediction_df is None or prediction_df.empty:
            raise ValueError("Predictions are required to run strategies.")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._implement_strategies_sync,
            strategies,
            prediction_df,
            dataset,
            port_analysis_config,
        )

    # --- Internal helpers -----------------------------------------------------------------

    def _implement_strategies_sync(
        self,
        strategies: Dict[str, Dict[str, Any]],
        prediction_df: pd.DataFrame,
        dataset,
        port_analysis_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        filtered_signal = self._filter_prediction_to_test(prediction_df, dataset)
        results: Dict[str, Any] = {}
        for strategy_name, strategy_cfg in strategies.items():
            self.logger.info("Running strategy %s", strategy_name)
            results[strategy_name] = self._run_strategy(
                strategy_name,
                strategy_cfg,
                filtered_signal,
                port_analysis_config,
            )
        return results

    def _run_strategy(
        self,
        strategy_name: str,
        strategy_cfg: Dict[str, Any],
        signal_df: pd.DataFrame,
        port_analysis_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        strategy_config = copy.deepcopy(strategy_cfg)
        strategy_kwargs = strategy_config.setdefault("kwargs", {})
        strategy_kwargs["signal"] = signal_df

        executor_cfg = copy.deepcopy(port_analysis_config.get("executor", _DEFAULT_EXECUTOR))
        backtest_cfg = copy.deepcopy(port_analysis_config.get("backtest", {}))
        if backtest_cfg.get("start_time") is None:
            backtest_cfg["start_time"] = signal_df.index.get_level_values("datetime").min()
        if backtest_cfg.get("end_time") is None:
            backtest_cfg["end_time"] = signal_df.index.get_level_values("datetime").max()
        backtest_cfg.setdefault("benchmark", "SH000300")
        backtest_cfg.setdefault("account", 100000000)
        backtest_cfg.setdefault("exchange_kwargs", {})

        try:
            portfolio_metrics, indicator_metrics = backtest(
                start_time=backtest_cfg["start_time"],
                end_time=backtest_cfg["end_time"],
                strategy=strategy_config,
                executor=executor_cfg,
                benchmark=backtest_cfg["benchmark"],
                account=backtest_cfg["account"],
                exchange_kwargs=backtest_cfg["exchange_kwargs"],
            )
        except (TypeError, ModuleNotFoundError, ImportError) as err:
            self.logger.warning("Skipping strategy %s due to configuration error: %s", strategy_name, err)
            return {"status": "skipped", "reason": str(err)}

        freq, (report_df, positions_df) = next(iter(portfolio_metrics.items()))
        indicator_obj = indicator_metrics.get(freq)
        indicator_df = indicator_obj[0] if indicator_obj else None

        report_path = self.results_dir / f"{self.name}_{strategy_name}_report_{freq}.csv"
        positions_path = self.results_dir / f"{self.name}_{strategy_name}_positions_{freq}.pkl"
        report_df.to_csv(report_path)
        with positions_path.open("wb") as fp:
            pickle.dump(positions_df, fp)

        indicator_path = None
        if indicator_df is not None:
            indicator_path = self.results_dir / f"{self.name}_{strategy_name}_indicators_{freq}.csv"
            indicator_df.to_csv(indicator_path)

        risk_without_cost = risk_analysis(report_df["return"] - report_df["bench"], freq=freq)
        risk_with_cost = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"], freq=freq)
        risk_summary = {
            "excess_return_without_cost": self._series_to_dict(risk_without_cost),
            "excess_return_with_cost": self._series_to_dict(risk_with_cost),
        }

        indicator_summary = {}
        if indicator_df is not None:
            indicator_summary = indicator_analysis(indicator_df)
            indicator_summary = indicator_summary["value"].to_dict()

        summary = {
            "frequency": freq,
            "risk": risk_summary,
            "indicators": indicator_summary,
            "artifacts": {
                "report": report_path.name,
                "positions": positions_path.name,
                "indicators": indicator_path.name if indicator_path else None,
            },
        }
        self.logger.info(
            "Strategy %s completed. Annual return (with cost): %s",
            strategy_name,
            risk_summary["excess_return_with_cost"].get("annual_return"),
        )
        return summary

    def _filter_prediction_to_test(self, prediction_df: pd.DataFrame, dataset) -> pd.DataFrame:
        if dataset is None:
            return prediction_df
        try:
            test_index = dataset.prepare("test", col_set="label").index
        except Exception:
            return prediction_df

        common = prediction_df.index.intersection(test_index)
        if common.empty:
            return prediction_df
        return prediction_df.loc[common]

    @staticmethod
    def _series_to_dict(series: pd.Series) -> Dict[str, float]:
        if isinstance(series, pd.Series):
            return {k: float(v) for k, v in series.items()}
        if isinstance(series, pd.DataFrame):
            column = series.columns[0]
            return {k: float(v) for k, v in series[column].items()}
        return {}
