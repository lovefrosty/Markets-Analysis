"""Codex evaluation agent responsible for analysis and reporting."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from .base_agent import BaseAgent


class EvaluationAgent(BaseAgent):
    def __init__(self, agent_config: Dict[str, Any], results_dir: Path):
        super().__init__(agent_config, results_dir)

    async def initialize(self) -> None:
        await super().initialize()
        self.logger.info("Evaluation agent ready for performance analysis")

    async def analyze_performance(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        summary = await loop.run_in_executor(None, self._analyze_sync, strategy_results)
        yaml_summary = yaml.safe_dump(summary, sort_keys=False)
        self.log_artifact("evaluation_summary.yaml", yaml_summary)
        self.logger.info("Performance analysis complete")
        return summary

    # --- Internal helpers -----------------------------------------------------------------

    def _analyze_sync(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        evaluation: Dict[str, Any] = {"strategies": {}, "best_strategy": None}
        best_name, best_metric = None, float("-inf")
        for strategy_name, result in strategy_results.items():
            risk_with_cost = result.get("risk", {}).get("excess_return_with_cost", {})
            annual_return = risk_with_cost.get("annualized_return")
            sharpe = risk_with_cost.get("sharpe") or risk_with_cost.get("information_ratio")
            evaluation["strategies"][strategy_name] = {
                "annual_return": annual_return,
                "sharpe": sharpe,
                "max_drawdown": risk_with_cost.get("max_drawdown"),
                "artifacts": result.get("artifacts", {}),
            }
            if sharpe is not None and sharpe > best_metric:
                best_metric = sharpe
                best_name = strategy_name

        evaluation["best_strategy"] = {"name": best_name, "sharpe": best_metric}
        return evaluation
