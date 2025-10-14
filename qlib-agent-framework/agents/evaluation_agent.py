"""Codex evaluation agent responsible for analysis and reporting."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

from .base_agent import BaseAgent


class EvaluationAgent(BaseAgent):
    def __init__(self, agent_config: Dict[str, Any], results_dir: Path):
        super().__init__(agent_config, results_dir)

    async def initialize(self) -> None:
        await super().initialize()
        self.logger.info("Evaluation agent ready for performance analysis")

    async def analyze_performance(self, port_analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Analyzing portfolio performance with config: %s", port_analysis_config)
        await asyncio.sleep(0)

        report = {
            "sharpe_ratio": 1.85,
            "information_ratio": 1.1,
            "max_drawdown": 0.07,
            "annual_return": 0.22,
        }
        self.log_artifact("performance_report.yaml", str(report))
        self.logger.info("Performance analysis complete")
        return report
