"""QLib Codex orchestrator coordinating specialized agents."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml

try:
    from .data_agent import DataAgent
    from .evaluation_agent import EvaluationAgent
    from .model_agent import ModelAgent
    from .strategy_agent import StrategyAgent
except ImportError:  # pragma: no cover - allow direct script execution
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(PROJECT_ROOT))
    from agents.data_agent import DataAgent
    from agents.evaluation_agent import EvaluationAgent
    from agents.model_agent import ModelAgent
    from agents.strategy_agent import StrategyAgent

ROLE_CLASS_MAP: Dict[str, Type[Any]] = {
    "data_preparation": DataAgent,
    "model_training": ModelAgent,
    "strategy_implementation": StrategyAgent,
    "results_analysis": EvaluationAgent,
}


class QLibOrchestrator:
    def __init__(self, config_path: str):
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self._resolve_path(config_path)
        self.config = self._load_config(self.config_path)
        self.results_dir = self.project_root / "results"
        self.state_path = self.results_dir / "workflow_state.yaml"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.agents: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {"status": "idle", "last_updated": None}
        self._restore_state()

    async def initialize_agents(self) -> None:
        logging.info("Initializing agents from configuration: %s", self.config_path)
        agents_config = self.config["agent_config"]["agents"]
        for agent_config in agents_config:
            role = agent_config.get("role")
            name = agent_config.get("name")
            agent_cls = ROLE_CLASS_MAP.get(role)
            if agent_cls is None:
                raise ValueError(f"Unsupported agent role: {role}")
            agent_instance = agent_cls(agent_config, self.results_dir)
            await agent_instance.initialize()
            self.agents[name] = agent_instance
        self._update_state(status="agents_initialized", agents=list(self.agents.keys()))

    async def execute_workflow(self) -> Dict[str, Any]:
        if not self.agents:
            await self.initialize_agents()

        data_agent: DataAgent = self.agents["data_agent"]
        model_agent: ModelAgent = self.agents["model_agent"]
        strategy_agent: StrategyAgent = self.agents["strategy_agent"]
        evaluation_agent: EvaluationAgent = self.agents["evaluation_agent"]

        dataset_config = self.config["dataset"]["kwargs"]
        data_config = {
            "qlib_init": self.config.get("qlib_init", {}),
            "handler": dataset_config.get("handler"),
            "dataset": self.config.get("dataset"),
        }
        data_summary = await data_agent.prepare_data(data_config)

        model_tasks = [
            model_agent.train_model(model_name, model_config, data_agent.dataset)
            for model_name, model_config in self.config.get("models", {}).items()
        ]
        model_results = await asyncio.gather(*model_tasks)

        strategies = self.config.get("strategies", {})
        best_model_name = self._select_best_model(model_results)
        prediction_df = model_agent.predictions.get(best_model_name)
        if prediction_df is None:
            raise RuntimeError("No predictions available for strategy evaluation.")
        port_analysis_cfg = self.config.get("port_analysis_config", {})
        strategy_results = await strategy_agent.implement_strategies(
            strategies,
            prediction_df,
            data_agent.dataset,
            port_analysis_cfg,
        )

        evaluation_results = await evaluation_agent.analyze_performance(strategy_results)

        final_results = {
            "data": data_summary,
            "models": model_results,
            "strategies": strategy_results,
            "evaluation": evaluation_results,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self._update_state(status="completed", results=final_results)
        self._save_results(final_results)
        return final_results

    def generate_qlib_config(self, experiment_params: Dict[str, Any]) -> Path:
        config = json.loads(json.dumps(self.config))  # Deep copy preserving anchors
        config.update(experiment_params)
        output_path = self.project_root / "configs" / f"experiment_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.yaml"
        with output_path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(config, fp, default_flow_style=False)
        logging.info("Generated experiment config: %s", output_path)
        return output_path

    def generate_report(self) -> Path:
        report_path = self.results_dir / "workflow_report.txt"
        state_snapshot = yaml.safe_dump(self.state, sort_keys=False)
        report_content = [
            "QLib Codex Workflow Report",
            f"Generated: {datetime.now(UTC).isoformat()}",
            "",
            "Current State:",
            state_snapshot,
        ]
        report_path.write_text("\n".join(report_content), encoding="utf-8")
        logging.info("Report generated at %s", report_path)
        return report_path

    def show_status(self) -> Dict[str, Any]:
        logging.info("Current workflow state: %s", self.state)
        return self.state

    async def handle_error(self, error: Exception) -> None:
        logging.error("Workflow failed: %s", error, exc_info=True)
        self._update_state(status="error", error=str(error))

    # Internal helpers -----------------------------------------------------------------

    @staticmethod
    def _select_best_model(model_results: List[Dict[str, Any]]) -> str:
        best_name: Optional[str] = None
        best_metric = float("-inf")
        for result in model_results:
            metrics = result.get("metrics", {})
            candidate = metrics.get("valid") or metrics.get("test") or metrics.get("train")
            ic_value = candidate.get("ic") if candidate else None
            if ic_value is not None and ic_value > best_metric:
                best_metric = ic_value
                best_name = result.get("model")
        if best_name is None:
            available = [res.get("model") for res in model_results if res.get("metrics")]
            if available:
                best_name = available[0]
        if best_name is None:
            raise RuntimeError("Unable to determine the best model for strategies.")
        return best_name

    def _update_state(self, **updates: Any) -> None:
        self.state.update(updates)
        self.state["last_updated"] = datetime.now(UTC).isoformat()
        self._persist_state()

    def _restore_state(self) -> None:
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as fp:
                saved_state = yaml.safe_load(fp) or {}
            self.state.update(saved_state)

    def _persist_state(self) -> None:
        with self.state_path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(self.state, fp, default_flow_style=False)

    def _save_results(self, results: Dict[str, Any]) -> None:
        output_path = self.results_dir / "workflow_results.yaml"
        with output_path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(results, fp, default_flow_style=False)
        logging.info("Workflow results saved to %s", output_path)

    def _load_config(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp)

    def _resolve_path(self, path_str: str) -> Path:
        candidate = Path(path_str)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate


async def run_cli(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(levelname)s] %(message)s")
    orchestrator = QLibOrchestrator(args.config)

    try:
        if args.status:
            orchestrator.show_status()
            return

        if args.generate_report:
            orchestrator.generate_report()
            return

        if args.initialize:
            await orchestrator.initialize_agents()

        if args.execute_full_pipeline:
            await orchestrator.execute_workflow()

    except Exception as exc:
        await orchestrator.handle_error(exc)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLib Codex Agent Orchestrator")
    parser.add_argument("--config", default="configs/base_config.yaml", help="Path to the orchestration config file.")
    parser.add_argument("--initialize", action="store_true", help="Initialize all agents.")
    parser.add_argument("--execute-full-pipeline", dest="execute_full_pipeline", action="store_true", help="Run the end-to-end workflow.")
    parser.add_argument("--status", action="store_true", help="Show the latest persisted workflow status.")
    parser.add_argument("--generate-report", action="store_true", help="Generate a consolidated workflow report.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (e.g. INFO, DEBUG).")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    asyncio.run(run_cli(args))


if __name__ == "__main__":
    main()
