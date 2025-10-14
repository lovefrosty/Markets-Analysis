"""Common utilities for Codex-powered QLib agents."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
try:
    from datetime import UTC  # Python 3.11+
except ImportError:
    UTC = timezone.utc
from pathlib import Path
from typing import Any, Dict


class BaseAgent:
    """Lightweight base class providing logging and checkpoint helpers."""

    def __init__(self, agent_config: Dict[str, Any], results_dir: Path):
        self.agent_config = agent_config
        self.results_dir = results_dir
        self.name = agent_config.get("name", self.__class__.__name__)
        self.role = agent_config.get("role", "unknown")
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Optional hook for asynchronous initialization logic."""
        self.logger.info("Initializing %s (%s)", self.name, self.role)
        await asyncio.sleep(0)  # Yield control back to the event loop.

    def checkpoint_path(self, filename: str) -> Path:
        return self.results_dir / f"{self.name}_{filename}"

    def log_artifact(self, filename: str, content: str) -> Path:
        path = self.checkpoint_path(filename)
        path.write_text(content, encoding="utf-8")
        self.logger.info("Logged artifact: %s", path)
        return path

    def log_event(self, message: str, **extra: Any) -> None:
        payload = {"timestamp": datetime.now(UTC).isoformat(), **extra}
        self.logger.info("%s | extra=%s", message, payload)
