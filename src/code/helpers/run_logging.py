"""Shared run logging helpers for command-line research scripts."""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml

from helpers.paths import OUTPUTS_DIR


@dataclass(frozen=True)
class RunLog:
    """Paths created for one script execution."""

    script_name: str
    run_id: str
    run_dir: Path
    log_path: Path


class StreamToLogger:
    """File-like object that forwards writes to a logger."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line.rstrip())
        return len(message)

    def flush(self) -> None:
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.rstrip())
        self._buffer = ""


def setup_run_logging(
    script_name: str,
    *,
    base_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> RunLog:
    """Create a timestamped run log folder and configure root logging.

    Logs are written to:
        outputs/logs/<script_name>/<YYYYMMDD_HHMMSS>/run.log
    """

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_root = Path(base_dir) if base_dir is not None else OUTPUTS_DIR / "logs"
    run_dir = logs_root / script_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.__stdout__)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(level)
    for handler in handlers:
        root_logger.addHandler(handler)

    logging.captureWarnings(True)
    warnings.simplefilter("default")

    run_log = RunLog(script_name=script_name, run_id=run_id, run_dir=run_dir, log_path=log_path)
    logger = logging.getLogger(script_name)
    logger.info("Run started")
    logger.info("Run directory: %s", run_dir)
    logger.info("Log file: %s", log_path)
    return run_log


def save_run_manifest(
    run_log: RunLog,
    *,
    config_path: Optional[Path] = None,
    config: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Save config and metadata alongside the run log."""

    manifest = {
        "script_name": run_log.script_name,
        "run_id": run_log.run_id,
        "run_dir": str(run_log.run_dir),
        "log_path": str(run_log.log_path),
    }
    if config_path is not None:
        manifest["config_path"] = str(config_path)
    if extra:
        manifest.update(extra)

    with open(run_log.run_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    if config_path is not None and Path(config_path).exists():
        shutil.copy2(config_path, run_log.run_dir / Path(config_path).name)

    if config is not None:
        with open(run_log.run_dir / "config_resolved.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


@contextlib.contextmanager
def redirect_std_streams(logger_name: str) -> Iterator[None]:
    """Capture prints/stdout/stderr into the active run logger."""

    logger = logging.getLogger(logger_name)
    stdout_proxy = StreamToLogger(logger, logging.INFO)
    stderr_proxy = StreamToLogger(logger, logging.ERROR)
    with contextlib.redirect_stdout(stdout_proxy), contextlib.redirect_stderr(stderr_proxy):
        try:
            yield
        finally:
            stdout_proxy.flush()
            stderr_proxy.flush()


def log_section(logger: logging.Logger, title: str) -> None:
    """Emit a visible section divider."""

    logger.info("")
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)
