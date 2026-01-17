"""
Utility helpers for logging, filesystem management, and JSON IO.

These helpers stay dependency-light so they can be used from both scripts and
library code without pulling in heavy modules.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict


def get_logger(name: str) -> logging.Logger:
    """
    Configure and return a module-scoped logger with an INFO-level stream
    handler attached only once.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def ensure_dir(path: Path) -> Path:
    """
    Create a directory if it does not exist and return the path.

    This is idempotent and safe to call repeatedly.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save a dictionary to a JSON file with utf-8 encoding and indenting."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON content from disk into a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
