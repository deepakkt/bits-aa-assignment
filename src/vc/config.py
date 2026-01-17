"""
Global configuration for the voice conversion assignment.

These constants are intentionally simple so they can be imported across scripts
without side effects. Seeds are exposed via a helper to keep determinism
explicit in entrypoints.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# Sampling and feature constants
TARGET_SR: int = 16_000
PREEMPH: float = 0.97
MFCC_N: int = 13

# Dataset split expectations
TRAIN_UTT: int = 40
TEST_UTT: int = 10

# Speaker defaults for CMU Arctic parallel data
SOURCE_SPK: str = "bdl"
TARGET_SPK: str = "slt"

# Reproducibility
DEFAULT_SEED: int = 42

# Paths
DATA_ROOT: Path = Path("data")
ARTIFACTS_ROOT: Path = Path("artifacts")
MANIFEST_DIR: Path = ARTIFACTS_ROOT / "manifests"
CACHE_DIR: Path = ARTIFACTS_ROOT / "cache"
MODELS_DIR: Path = ARTIFACTS_ROOT / "models"
OUTPUTS_DIR: Path = ARTIFACTS_ROOT / "outputs"


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    """
    Set random seeds for reproducibility across numpy and Python's RNG.

    Args:
        seed: Seed to apply.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
