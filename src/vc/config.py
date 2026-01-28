"""Global configuration for the voice conversion assignment.

These values are shared across scripts to keep behavior deterministic and
aligned with the grading rubric.
"""
from pathlib import Path

# Audio processing constants (fixed by assignment spec)
TARGET_SR: int = 16_000
PREEMPH: float = 0.97
MFCC_N: int = 13

# Default speakers (parallel CMU Arctic pair)
DEFAULT_SOURCE_SPK: str = "bdl"
DEFAULT_TARGET_SPK: str = "slt"

# Repository paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
MANIFEST_DIR = ARTIFACT_ROOT / "manifests"
CACHE_DIR = ARTIFACT_ROOT / "cache"
MODELS_DIR = ARTIFACT_ROOT / "models"
OUTPUTS_DIR = ARTIFACT_ROOT / "outputs"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def ensure_directories() -> None:
    """Create required directories if they do not exist."""
    for path in [
        ARTIFACT_ROOT,
        MANIFEST_DIR,
        CACHE_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        DATA_ROOT,
        NOTEBOOK_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


# Ensure directories are present on import for smoother first run.
ensure_directories()
