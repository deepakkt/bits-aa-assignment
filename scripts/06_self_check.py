"""
Placeholder self-check script for the voice conversion assignment.

Later parts will expand this to validate the full pipeline end-to-end. For now
it only verifies that the expected directory layout exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from vc.config import (  # noqa: E402
    ARTIFACTS_ROOT,
    CACHE_DIR,
    DATA_ROOT,
    MANIFEST_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from vc.io_utils import get_logger  # noqa: E402


def main() -> None:
    logger = get_logger("self_check")
    required = [
        DATA_ROOT,
        ARTIFACTS_ROOT,
        MANIFEST_DIR,
        CACHE_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        logger.warning("Missing required paths: %s", ", ".join(str(p) for p in missing))
    else:
        logger.info("All core directories are present.")
    logger.info("Detailed pipeline checks will be added in later parts.")


if __name__ == "__main__":
    main()
