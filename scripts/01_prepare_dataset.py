"""Dataset preparation placeholder (Part 2).

This script will generate deterministic manifests once implemented.
"""
from __future__ import annotations

import logging

from vc import config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config.ensure_directories()
    logging.info(
        "Dataset preparation (Part 2) not yet implemented. See README for upcoming steps."
    )


if __name__ == "__main__":
    main()
