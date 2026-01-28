"""Evaluation placeholder (Part 8)."""
from __future__ import annotations

import logging

from vc import config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config.ensure_directories()
    logging.info("Evaluation (Part 8) not yet implemented.")


if __name__ == "__main__":
    main()
