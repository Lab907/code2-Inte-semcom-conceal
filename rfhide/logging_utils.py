"""Centralized logging helpers for scripts and package modules."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "rfhide", level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a consistently formatted logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    return logger

