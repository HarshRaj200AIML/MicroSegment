"""Shared project paths, logging, and serialization helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42


def ensure_project_dirs() -> None:
    for relative in ("data/raw", "data/processed", "models", "outputs/figures", "outputs/reports", "outputs/cluster_profiles", "outputs/segmented_data", "outputs/experiments"):
        (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "microsegment") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


class MicroSegmentError(Exception):
    """Base exception for expected MicroSegment failures."""


class DatasetValidationError(MicroSegmentError, ValueError):
    """The selected dataset cannot be loaded or analyzed safely."""


class DatasetLoadError(MicroSegmentError, ValueError):
    """No supported loader could read the source."""


class ModelProvenanceError(MicroSegmentError):
    """A trained model cannot be tied safely to the active dataset."""


class MemorySafetyError(MicroSegmentError):
    """A clustering operation exceeds configured memory safeguards."""


class ExportValidationError(MicroSegmentError):
    """An output would not contain validated model provenance."""