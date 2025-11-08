#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Cats vs Dogs Classification using CNN (Keras)
File: config.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Central configuration module defining paths and hyperparameters for the project.

Usage:
from cats_vs_dogs.config import TrainingConfig
cfg = TrainingConfig()

Notes:
- Adjust default paths and hyperparameters here to customize experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class TrainingConfig:
    """Configuration dataclass for the Cats vs Dogs project."""

    project_name: str = "Cats vs Dogs Classification using CNN (Keras)"

    # Base directory of the repository (two levels up from this file)
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    # Data directories
    data_raw_dir: Path = field(init=False)
    data_processed_dir: Path = field(init=False)

    # Artifacts
    models_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    # Data / training hyperparameters
    image_size: Tuple[int, int] = (180, 180)
    batch_size: int = 32
    seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.15
    epochs: int = 15
    learning_rate: float = 1e-4

    def __post_init__(self) -> None:
        self.data_raw_dir = self.base_dir / "data" / "raw" / "train"
        self.data_processed_dir = self.base_dir / "data" / "processed"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"

        # Ensure artifact directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
