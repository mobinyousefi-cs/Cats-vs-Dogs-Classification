#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Cats vs Dogs Classification using CNN (Keras)
File: evaluate.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Evaluate a trained Cats vs Dogs CNN model on the test dataset.

Usage:
python -m cats_vs_dogs.evaluate --model-path models/cats_vs_dogs_cnn.keras

Notes:
- Assumes data has been prepared under data/processed/test.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import tensorflow as tf

from .config import TrainingConfig
from .data import create_datasets

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Cats vs Dogs CNN classifier on the test set."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to the trained model (.keras). "
        "Defaults to models/cats_vs_dogs_cnn.keras",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()
    config = TrainingConfig()

    model_path = args.model_path or (config.models_dir / "cats_vs_dogs_cnn.keras")
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info("Loading model from: %s", model_path)
    model = tf.keras.models.load_model(model_path)

    _, _, test_ds = create_datasets(
        processed_dir=config.data_processed_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    logger.info("Evaluating on test dataset...")
    results = model.evaluate(test_ds, return_dict=True)
    for name, value in results.items():
        logger.info("Test %s: %.4f", name, value)


if __name__ == "__main__":
    main()
