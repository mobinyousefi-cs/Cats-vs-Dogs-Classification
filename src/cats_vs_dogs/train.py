#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Cats vs Dogs Classification using CNN (Keras)
File: train.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Training script for the Cats vs Dogs CNN classifier.

Usage:
python -m cats_vs_dogs.train --epochs 15

Notes:
- Assumes data has been prepared under data/processed (train/val/test).
- See cats_vs_dogs.data for dataset preparation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import tensorflow as tf

from .config import TrainingConfig
from .data import create_datasets
from .model import build_cnn_model


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Cats vs Dogs CNN classifier."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config if provided).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()
    config = TrainingConfig()
    if args.epochs is not None:
        config.epochs = args.epochs

    logger.info("Using base directory: %s", config.base_dir)
    logger.info("Loading datasets from: %s", config.data_processed_dir)

    train_ds, val_ds, _ = create_datasets(
        processed_dir=config.data_processed_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    model = build_cnn_model(config)
    model.summary(print_fn=logger.info)

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = config.logs_dir / f"fit-{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = config.models_dir / "cats_vs_dogs_cnn.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]

    logger.info("Starting training for %d epochs", config.epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
    )

    logger.info("Training finished.")
    final_val_acc = history.history.get("val_accuracy", [None])[-1]
    if final_val_acc is not None:
        logger.info("Final validation accuracy: %.4f", final_val_acc)

    # Ensure final model is saved (best weights already saved via checkpoint)
    model.save(checkpoint_path)
    logger.info("Model saved to: %s", checkpoint_path)


if __name__ == "__main__":
    main()
