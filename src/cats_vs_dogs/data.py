#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Cats vs Dogs Classification using CNN (Keras)
File: data.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Data preparation utilities and TensorFlow dataset pipelines for the project.

Usage:
python -m cats_vs_dogs.data --raw-dir data/raw/train --output-dir data/processed

Notes:
- Expects Kaggle Dogs vs Cats "train" images in a single folder.
- Splits into train/val/test and creates class subfolders for Keras.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Tuple

import tensorflow as tf

logger = logging.getLogger(__name__)


def prepare_data(
    raw_dir: Path,
    output_dir: Path,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Split raw images into train/val/test directories with class subfolders.

    Parameters
    ----------
    raw_dir:
        Directory containing original Kaggle images (cat.*.jpg, dog.*.jpg).
    output_dir:
        Root directory where train/val/test folders will be created.
    val_split:
        Fraction of data used for validation.
    test_split:
        Fraction of data used for testing.
    seed:
        Random seed for reproducible shuffling.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    logger.info("Collecting image files from %s", raw_dir)
    image_paths = sorted(raw_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No .jpg files found in {raw_dir}")

    # Assign labels based on filename prefix
    samples = []
    for path in image_paths:
        name = path.name.lower()
        if name.startswith("cat"):
            label = "cats"
        elif name.startswith("dog"):
            label = "dogs"
        else:
            logger.warning("Skipping file with unknown label: %s", path)
            continue
        samples.append((path, label))

    if not samples:
        raise RuntimeError("No labeled images (cat*/dog*) found.")

    random.seed(seed)
    random.shuffle(samples)

    n_total = len(samples)
    n_val = int(n_total * val_split)
    n_test = int(n_total * test_split)
    n_train = n_total - n_val - n_test

    train_samples = samples[:n_train]
    val_samples = samples[n_train : n_train + n_val]
    test_samples = samples[n_train + n_val :]

    logger.info(
        "Total: %d | Train: %d | Val: %d | Test: %d",
        n_total,
        len(train_samples),
        len(val_samples),
        len(test_samples),
    )

    def _copy_subset(subset, subset_name: str) -> None:
        subset_dir = output_dir / subset_name
        for _, label in subset:
            (subset_dir / label).mkdir(parents=True, exist_ok=True)

        for src, label in subset:
            dst = subset_dir / label / src.name
            dst.write_bytes(src.read_bytes())

    logger.info("Writing train/val/test splits to %s", output_dir)
    _copy_subset(train_samples, "train")
    _copy_subset(val_samples, "val")
    _copy_subset(test_samples, "test")
    logger.info("Data preparation finished.")


def create_datasets(
    processed_dir: Path,
    image_size: Tuple[int, int],
    batch_size: int,
    seed: int,
):
    """
    Create TensorFlow datasets for train/val/test from processed directory.

    Parameters
    ----------
    processed_dir:
        Directory containing train/val/test subfolders with class subfolders.
    image_size:
        Target (height, width) for image resizing.
    batch_size:
        Batch size for training and evaluation.
    seed:
        Random seed for shuffling.

    Returns
    -------
    (train_ds, val_ds, test_ds)
    """
    processed_dir = Path(processed_dir)
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Processed dataset folders not found under {processed_dir}. "
            "Run data preparation first."
        )

    autotune = tf.data.AUTOTUNE

    def _make_ds(split_dir: Path, shuffle: bool):
        ds = tf.keras.utils.image_dataset_from_directory(
            split_dir,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="binary",
            shuffle=shuffle,
            seed=seed,
        )
        return ds.cache().prefetch(autotune)

    train_ds = _make_ds(train_dir, shuffle=True)
    val_ds = _make_ds(val_dir, shuffle=False)
    test_ds = _make_ds(test_dir, shuffle=False)

    return train_ds, val_ds, test_ds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Cats vs Dogs dataset (train/val/test splits)."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Path to raw Kaggle train images (cat.*.jpg, dog.*.jpg).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for processed train/val/test splits.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split fraction (default: 0.15).",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Test split fraction (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()
    prepare_data(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
