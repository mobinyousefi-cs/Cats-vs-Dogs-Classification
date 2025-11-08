#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Cats vs Dogs Classification using CNN (Keras)
File: predict.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Inference utilities for predicting whether images contain a cat or a dog.

Usage:
python -m cats_vs_dogs.predict --model-path models/cats_vs_dogs_cnn.keras --image path/to/image.jpg
python -m cats_vs_dogs.predict --model-path models/cats_vs_dogs_cnn.keras --folder path/to/folder

Notes:
- Assumes the model was trained with TrainingConfig.image_size.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import tensorflow as tf

from .config import TrainingConfig

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the Cats vs Dogs CNN classifier."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the trained model (.keras).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        type=Path,
        help="Path to a single image file.",
    )
    group.add_argument(
        "--folder",
        type=Path,
        help="Path to a folder containing image files.",
    )
    return parser.parse_args()


def _iter_images_from_folder(folder: Path) -> Iterable[Path]:
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for path in folder.glob(ext):
            yield path


def _load_and_preprocess_image(
    image_path: Path, config: TrainingConfig
) -> np.ndarray:
    img = tf.keras.utils.load_img(
        image_path,
        target_size=config.image_size,
    )
    arr = tf.keras.utils.img_to_array(img)
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def _predict_one(
    model: tf.keras.Model, image_path: Path, config: TrainingConfig
) -> Tuple[str, float]:
    x = _load_and_preprocess_image(image_path, config)
    prob = float(model.predict(x, verbose=0)[0][0])
    label = "dog" if prob >= 0.5 else "cat"
    confidence = prob if label == "dog" else 1.0 - prob
    return label, confidence


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()
    config = TrainingConfig()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    logger.info("Loading model from: %s", args.model_path)
    model = tf.keras.models.load_model(args.model_path)

    if args.image:
        label, conf = _predict_one(model, args.image, config)
        print(f"Image: {args.image}")
        print(f"Predicted class: {label}")
        print(f"Confidence: {conf:.2f}")
    else:
        folder = args.folder
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        images: List[Path] = list(_iter_images_from_folder(folder))
        if not images:
            logger.warning("No images found in folder: %s", folder)
            return

        for img_path in images:
            label, conf = _predict_one(model, img_path, config)
            print(f"{img_path}\t{label}\t{conf:.2f}")


if __name__ == "__main__":
    main()
