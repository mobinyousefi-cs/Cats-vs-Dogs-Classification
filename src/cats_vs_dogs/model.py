#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Cats vs Dogs Classification using CNN (Keras)
File: model.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Model definition utilities for the Cats vs Dogs CNN classifier.

Usage:
from cats_vs_dogs.config import TrainingConfig
from cats_vs_dogs.model import build_cnn_model

cfg = TrainingConfig()
model = build_cnn_model(cfg)

Notes:
- Uses a simple CNN with dropout and batch normalization.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from .config import TrainingConfig


def build_cnn_model(config: TrainingConfig) -> tf.keras.Model:
    """
    Build and compile a CNN model for binary classification.

    Parameters
    ----------
    config:
        TrainingConfig instance containing image_size and learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """
    height, width = config.image_size

    model = models.Sequential(
        [
            layers.Input(shape=(height, width, 3)),
            layers.Rescaling(1.0 / 255.0),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    optimizer: Any = optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
