#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Cats vs Dogs Classification using CNN (Keras)
File: test_model.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Sanity checks for the CNN model definition.

Usage:
pytest tests/test_model.py

Notes:
- Checks model output shape and compilation.
"""

import tensorflow as tf

from cats_vs_dogs import TrainingConfig
from cats_vs_dogs.model import build_cnn_model


def test_build_cnn_model_output_shape() -> None:
    cfg = TrainingConfig()
    model = build_cnn_model(cfg)
    # Batch dimension is None; binary output should be shape (None, 1)
    assert model.output_shape[-1] == 1

    # Forward pass on dummy batch
    dummy = tf.zeros((2, cfg.image_size[0], cfg.image_size[1], 3))
    out = model(dummy)
    assert out.shape == (2, 1)
