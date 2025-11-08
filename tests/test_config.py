#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Cats vs Dogs Classification using CNN (Keras)
File: test_config.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Basic tests for TrainingConfig.

Usage:
pytest tests/test_config.py

Notes:
- Ensures configuration can be instantiated without errors.
"""

from pathlib import Path

from cats_vs_dogs import TrainingConfig


def test_training_config_paths() -> None:
    cfg = TrainingConfig()
    assert isinstance(cfg.base_dir, Path)
    assert isinstance(cfg.data_raw_dir, Path)
    assert isinstance(cfg.data_processed_dir, Path)
    assert isinstance(cfg.models_dir, Path)
    assert isinstance(cfg.logs_dir, Path)
