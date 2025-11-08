#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Cats vs Dogs Classification using CNN (Keras)
File: __init__.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Package initialization for the Cats vs Dogs classification project.

Usage:
from cats_vs_dogs import TrainingConfig

Notes:
- This file exposes key public APIs for convenient imports.
"""

from .config import TrainingConfig

__app_name__ = "cats-vs-dogs-classifier"
__version__ = "0.1.0"

__all__ = [
    "__app_name__",
    "__version__",
    "TrainingConfig",
]
