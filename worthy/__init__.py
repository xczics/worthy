# -*- coding: utf-8 -*-
from .config import LLMConfig, EmbeddingConfig, Thresholds, ClassificationCriteria
from .core import Worthy, Verdict

__all__ = [
    "Worthy",
    "Verdict",
    "LLMConfig",
    "EmbeddingConfig",
    "Thresholds",
    "ClassificationCriteria",
]

__version__ = "0.0.1"
