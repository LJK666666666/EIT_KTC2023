"""
EIT Reconstruction Framework
"""

__version__ = "1.0.0"

from .core import (
    BaseReconstructionMethod,
    EITDataset,
    EITDataModule,
    EITEvaluator,
    UnifiedTrainer,
    ConfigManager
)

from .methods import create_method

__all__ = [
    'BaseReconstructionMethod',
    'EITDataset',
    'EITDataModule',
    'EITEvaluator',
    'UnifiedTrainer',
    'ConfigManager',
    'create_method'
]
