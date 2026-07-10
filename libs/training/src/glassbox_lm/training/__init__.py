"""Glassbox LM training loop.

Single-GPU / CPU:      python -m glassbox_lm.training
Multi-GPU (DDP):       torchrun --standalone --nproc_per_node=N -m glassbox_lm.training

All hyperparameters come from environment variables via
:class:`glassbox_lm.core.config.Hyperparameters`. Architectures are resolved
through the registry, so any installed architecture package is trainable.
"""
from glassbox_lm.training.train import main

__all__ = ["main"]
