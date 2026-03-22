"""Base class for all AGI model variants."""
from __future__ import annotations

import abc
import inspect
from typing import ClassVar

import torch.nn as nn
from pydantic_settings import BaseSettings
from torch import Tensor


class CommonSettings(BaseSettings):
    """Shared hyperparameters across most model variants."""
    vocab_size: int = 1024
    num_steps: int = 8
    n_fourier_basis: int = 16
    n_channels: int = 128
    activation: str = "gelu"
    logit_softcap: float = 30.0
    decay_init: float = 3.0


class AgiModel(nn.Module, abc.ABC):
    """Abstract base for all model variants.

    Subclasses must define class-level metadata, a Settings inner class,
    and implement forward(). The registry auto-discovers any AgiModel
    subclass that has a `version` set.
    """

    # -- Metadata (override in each subclass) --
    version: ClassVar[str] = ""
    architecture: ClassVar[str] = ""
    cross_position: ClassVar[str] = ""
    within_position: ClassVar[str] = ""

    # -- Config (override with a Settings subclass) --
    Settings: ClassVar[type[CommonSettings]] = CommonSettings

    @classmethod
    def _read_args(cls, args) -> dict:
        """Read Settings fields from an args namespace (unfiltered)."""
        return {k: getattr(args, k) for k in cls.Settings.model_fields if hasattr(args, k)}

    @classmethod
    def _filter_init(cls, kw: dict) -> dict:
        """Keep only keys that __init__ accepts."""
        init_params = set(inspect.signature(cls.__init__).parameters) - {"self"}
        return {k: v for k, v in kw.items() if k in init_params}

    @classmethod
    def build_kwargs(cls, args) -> dict:
        """Extract constructor kwargs from a config/args namespace.

        Override in subclasses to rename fields (e.g. num_steps -> num_cycles).
        Call _read_args() then _filter_init() in custom overrides.
        """
        return cls._filter_init(cls._read_args(args))

    @abc.abstractmethod
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Return scalar loss."""
        ...
