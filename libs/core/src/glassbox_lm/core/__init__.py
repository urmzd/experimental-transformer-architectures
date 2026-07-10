"""Glassbox LM core — public API.

``glassbox_lm`` is a namespace package; the importable surface lives in the
member distributions (``glassbox_lm.core``, ``glassbox_lm.architectures``,
``glassbox_lm.data``, ``glassbox_lm.training``, ``glassbox_lm.cli``). This
module re-exports the pieces most callers need:

    from glassbox_lm.core import AgiModel, build_model, get_registry, register
"""
from glassbox_lm.core.base import AgiModel, CommonSettings
from glassbox_lm.core.registry import build_model, get_registry, register

__all__ = [
    "AgiModel",
    "CommonSettings",
    "build_model",
    "get_registry",
    "register",
]
