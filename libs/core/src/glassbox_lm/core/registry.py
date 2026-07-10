"""Auto-discovery registry for AgiModel subclasses.

Three sources feed the registry:

1. Entry points — any installed distribution can expose architectures via the
   ``glassbox_lm.architectures`` entry-point group. Each entry point loads to
   an AgiModel subclass, a module, or a package. Packages are walked: every
   ``<subpackage>.model`` module is imported and scanned. The bundled
   ``glassbox-lm-architectures`` distribution registers itself this way.
2. Direct import — as a fallback (e.g. running from a source tree without
   installed dist metadata), ``glassbox_lm.architectures`` is imported and
   walked if present.
3. Explicit registration — the :func:`register` decorator, for architectures
   defined in scripts or notebooks.

Discovery is idempotent: the registry is keyed by ``version``, so a source
seen twice just overwrites itself.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import pkgutil
from types import ModuleType

from glassbox_lm.core.base import AgiModel

ENTRY_POINT_GROUP = "glassbox_lm.architectures"

_REGISTRY: dict[str, type[AgiModel]] | None = None
_EXPLICIT: dict[str, type[AgiModel]] = {}


def _scan_module(mod: ModuleType, registry: dict[str, type[AgiModel]]) -> None:
    """Collect AgiModel subclasses with a version set from a module."""
    for attr in dir(mod):
        cls = getattr(mod, attr)
        if (
            isinstance(cls, type)
            and issubclass(cls, AgiModel)
            and cls is not AgiModel
            and cls.version
        ):
            registry[cls.version] = cls


def _scan_package(pkg: ModuleType, registry: dict[str, type[AgiModel]]) -> None:
    """Scan a package: its own module plus every ``<subpackage>.model``."""
    _scan_module(pkg, registry)
    for info in pkgutil.iter_modules(pkg.__path__):
        try:
            mod = importlib.import_module(f"{pkg.__name__}.{info.name}.model")
        except ModuleNotFoundError:
            # Doc-only variants (e.g. v0_register_lm) ship no model.py.
            continue
        _scan_module(mod, registry)


def _discover() -> dict[str, type[AgiModel]]:
    registry: dict[str, type[AgiModel]] = {}

    for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        obj = ep.load()
        if isinstance(obj, ModuleType):
            if hasattr(obj, "__path__"):
                _scan_package(obj, registry)
            else:
                _scan_module(obj, registry)
        elif isinstance(obj, type) and issubclass(obj, AgiModel) and obj.version:
            registry[obj.version] = obj

    try:
        arch = importlib.import_module("glassbox_lm.architectures")
    except ModuleNotFoundError:
        pass
    else:
        _scan_package(arch, registry)

    registry.update(_EXPLICIT)
    return registry


def register(cls: type[AgiModel]) -> type[AgiModel]:
    """Class decorator: add an externally defined architecture to the registry."""
    if not (isinstance(cls, type) and issubclass(cls, AgiModel)):
        raise TypeError(f"register() expects an AgiModel subclass, got {cls!r}")
    if not cls.version:
        raise ValueError(f"{cls.__name__} must set a `version` to be registered")
    _EXPLICIT[cls.version] = cls
    if _REGISTRY is not None:
        _REGISTRY[cls.version] = cls
    return cls


def get_registry() -> dict[str, type[AgiModel]]:
    """Return the version -> class mapping, discovering on first call."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _discover()
    return _REGISTRY


def build_model(version: str, args) -> AgiModel:
    """Instantiate a model by version string."""
    registry = get_registry()
    if version not in registry:
        raise ValueError(
            f"Unknown model version: {version!r}. "
            f"Available: {sorted(registry.keys())}"
        )
    cls = registry[version]
    return cls(**cls.build_kwargs(args))
