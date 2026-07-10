"""Bundled model architectures — the reference library.

Each subpackage is one architecture variant (``v9_linattn``,
``v8_lowrank_vv``, ...) containing a ``model.py`` with a single
:class:`~glassbox_lm.core.base.AgiModel` subclass and, usually, a README
explaining the mechanism and what was learned. They are working examples as
much as library code: copy one as the starting point for a new variant.

Discovery is automatic. :mod:`glassbox_lm.core.registry` imports every
``<subpackage>.model`` module found here and registers any ``AgiModel``
subclass with a ``version`` set — no registration list to edit. Third-party
packages can join the same registry via the ``glassbox_lm.architectures``
entry-point group or the :func:`glassbox_lm.core.register` decorator.
"""
