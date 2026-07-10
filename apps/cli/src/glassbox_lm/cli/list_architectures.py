"""List every architecture visible to the registry.

Covers the bundled zoo (glassbox-lm-architectures), any installed
distribution exposing the ``glassbox_lm.architectures`` entry-point group,
and anything added with ``glassbox_lm.core.register``.
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glassbox list", description=__doc__)
    parser.add_argument(
        "--long", action="store_true",
        help="also show the implementing class for each version",
    )
    args = parser.parse_args(argv)

    from glassbox_lm.core.registry import get_registry

    registry = get_registry()
    if not registry:
        print(
            "No architectures discovered. Install glassbox-lm-architectures or "
            "register one via the glassbox_lm.architectures entry-point group."
        )
        return 1

    headers = ["version", "architecture", "cross-position", "within-position"]
    if args.long:
        headers.append("class")
    rows = []
    for version in sorted(registry):
        cls = registry[version]
        row = [
            version,
            cls.architecture or "-",
            cls.cross_position or "-",
            cls.within_position or "-",
        ]
        if args.long:
            row.append(f"{cls.__module__}.{cls.__name__}")
        rows.append(row)

    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    print(f"\n{len(rows)} architectures")
    return 0
