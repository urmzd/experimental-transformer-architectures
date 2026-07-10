"""Dataset commands.

Artifacts land under ./data in the working directory (override with
GLASSBOX_DATA_DIR), matching the defaults in glassbox_lm.core.config.
"""
from __future__ import annotations

import importlib
import sys

_SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "download": (
        "glassbox_lm.data.download_data",
        "Download FineWeb shards and tokenizer from Hugging Face",
    ),
    "prepare-code": (
        "glassbox_lm.data.prepare_code",
        "Build the code dataset (tokenize and shard; needs sentencepiece)",
    ),
}


def _usage() -> str:
    width = max(map(len, _SUBCOMMANDS))
    lines = ["usage: glassbox data <subcommand> [args]", "", "subcommands:"]
    lines += [f"  {name:<{width}}  {help_text}" for name, (_, help_text) in _SUBCOMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(argv or [])
    if not args or args[0] in ("-h", "--help"):
        print(_usage())
        return 0
    sub, rest = args[0], args[1:]
    if sub not in _SUBCOMMANDS:
        print(f"glassbox data: unknown subcommand {sub!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(_SUBCOMMANDS[sub][0])
    return module.main(rest) or 0
