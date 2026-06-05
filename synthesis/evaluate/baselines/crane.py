"""CRANE baseline: patched ``legacy/CRANE/src/main.py`` adaptive grammar mode."""

from __future__ import annotations

import argparse

ADAPTER_ID = "crane_legacy_crane"


def run(args: argparse.Namespace) -> int:
    from synthesis.evaluate.run_legacy_fixed_strategy import run_crane_legacy_adapter

    if args.strategy != "crane":
        raise ValueError(f"crane adapter requires strategy=crane, got {args.strategy!r}")
    return run_crane_legacy_adapter(args)
