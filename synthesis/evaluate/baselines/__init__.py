"""Per-strategy baseline adapters backed by patched legacy repos."""

from .registry import ADAPTER_IDS, STRATEGIES, run_baseline_strategy

__all__ = ["ADAPTER_IDS", "STRATEGIES", "run_baseline_strategy"]
