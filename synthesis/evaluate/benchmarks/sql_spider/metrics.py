"""
Evaluation metrics for SQL Spider.

Tracks per-hardness execution accuracy plus the usual CSD diagnostics
(delimiters, syntax validity of constrained segments, tokens, time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class SQLMetrics:
    n: int = 0
    correct: int = 0
    examples_with_delimiters: int = 0
    syntax_valid_segments: int = 0
    total_segments: int = 0
    total_tokens: int = 0
    total_time: float = 0.0

    # Populated once at the end by record_batch_scores()
    per_level_exec: Dict[str, float] = field(default_factory=dict)
    per_level_count: Dict[str, int] = field(default_factory=dict)
    error_types: Dict[str, int] = field(default_factory=dict)

    def accuracy(self) -> float:
        """Exec accuracy across all examples, as percentage."""
        if "all" in self.per_level_exec:
            return 100.0 * float(self.per_level_exec["all"])
        return 100.0 * self.correct / max(1, self.n)

    def syntax_validity(self) -> float:
        return 100.0 * self.syntax_valid_segments / max(1, self.total_segments)

    def all_examples_contain_delimiters(self) -> bool:
        return self.examples_with_delimiters == self.n and self.n > 0

    def avg_tokens(self) -> float:
        return self.total_tokens / max(1, self.n)

    def avg_time(self) -> float:
        return self.total_time / max(1, self.n)

    def update_generation(
        self,
        contains_delimiters: bool,
        token_count: int,
        time_seconds: float,
        constrained_segments: List[Tuple[str, bool]] | None = None,
    ) -> None:
        """Record one example's generation-side counters (delimiters/tokens/time)."""
        self.n += 1
        self.examples_with_delimiters += 1 if contains_delimiters else 0
        self.total_tokens += token_count
        self.total_time += time_seconds
        if constrained_segments:
            for _seg, is_valid in constrained_segments:
                self.total_segments += 1
                if is_valid:
                    self.syntax_valid_segments += 1

    def record_batch_scores(
        self,
        scores: Dict[str, Any],
        error_types: Dict[str, int],
    ) -> None:
        """Record the Spider evaluator's per-hardness exec rates."""
        for level, payload in scores.items():
            if isinstance(payload, dict):
                if "exec" in payload:
                    self.per_level_exec[level] = float(payload["exec"])
                if "count" in payload:
                    self.per_level_count[level] = int(payload["count"])
        self.error_types = dict(error_types)
        # Sync the simple "correct" counter to match the all-level exec accuracy.
        if "all" in self.per_level_exec and "all" in self.per_level_count:
            self.correct = int(round(self.per_level_exec["all"] * self.per_level_count["all"]))

    def summary(self) -> str:
        lines = [
            f"Examples: {self.n}",
            f"Execution Accuracy (all): {self.accuracy():.1f}%",
        ]
        for level in ("easy", "medium", "hard", "extra"):
            if level in self.per_level_exec:
                exec_pct = 100.0 * self.per_level_exec[level]
                count = self.per_level_count.get(level, 0)
                lines.append(f"  {level:<6}: {exec_pct:5.1f}%  (n={count})")
        if self.error_types:
            lines.append(f"Validity breakdown: {self.error_types}")
        lines.append(
            "Contains << >>: "
            f"{'yes' if self.all_examples_contain_delimiters() else 'no'} "
            f"({self.examples_with_delimiters}/{self.n})"
        )
        if self.total_segments > 0:
            lines.append(
                f"Syntax Validity: {self.syntax_validity():.1f}% "
                f"({self.syntax_valid_segments}/{self.total_segments} segments)"
            )
        lines.extend(
            [
                f"Avg Tokens: {self.avg_tokens():.1f}",
                f"Avg Time: {self.avg_time():.2f}s",
            ]
        )
        return "\n".join(lines)
