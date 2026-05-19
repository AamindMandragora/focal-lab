"""Prompt tier selection and rendering for MetaDecode / baseline evaluation.

Tier 1 (answer-only): GCD, IterGen, CARS — grammar-masked from the first token.
Tier 2 (chain-of-thought): Unconstrained, CRANE, MetaDecode — free-form reasoning allowed.

Templates live under ``synthesis/evaluate/prompts/<benchmark>/tier{1,2}.txt`` with frozen
few-shot rows in ``shots.json`` beside them. GSM shots mirror CRANE ``gsm_symbolic.yaml``;
Spider shots mirror CRANE ``spider.yaml``; SMILES shots mirror CARS class ``data/*.txt``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

PromptTier = Literal[1, 2]

PROMPTS_ROOT = Path(__file__).resolve().parent / "prompts"

# Paper-facing decode caps (applied via ``effective_max_new_tokens`` unless CLI is lower).
BENCHMARK_MAX_NEW_TOKENS: dict[str, int] = {
    "gsm_symbolic": 600,
    "gsm": 600,
    "spider": 512,
    "smiles": 256,
}

TIER1_STRATEGIES = frozenset({"gcd", "itergen", "cars"})
TIER2_STRATEGIES = frozenset({"unconstrained", "crane", "metadecode"})

# Few-shot caps (full frozen shot lists live in shots.json; CRANE yaml used 8).
TIER1_FEWSHOT_CAP = 0
TIER2_FEWSHOT_CAP = 4


def prompt_tier_for_strategy(strategy: str) -> PromptTier:
    """Map a baseline / evaluation strategy name to prompt tier 1 or 2."""
    key = (strategy or "").strip().lower()
    if key in TIER1_STRATEGIES:
        return 1
    if key in TIER2_STRATEGIES:
        return 2
    raise ValueError(
        f"Unknown strategy {strategy!r}; expected one of "
        f"{sorted(TIER1_STRATEGIES | TIER2_STRATEGIES)}"
    )


def benchmark_max_new_tokens(dataset: str) -> int:
    """Return the uniform per-benchmark generation cap from the evaluation spec."""
    key = "gsm_symbolic" if dataset == "gsm" else dataset
    try:
        return int(BENCHMARK_MAX_NEW_TOKENS[key])
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark for decode cap: {dataset}") from exc


def effective_max_new_tokens(dataset: str, cli_max_steps: int) -> int:
    """Use the benchmark cap unless the CLI requests a smaller budget."""
    cap = benchmark_max_new_tokens(dataset)
    return min(max(1, int(cli_max_steps)), cap)


def fewshot_count_for_tier(
    tier: PromptTier,
    *,
    requested: int | None = None,
) -> int:
    """Return how many frozen few-shot rows to include for a tier."""
    cap = TIER1_FEWSHOT_CAP if tier == 1 else TIER2_FEWSHOT_CAP
    if requested is None:
        return cap
    return min(max(0, int(requested)), cap)


@lru_cache(maxsize=None)
def _load_template(benchmark: str, tier: PromptTier) -> str:
    path = PROMPTS_ROOT / benchmark / f"tier{tier}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt template: {path}")
    return path.read_text()


@lru_cache(maxsize=None)
def _load_shots(benchmark: str) -> Any:
    path = PROMPTS_ROOT / benchmark / "shots.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen few-shot file: {path}")
    return json.loads(path.read_text())


def _gsm_fewshot_block(shots: list[dict[str, Any]], *, tier: PromptTier) -> str:
    lines: list[str] = []
    for shot in shots:
        q = str(shot.get("question", "")).strip()
        if tier == 1:
            response = str(shot.get("response_std", "")).strip()
        else:
            response = str(shot.get("response_cot", "")).strip()
        lines.append(f"Q: {q}")
        lines.append(f"A: {response}")
        lines.append("")
    return "\n".join(lines).rstrip() + ("\n\n" if lines else "")


def _spider_fewshot_block(shots: list[dict[str, Any]], *, tier: PromptTier) -> str:
    lines: list[str] = []
    for shot in shots:
        schema = str(shot.get("schema", "")).strip()
        question = str(shot.get("question", "")).strip()
        sql = str(shot.get("sql", "")).strip().rstrip(";")
        reasoning = str(shot.get("reasoning", "")).strip()
        lines.append(f"Schema: {schema}")
        lines.append(f"Question: {question}")
        if tier == 1:
            lines.append(f"SQL: {sql}")
        else:
            if reasoning:
                lines.append(reasoning)
            lines.append(sql)
        lines.append("")
    return "\n".join(lines).rstrip() + ("\n\n" if lines else "")


_LEGACY_CARS_RESPONSE_LINE = (
    "Your response must be a single SMILES molecule and nothing else."
)
_CARS_DELIMITER_RESPONSE_LINE = (
    "Your response must be a single SMILES molecule wrapped in << and >> "
    "and nothing else."
)


def _smiles_instruction_from_example(example: dict[str, Any]) -> str:
    """Class-level CARS instruction (text before the first exemplar ``Molecule:`` line)."""
    props = str(example.get("smiles_properties") or example.get("properties") or "").strip()
    if props:
        return props
    raw = str(example.get("_smiles_base_prompt") or example.get("prompt") or "")
    head = raw.split("Molecule:")[0].split("SMILES:")[0].strip()
    if head.startswith("Properties:"):
        head = head.split(":", 1)[1].strip()
    return head


def _smiles_cars_instruction(example: dict[str, Any]) -> str:
    """Legacy CARS prose with a single harness-compatible delimiter requirement."""
    text = _smiles_instruction_from_example(example)
    if _LEGACY_CARS_RESPONSE_LINE in text:
        return text.replace(_LEGACY_CARS_RESPONSE_LINE, _CARS_DELIMITER_RESPONSE_LINE)
    if _CARS_DELIMITER_RESPONSE_LINE in text:
        return text
    if text:
        return text + "\n\n" + _CARS_DELIMITER_RESPONSE_LINE
    return _CARS_DELIMITER_RESPONSE_LINE


def render_smiles_cars_prompt(
    example: dict[str, Any],
    *,
    tier: PromptTier,
) -> str:
    """Render a CARS-style class prompt (``data/*.txt``) with shared ``<<`` / ``>>`` scoring."""
    lines = [_smiles_cars_instruction(example), ""]
    for smiles in example.get("prompt_exemplars") or []:
        value = str(smiles).strip()
        if value:
            lines.append(f"Molecule: {value}")
    if tier == 2:
        lines.extend(["", "Reasoning:"])
    lines.append("Molecule: <<")
    return "\n".join(lines)


def render_benchmark_prompt(
    benchmark: str,
    *,
    tier: PromptTier,
    example: dict[str, Any],
    max_fewshots: int | None = None,
) -> str:
    """Render a full in-context prompt for one evaluation example."""
    bench = "gsm_symbolic" if benchmark == "gsm" else benchmark
    template = _load_template(bench, tier)
    shots_raw = _load_shots(bench)
    if max_fewshots is not None:
        shot_limit = max(0, int(max_fewshots))
    else:
        shot_limit = fewshot_count_for_tier(tier)

    if bench == "gsm_symbolic":
        target_question = (
            example.get("question_parsed")
            or example.get("original_question")
            or example.get("question", "")
        )
        target_question = str(target_question or "").strip()
        fewshot = _gsm_fewshot_block(list(shots_raw)[:shot_limit], tier=tier)
        return template.format(
            FEWSHOT_BLOCK=fewshot,
            TARGET_QUESTION=target_question,
        )

    if bench == "spider":
        fewshot = _spider_fewshot_block(list(shots_raw)[:shot_limit], tier=tier)
        return template.format(
            FEWSHOT_BLOCK=fewshot,
            TARGET_SCHEMA=str(example.get("db_info", "")).strip(),
            TARGET_QUESTION=str(example.get("question", "")).strip(),
        )

    if bench == "smiles":
        return render_smiles_cars_prompt(example, tier=tier)

    raise ValueError(f"Unsupported benchmark: {benchmark}")


def format_prompt_for_tier(
    evaluator: Any,
    example: dict[str, Any],
    *,
    benchmark: str,
    tier: PromptTier,
    constrained_suffix: bool = False,
    max_fewshots: int | None = None,
) -> str:
    """Render tier prompt; tier-1 templates end with the grammar start token (``<<``)."""
    prompt = render_benchmark_prompt(
        benchmark, tier=tier, example=example, max_fewshots=max_fewshots
    )
    if not constrained_suffix:
        return prompt
    # Legacy callers may still request a suffix; templates already include ``<<``.
    stripped = prompt.rstrip()
    if stripped.endswith("<<"):
        return prompt
    bench = "gsm_symbolic" if benchmark == "gsm" else benchmark
    if bench in ("gsm_symbolic", "gsm"):
        if stripped.endswith("A:"):
            return stripped + " <<"
        return stripped + "<<"
    if bench == "spider":
        if stripped.endswith("SQL:"):
            return stripped + " <<"
        return stripped + "\nSQL: <<"
    if bench == "smiles":
        if stripped.endswith("Molecule:"):
            return stripped + " <<"
        return stripped + "\nMolecule: <<"
    return prompt
