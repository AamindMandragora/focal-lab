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
    # Tier-2 ceiling (CRANE / unconstrained CoT + <<SMILES>>). Tier-1 uses ``smiles_tier1_max_new_tokens``.
    "smiles": 256,
}

# Longest frozen exemplar SMILES is ~96 characters. Tier-1 grammars are unbounded (``rest*``) and
# SynCode rarely accepts EOS early, so a tight cap prevents repetitive padding to the old 256 limit.
SMILES_TIER1_MAX_NEW_TOKENS = 96
SMILES_TIER2_MAX_NEW_TOKENS = 256

TIER1_STRATEGIES = frozenset({"gcd", "itergen", "cars"})
TIER2_STRATEGIES = frozenset({"unconstrained", "crane", "metadecode"})

# Few-shot caps (full frozen shot lists live in shots.json; CRANE yaml used 8).
TIER1_FEWSHOT_CAP = 0
TIER2_FEWSHOT_CAP = 4

# SMILES CARS prompts always use exactly eight in-context exemplar molecules.
SMILES_FEWSHOT_COUNT = 8


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


def smiles_tier1_max_new_tokens(decode_cap: int) -> int:
    """Decode budget for GCD / IterGen / CARS (body-only SMILES, no CoT)."""
    return min(SMILES_TIER1_MAX_NEW_TOKENS, max(1, int(decode_cap)))


def smiles_tier2_max_new_tokens(decode_cap: int) -> int:
    """Decode budget for CRANE / unconstrained (brief reasoning + delimited answer)."""
    return min(SMILES_TIER2_MAX_NEW_TOKENS, max(1, int(decode_cap)))


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


_GSM_TIER1_CARS_INFO = (
    "The expression must be symbolic: include at least one variable name from the question "
    "(not a plain numeric literal alone).\n\n"
)


@lru_cache(maxsize=None)
def _load_shots(benchmark: str) -> Any:
    path = PROMPTS_ROOT / benchmark / "shots.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen few-shot file: {path}")
    return json.loads(path.read_text())


def _tier1_undelimited_answer(response_std: str) -> str:
    """Strip legacy ``<<`` / ``>>`` wrappers from tier-1 few-shot answers."""
    text = str(response_std or "").strip()
    if text.startswith("<<") and text.endswith(">>"):
        return text[2:-2].strip()
    return text


def _gsm_fewshot_block(shots: list[dict[str, Any]], *, tier: PromptTier) -> str:
    lines: list[str] = []
    for shot in shots:
        q = str(shot.get("question", "")).strip()
        if tier == 1:
            response = _tier1_undelimited_answer(str(shot.get("response_std", "")))
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
            lines.append(f"<<{sql}>>")
        lines.append("")
    return "\n".join(lines).rstrip() + ("\n\n" if lines else "")


def _smiles_class_label(class_name: str) -> str:
    return str(class_name).replace("_", " ")


def smiles_no_reuse_clause() -> str:
    """Task requirement shared across SMILES prompt tiers (not strategy advice)."""
    return (
        "Do not copy, quote, or repeat any example SMILES verbatim. "
        "Your answer must be a novel molecule for the requested class."
    )


def smiles_class_properties(class_name: str) -> str:
    """Tier-neutral class instruction (eight in-context examples are appended separately)."""
    label = _smiles_class_label(class_name)
    return (
        f"You are an expert in chemistry. Below are exactly {SMILES_FEWSHOT_COUNT} example "
        f"{label} molecules, each shown as `Molecule: <SMILES>`. "
        "These lines are in-context demonstrations only — not your answer.\n\n"
        f"Your task: generate exactly one new, valid {label} molecule that is different "
        f"from all {SMILES_FEWSHOT_COUNT} examples. "
        f"{smiles_no_reuse_clause()}"
    )


def _smiles_instruction_from_example(example: dict[str, Any]) -> str:
    """Class-level instruction; prefer stored properties, else rebuild from class name."""
    props = str(example.get("smiles_properties") or example.get("properties") or "").strip()
    if props and "exactly" in props.lower() and str(SMILES_FEWSHOT_COUNT) in props:
        return props
    class_name = str(example.get("class_name") or "").strip()
    if class_name:
        return smiles_class_properties(class_name)
    raw = str(example.get("_smiles_base_prompt") or example.get("prompt") or "")
    head = raw.split("Molecule:")[0].split("SMILES:")[0].strip()
    if head.startswith("Properties:"):
        head = head.split(":", 1)[1].strip()
    return head


def _smiles_tier1_suffix(*, delimited_answer: bool = False) -> str:
    if delimited_answer:
        return (
            "\n\nOutput exactly one new SMILES string inside double angle brackets, "
            "for example: <<CCO>>. "
            "Do not include reasoning, labels, or multiple molecules. "
            f"{smiles_no_reuse_clause()}"
        )
    return (
        "\n\nOutput exactly one SMILES string with no other text "
        "(no labels, no reasoning, no multiple molecules). "
        f"{smiles_no_reuse_clause()}"
    )


def _smiles_tier2_suffix() -> str:
    return (
        "\n\nYou may use at most two short sentences of reasoning, then output your final "
        "answer immediately as a single SMILES string inside double angle brackets, "
        f"for example: <<CCO>>. Do not enumerate or repeat the example molecules. "
        f"{smiles_no_reuse_clause()}"
    )


def strategy_uses_reasoning_prompt(strategy_code: str) -> bool:
    """
    Infer whether evaluation should use the tier-2 (CoT) SMILES prompt for this CSD.

    Strategies that only open a constrained span immediately (``OpenConstrainedSpan``
    with a short free prefix) or decode a fully constrained object without free LM
    steps use the tier-1 answer-only prompt text while keeping delimited grammars.
    """
    from synthesis.generate.rationale import extract_rationale

    extracted = extract_rationale(strategy_code)
    body = (
        extracted.body_without_rationale.strip()
        if extracted.has_markers
        else strategy_code.strip()
    )
    if "UnconstrainedChunk" in body:
        return True
    if "OpenConstrainedSpan" in body and "freePrefixLimit" in body:
        return False
    if "UnconstrainedStep" in body:
        return True
    return False


def smiles_grammar_tier_for_csd() -> PromptTier:
    """Compiled CSD evaluation keeps tier-2 delimited grammars regardless of prompt tier."""
    return 2


def render_smiles_cars_prompt(
    example: dict[str, Any],
    *,
    tier: PromptTier,
    delimited_answer: bool = False,
) -> str:
    """Render a CARS-style class prompt; tier 1 is body-only, tier 2 ends before CoT."""
    instruction = _smiles_instruction_from_example(example)
    if tier == 1:
        instruction += _smiles_tier1_suffix(delimited_answer=delimited_answer)
    else:
        instruction += _smiles_tier2_suffix()

    lines = [instruction, ""]
    exemplars = list(example.get("prompt_exemplars") or [])[:SMILES_FEWSHOT_COUNT]
    for smiles in exemplars:
        value = str(smiles).strip()
        if value:
            lines.append(f"Molecule: {value}")
    good_results = list(example.get("smiles_good_results") or [])
    bad_results = list(example.get("smiles_bad_results") or [])
    if tier == 2:
        if good_results or bad_results:
            from synthesis.evaluate.benchmarks.smiles.prompt_state import format_attempt_suffix

            lines.append(format_attempt_suffix(good_results, bad_results).rstrip("\n"))
        else:
            lines.extend(["", "Reasoning:"])
    return "\n".join(lines)


def render_benchmark_prompt(
    benchmark: str,
    *,
    tier: PromptTier,
    example: dict[str, Any],
    max_fewshots: int | None = None,
    strategy: str | None = None,
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
        cars_info = ""
        if tier == 1 and (strategy or "").strip().lower() == "cars":
            cars_info = _GSM_TIER1_CARS_INFO
        return template.format(
            CARS_INFO=cars_info,
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


def configure_smiles_eval_prompts(evaluator: Any, strategy_code: str) -> None:
    """Align SMILES prompt text with a compiled CSD while keeping delimited decode grammar."""
    uses_reasoning = strategy_uses_reasoning_prompt(strategy_code)
    evaluator.use_reasoning_prompt = uses_reasoning
    evaluator.prompt_tier = 2 if uses_reasoning else 1
    evaluator.grammar_prompt_tier = smiles_grammar_tier_for_csd()
    evaluator.smiles_delimited_answer_prompt = evaluator.prompt_tier == 1


def render_smiles_eval_prompt(evaluator: Any, example: dict[str, Any]) -> str:
    """Render a SMILES row prompt using evaluator tier / delimiter flags."""
    tier: PromptTier = getattr(evaluator, "prompt_tier", 2)
    delimited = bool(getattr(evaluator, "smiles_delimited_answer_prompt", False))
    return render_smiles_cars_prompt(example, tier=tier, delimited_answer=delimited)


def format_prompt_for_tier(
    evaluator: Any,
    example: dict[str, Any],
    *,
    benchmark: str,
    tier: PromptTier,
    constrained_suffix: bool = False,
    max_fewshots: int | None = None,
    strategy: str | None = None,
) -> str:
    """Render tier prompt. ``constrained_suffix`` adds ``<<``/``>>`` to tier-1 SMILES text."""
    bench = "gsm_symbolic" if benchmark == "gsm" else benchmark
    if bench == "smiles":
        return render_smiles_cars_prompt(
            example,
            tier=tier,
            delimited_answer=constrained_suffix,
        )
    return render_benchmark_prompt(
        benchmark,
        tier=tier,
        example=example,
        max_fewshots=max_fewshots,
        strategy=strategy,
    )
