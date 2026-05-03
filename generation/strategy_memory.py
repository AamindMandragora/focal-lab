from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BEST_DIR = OUTPUTS_DIR / "best"
WORST_DIR = OUTPUTS_DIR / "worst"

_HELPER_RE = re.compile(r"helpers\.([A-Za-z_]\w*)")
_RATIONALE_RE = re.compile(
    r"(?:#|//)\s*CSD_RATIONALE_BEGIN\s*(.*?)(?:#|//)\s*CSD_RATIONALE_END",
    re.DOTALL,
)


def task_family_from_description(task_description: str) -> str:
    text = task_description.lower()
    if "gsm" in text or "math" in text or "arithmetic" in text:
        return "gsm_symbolic"
    if "spider" in text or "sql" in text:
        return "spider"
    if "chem-cot-bench" in text or "chem cot bench" in text or "chemistry" in text or "molecule" in text:
        return "chem_cot_bench"
    return "general"


def _family_matches(name: str, family: str) -> bool:
    lowered = name.lower()
    if family == "gsm_symbolic":
        return "gsm" in lowered
    if family == "spider":
        return "spider" in lowered or "sql" in lowered
    if family == "chem_cot_bench":
        return any(marker in lowered for marker in ("chem", "molecule", "reaction"))
    return family in lowered


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_metrics(blob: Any) -> list[dict[str, float]]:
    found: list[dict[str, float]] = []
    if isinstance(blob, dict):
        if {"accuracy", "format_rate", "syntax_rate"} <= set(blob):
            try:
                found.append(
                    {
                        "accuracy": float(blob["accuracy"]),
                        "format_rate": float(blob["format_rate"]),
                        "syntax_rate": float(blob["syntax_rate"]),
                    }
                )
            except Exception:
                pass
        for value in blob.values():
            found.extend(_extract_metrics(value))
    elif isinstance(blob, list):
        for item in blob:
            found.extend(_extract_metrics(item))
    return found


def _best_metric_summary(run_dir: Path) -> str:
    report_candidates = [
        run_dir / "success_report.json",
        run_dir / "failure_report.json",
    ]
    metrics: list[dict[str, float]] = []
    for report_path in report_candidates:
        if report_path.exists():
            metrics.extend(_extract_metrics(_read_json(report_path)))
    if not metrics:
        return "metrics unavailable"
    best = max(
        metrics,
        key=lambda item: (2.0 * item["accuracy"]) + item["format_rate"] + item["syntax_rate"],
    )
    return (
        f"accuracy={best['accuracy']:.1%}, "
        f"format={best['format_rate']:.1%}, "
        f"syntax={best['syntax_rate']:.1%}"
    )


def _helper_summary(code: str, *, limit: int = 6) -> str:
    helpers = []
    seen: set[str] = set()
    for helper in _HELPER_RE.findall(code):
        if helper not in seen:
            helpers.append(helper)
            seen.add(helper)
        if len(helpers) >= limit:
            break
    return ", ".join(helpers) if helpers else "no helper summary"


def _rationale_summary(code: str, *, max_chars: int = 240) -> str:
    match = _RATIONALE_RE.search(code)
    if match is None:
        lines = [
            line.strip("#/ ").strip()
            for line in code.splitlines()
            if line.strip().startswith("#") or line.strip().startswith("//")
        ]
        summary = " ".join(lines[:4]).strip()
    else:
        lines = [line.strip("#/ ").strip() for line in match.group(1).splitlines() if line.strip()]
        summary = " ".join(lines).strip()
    summary = re.sub(r"\s+", " ", summary)
    if len(summary) > max_chars:
        return summary[: max_chars - 3] + "..."
    return summary or "No rationale summary available."


def _strategy_excerpt(
    code: str,
    *,
    strategy_language: str = "python",
    max_lines: int = 24,
    max_chars: int = 1400,
) -> str:
    body_lines: list[str] = []
    capture = False
    saw_markers = False
    for raw_line in code.splitlines():
        line = raw_line.rstrip()
        if "# QWEN_INSERT_STRATEGY_BEGIN" in line:
            capture = True
            saw_markers = True
            continue
        if "# QWEN_INSERT_STRATEGY_END" in line:
            break
        if not capture:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        body_lines.append(stripped)
        if len(body_lines) >= max_lines:
            break
    if not saw_markers and strategy_language == "python":
        in_body = False
        in_signature = False
        body_indent: int | None = None
        body_lines = []
        for raw_line in code.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if stripped.startswith("def MyCSDStrategy("):
                in_signature = True
                continue
            if in_signature:
                if stripped.endswith(":"):
                    in_body = True
                    in_signature = False
                continue
            if not in_body:
                continue
            if body_indent is None:
                if not stripped:
                    continue
                body_indent = len(line) - len(line.lstrip())
            current_indent = len(line) - len(line.lstrip())
            if stripped and current_indent < body_indent:
                break
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if stripped in {
                "helpers = CSDHelpers(lm, parser)",
                "lm.ValidTokensIdsLogitsAlways()",
                "generated = []",
                "stepsLeft = maxSteps",
                "remainingSteps = stepsLeft",
                "return generated, remainingSteps",
            }:
                continue
            body_lines.append(stripped)
            if len(body_lines) >= max_lines:
                break
    if not saw_markers and strategy_language == "dafny":
        in_method = False
        in_body = False
        body_lines = []
        for raw_line in code.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if stripped.startswith("method MyCSDStrategy("):
                in_method = True
                continue
            if not in_method:
                continue
            if not in_body:
                if stripped == "{":
                    in_body = True
                continue
            if stripped == "}":
                break
            if not stripped or stripped.startswith("//"):
                continue
            if stripped in {
                "var helpers := new CSDHelpers(lm, parser);",
                "lm.ValidTokensIdsLogitsAlways();",
                "generated := [];",
                "var stepsLeft := maxSteps;",
                "remainingSteps := stepsLeft;",
            }:
                continue
            body_lines.append(stripped)
            if len(body_lines) >= max_lines:
                break
    excerpt = "\n".join(body_lines)
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 3] + "..."
    return excerpt


def _generation_diagnostics_summary(
    generation_diagnostics: list[dict[str, object]] | None,
    *,
    max_items: int = 3,
) -> list[str]:
    if not generation_diagnostics:
        return []

    lines: list[str] = []
    for item in generation_diagnostics:
        if not isinstance(item, dict):
            continue
        candidate = item.get("candidate", "?")
        issue = re.sub(r"\s+", " ", str(item.get("issue", "")).strip())
        if not issue:
            if item.get("raw_output_empty"):
                issue = "empty raw output"
            elif item.get("extracted_strategy_empty"):
                issue = "empty extracted strategy"
            else:
                issue = "rejected candidate"
        if len(issue) > 220:
            issue = issue[:217] + "..."
        strategy_excerpt = re.sub(
            r"\s+",
            " ",
            str(item.get("final_strategy") or item.get("extracted_strategy") or "").strip(),
        )
        if strategy_excerpt:
            if len(strategy_excerpt) > 180:
                strategy_excerpt = strategy_excerpt[:177] + "..."
            issue += f" | excerpt: {strategy_excerpt}"
        lines.append(f"candidate {candidate}: {issue}")
        if len(lines) >= max_items:
            break
    return lines


def build_prompt_memory(task_description: str, *, strategy_language: str = "python") -> str:
    family = task_family_from_description(task_description)
    sections: list[str] = []

    best_lines: list[str] = []
    if BEST_DIR.exists():
        run_dirs = sorted(
            [path for path in BEST_DIR.iterdir() if path.is_dir() and _family_matches(path.name, family)],
            key=lambda path: path.name,
            reverse=True,
        )
        for run_dir in run_dirs[:2]:
            strategy_filename = "GeneratedCSD.py"
            if strategy_language == "dafny" and (run_dir / "GeneratedCSD.dfy").exists():
                strategy_filename = "GeneratedCSD.dfy"
            strategy_path = run_dir / strategy_filename
            code = strategy_path.read_text(encoding="utf-8") if strategy_path.exists() else ""
            best_lines.append(
                f"- `{run_dir.name}`: {_best_metric_summary(run_dir)}. "
                f"Shape: {_rationale_summary(code)} "
                f"Helpers: {_helper_summary(code)}. "
                f"Code excerpt:\n```{('dafny' if strategy_language == 'dafny' else 'python')}\n"
                f"{_strategy_excerpt(code, strategy_language=strategy_language)}\n```"
            )
    if best_lines:
        sections.append(
            "### Best Prior Runs\n"
            "Use these as stepping stones rather than fixed templates.\n"
            + "\n".join(best_lines)
        )

    worst_family_dir = WORST_DIR / family
    worst_lines: list[str] = []
    if worst_family_dir.exists():
        worst_files = sorted(worst_family_dir.glob("*.md"), reverse=True)
        for path in worst_files[:4]:
            text = path.read_text(encoding="utf-8")
            text = re.sub(r"\s+", " ", text)
            if len(text) > 320:
                text = text[:317] + "..."
            worst_lines.append(f"- `{path.stem}`: {text}")
    if worst_lines:
        sections.append(
            "### Failed Directions To Avoid\n"
            "Read these before writing a new strategy so you do not repeat the same dead ends.\n"
            + "\n".join(worst_lines)
        )

    if not sections:
        return ""

    return (
        "## Strategy Memory\n"
        "Read this prior-run context before producing a new strategy.\n\n"
        + "\n\n".join(sections)
    )


def record_failure_memory(
    task_description: str,
    *,
    attempt_number: int,
    stage: str,
    attempt_brief: str,
    revision_instruction: str,
    error_summary: str,
    strategy_code: str,
    change_diff: str,
    metrics: dict[str, float] | None = None,
    generation_diagnostics: list[dict[str, object]] | None = None,
) -> Path:
    family = task_family_from_description(task_description)
    out_dir = WORST_DIR / family
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{timestamp}_attempt{attempt_number}_{stage}.md"

    lines = [
        f"Task family: {family}.",
        f"Attempt {attempt_number} failed at {stage}.",
        f"Brief: {attempt_brief}",
    ]
    if metrics is not None:
        lines.append(
            "Metrics: "
            f"accuracy={metrics.get('accuracy', 0.0):.1%}, "
            f"format={metrics.get('format_rate', 0.0):.1%}, "
            f"syntax={metrics.get('syntax_rate', 0.0):.1%}."
        )
    if revision_instruction:
        lines.append(f"Revision guidance: {revision_instruction}")
    if error_summary:
        compact_error = re.sub(r"\s+", " ", error_summary).strip()
        if len(compact_error) > 700:
            compact_error = compact_error[:697] + "..."
        lines.append(f"Why it failed: {compact_error}")
    if change_diff and change_diff != "(initial strategy)":
        compact_diff = re.sub(r"\s+", " ", change_diff).strip()
        if len(compact_diff) > 500:
            compact_diff = compact_diff[:497] + "..."
        lines.append(f"What changed: {compact_diff}")
    if strategy_code:
        lines.append(f"Strategy shape: {_rationale_summary(strategy_code, max_chars=320)}")
        lines.append(f"Helpers used: {_helper_summary(strategy_code, limit=10)}.")
    diagnostic_lines = _generation_diagnostics_summary(generation_diagnostics)
    if diagnostic_lines:
        lines.append("Generation diagnostics:")
        lines.extend(f"- {line}" for line in diagnostic_lines)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
