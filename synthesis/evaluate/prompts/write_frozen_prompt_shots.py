#!/usr/bin/env python3
"""Regenerate frozen few-shot JSON files under ``synthesis/evaluate/prompts/``."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

PROMPTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROMPTS_ROOT.parents[2]
CRANE_PROMPTS = REPO_ROOT / "legacy" / "CRANE" / "src" / "prompt_templates"


def _crane_delimiters(text: str) -> str:
    return text.replace("[[START]]", "<<").replace("[[END]]", ">>")


def write_gsm_shots() -> None:
    """Symbolic GSM-Symbolic few-shots from CRANE ``gsm_symbolic.yaml``."""
    crane_yaml = CRANE_PROMPTS / "gsm_symbolic.yaml"
    cfg = yaml.safe_load(crane_yaml.read_text())
    cot_rows = list(cfg["fewshots"]["cot"]["gsm"])[:8]
    std_rows = list(cfg["fewshots"]["std"]["gsm"])[:8]
    if len(cot_rows) < 8 or len(std_rows) < 8:
        raise RuntimeError("CRANE gsm_symbolic.yaml needs at least 8 std and cot gsm shots")

    shots = []
    for cot_ex, std_ex in zip(cot_rows, std_rows):
        shots.append(
            {
                "question": str(cot_ex["question"]).strip(),
                "response_std": _crane_delimiters(str(std_ex["response"]).strip()),
                "response_cot": _crane_delimiters(str(cot_ex["response"]).strip()),
            }
        )
    out = PROMPTS_ROOT / "gsm_symbolic" / "shots.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(shots, indent=2) + "\n")


def write_spider_shots() -> None:
    """Spider few-shots from CRANE ``spider.yaml`` (IterGen/GCD/CARS use the same tiers)."""
    from synthesis.evaluate.benchmarks.sql_spider.dataset import load_spider

    rows = load_spider(source="auto", limit=200)
    by_q = {r["question"]: r for r in rows}
    crane_yaml = CRANE_PROMPTS / "spider.yaml"
    cfg = yaml.safe_load(crane_yaml.read_text())

    def _resolve_row(question: str) -> dict:
        row = by_q.get(question)
        if row is None and "stadium" in question.lower():
            row = next(
                (
                    r
                    for r in rows
                    if "names" in r["question"].lower() and "stadium" in r["question"].lower()
                ),
                None,
            )
        if row is None:
            raise RuntimeError(f"No Spider schema row for few-shot question: {question!r}")
        return row

    shots = []
    for ex in cfg["fewshots"]["cot"]["text"]:
        question = str(ex["question"]).strip()
        row = _resolve_row(question)
        resp = str(ex["response"]).strip()
        if "\n" in resp:
            reasoning, sql = resp.split("\n", 1)
        else:
            reasoning, sql = "", resp
        shots.append(
            {
                "schema": row.get("db_info", "").strip(),
                "db_id": row.get("db_id", ""),
                "question": question,
                "sql": sql.strip().rstrip(";"),
                "reasoning": reasoning.strip(),
            }
        )

    if len(shots) < 2:
        raise RuntimeError(f"Expected at least 2 Spider shots, got {len(shots)}")

    out = PROMPTS_ROOT / "spider" / "shots.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(shots, indent=2) + "\n")


def write_smiles_shots() -> None:
    """CARS-style class prompts and exemplar molecules from ``benchmarks/smiles/data``."""
    from synthesis.evaluate.benchmarks.smiles.dataset import SMILES_CLASSES, prompt_exemplars_for_class
    from synthesis.evaluate.prompt_tiers import smiles_class_properties

    payload: dict[str, list[dict[str, str]]] = {}
    for cls in SMILES_CLASSES:
        props = smiles_class_properties(cls)
        payload[cls] = [
            {
                "properties": props,
                "smiles": mol,
            }
            for mol in prompt_exemplars_for_class(cls)
        ]
    out = PROMPTS_ROOT / "smiles" / "shots.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    write_gsm_shots()
    write_spider_shots()
    write_smiles_shots()
    print(f"Wrote frozen shots under {PROMPTS_ROOT}")


if __name__ == "__main__":
    main()
