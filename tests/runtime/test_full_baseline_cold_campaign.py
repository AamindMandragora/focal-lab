import hashlib
import json
from pathlib import Path

import pytest

from scripts.runtime import build_full_baseline_cold_manifest as builder
from scripts.runtime import run_cold_synthesis_queue as queue


def _write_artifact(path: Path, *, correct: int, syntax: int, total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "accuracy": correct / total,
                "syntax_rate": syntax / total,
                "metrics": {"num_examples": total},
                "answers": [
                    {"question": f"q-{index}", "generated_answer": f"a-{index}"}
                    for index in range(total)
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_all_artifacts(repo: Path, *, perfect: bool = False) -> None:
    for cohort in builder.COHORTS:
        for model in builder.MODELS:
            for index, strategy in enumerate(builder.STRATEGIES):
                total = cohort.sample_size
                correct = total if perfect and strategy == "cars" else min(total - 1, index + 1)
                syntax = min(total, total - index)
                _write_artifact(
                    builder.baseline_artifact(repo, cohort, model, strategy),
                    correct=correct,
                    syntax=syntax,
                    total=total,
                )


def test_build_campaign_uses_all_five_baselines_and_exact_thresholds(tmp_path: Path) -> None:
    _write_all_artifacts(tmp_path)

    evidence, manifest = builder.build_campaign(tmp_path, "a" * 40)

    assert len(evidence["cells"]) == 20
    assert len(manifest["jobs"]) == 20
    assert sum(job["max_iterations"] for job in manifest["jobs"]) == 800
    cell = evidence["cells"]["gsm-qwen25-1p5b"]
    assert [row["strategy"] for row in cell["baselines"]] == list(builder.STRATEGIES)
    assert all(len(row["source_sha256"]) == 64 for row in cell["baselines"])
    assert cell["max_accuracy_count"] == 5
    assert cell["max_syntax_count"] == 49
    job = next(job for job in manifest["jobs"] if job["cell_id"] == "gsm-qwen25-1p5b")
    assert job["baseline_num_correct"] == 5
    assert job["min_accuracy"] == 6 / 49
    assert job["min_syntax_rate"] == 0.9
    assert job["threshold_policy"] == "strict_plus_one"
    assert job["claude_expected_account"] == "aadivya@fermi.ai"
    assert job["eval_max_steps"] == 900


def test_perfect_baseline_uses_the_approved_95_percent_exception(tmp_path: Path) -> None:
    _write_all_artifacts(tmp_path, perfect=True)

    evidence, manifest = builder.build_campaign(tmp_path, "b" * 40)

    cell = evidence["cells"]["spider-qwen35-4b"]
    job = next(job for job in manifest["jobs"] if job["cell_id"] == "spider-qwen35-4b")
    assert cell["max_accuracy_count"] == 300
    assert job["min_accuracy"] == 0.95
    assert job["threshold_policy"] == "perfect_baseline_95_percent_exception"


def test_incomplete_or_tampered_baseline_blocks_manifest(tmp_path: Path) -> None:
    _write_all_artifacts(tmp_path)
    missing = builder.baseline_artifact(
        tmp_path, builder.COHORTS[0], builder.MODELS[0], "gcd"
    )
    missing.unlink()

    with pytest.raises(builder.CampaignError, match="missing baseline artifact"):
        builder.build_campaign(tmp_path, "c" * 40)

    _write_artifact(missing, correct=1, syntax=49, total=49)
    payload = json.loads(missing.read_text(encoding="utf-8"))
    payload["answers"].pop()
    missing.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(builder.CampaignError, match="answer count"):
        builder.build_campaign(tmp_path, "c" * 40)


def test_evidence_hash_binds_each_raw_artifact(tmp_path: Path) -> None:
    _write_all_artifacts(tmp_path)
    evidence, _ = builder.build_campaign(tmp_path, "d" * 40)
    row = evidence["cells"]["smiles-isocyanates-qwen35-4b"]["baselines"][0]
    source = tmp_path / row["source_artifact"]

    assert row["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_cold_environment_uses_the_manifest_bound_claude_account(tmp_path: Path) -> None:
    _write_all_artifacts(tmp_path)
    _, manifest = builder.build_campaign(tmp_path, "e" * 40)
    job = manifest["jobs"][0]

    environment = queue.synthesis_environment(job, (0, 2), {}, tmp_path)

    assert environment["CSD_CLAUDE_CONFIG_DIR"] == "/home/aadivyar/.claude-csd-synthesis"
    assert environment["CSD_CLAUDE_EXPECTED_ACCOUNT"] == "aadivya@fermi.ai"
