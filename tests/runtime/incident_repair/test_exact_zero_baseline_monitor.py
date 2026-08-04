import hashlib
import json
from pathlib import Path

import pytest

from scripts.runtime.incident_repair import exact_zero_baseline_monitor as monitor
from scripts.runtime.incident_repair.exact_zero_baseline_monitor import (
    load_zero_acceptances,
    review_manifest,
    write_report_and_gate,
)


LABEL = "spider-qwen35-2b-itergen"
REPAIR_ROOT = Path("outputs/baselines/exact-zero-repair-20260804")


def _write_artifact(
    path: Path,
    *,
    answers: list[str],
    accuracy: float = 0.0,
    syntax_rate: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "syntax_rate": syntax_rate,
                "metrics": {"num_examples": len(answers)},
                "answers": [
                    {"question": f"q-{index}", "generated_answer": answer}
                    for index, answer in enumerate(answers)
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_manifest(repo: Path, artifact: Path) -> Path:
    source = repo / "source" / "original.json"
    _write_artifact(
        source,
        answers=[f"source-zero-{index}" for index in range(300)],
    )
    manifest = repo / "repair-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "campaign": "exact-zero-repair-test",
                "source_exact_zero_count": 1,
                "repair_output_root": str(REPAIR_ROOT),
                "rows": [
                    {
                        "label": LABEL,
                        "cell_id": "spider-qwen35-2b",
                        "strategy": "itergen",
                        "source_artifact": str(source.relative_to(repo)),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "replacement_artifact": str(artifact.relative_to(repo)),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_blank_exact_zero_is_quarantined_and_blocks_synthesis(tmp_path: Path) -> None:
    artifact = tmp_path / REPAIR_ROOT / "itergen.json"
    _write_artifact(artifact, answers=[" "] * 300)
    manifest = _write_manifest(tmp_path, artifact)

    report = review_manifest(tmp_path, manifest, zero_acceptances={})

    row = report["rows"][0]
    assert row["status"] == "quarantined_system_failure"
    assert "all generated answers are blank" in row["reason"]
    assert row["artifact_preserved"] is True
    assert report["synthesis_blocked"] is True
    assert report["counts"] == {
        "accepted": 0,
        "pending": 0,
        "needs_skeptical_review": 0,
        "quarantined_system_failure": 1,
    }


def test_diverse_exact_zero_requires_hash_bound_skeptical_acceptance(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / REPAIR_ROOT / "itergen.json"
    _write_artifact(artifact, answers=[f"bad-{index}" for index in range(300)])
    manifest = _write_manifest(tmp_path, artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    blocked = review_manifest(tmp_path, manifest, zero_acceptances={})
    wrong_hash = review_manifest(
        tmp_path,
        manifest,
        zero_acceptances={LABEL: "f" * 64},
    )
    accepted = review_manifest(
        tmp_path,
        manifest,
        zero_acceptances={LABEL: digest},
    )

    assert blocked["rows"][0]["status"] == "needs_skeptical_review"
    assert blocked["rows"][0]["nonblank_answer_count"] == 300
    assert blocked["rows"][0]["unique_generated_answer_count"] == 300
    assert wrong_hash["rows"][0]["status"] == "needs_skeptical_review"
    assert accepted["rows"][0]["status"] == "accepted_reviewed_zero"
    assert accepted["baseline_review_complete"] is True
    assert accepted["synthesis_blocked"] is True
    assert accepted["block_reason"] == "awaiting corrected evidence and queue validation"


def test_nonzero_structurally_complete_artifact_is_accepted(tmp_path: Path) -> None:
    artifact = tmp_path / REPAIR_ROOT / "itergen.json"
    _write_artifact(
        artifact,
        answers=[f"answer-{index}" for index in range(300)],
        accuracy=1 / 300,
        syntax_rate=2 / 300,
    )
    manifest = _write_manifest(tmp_path, artifact)

    report = review_manifest(tmp_path, manifest, zero_acceptances={})

    row = report["rows"][0]
    assert row["status"] == "accepted"
    assert row["num_correct"] == 1
    assert row["syntax_count"] == 2
    assert report["baseline_review_complete"] is True
    assert report["synthesis_blocked"] is True


def test_missing_artifact_is_pending_and_gate_files_are_atomic(tmp_path: Path) -> None:
    artifact = tmp_path / REPAIR_ROOT / "itergen.json"
    manifest = _write_manifest(tmp_path, artifact)
    report_path = tmp_path / "saved-results" / "monitor.json"
    block_path = tmp_path / ".context" / "synthesis.blocked"

    pending = review_manifest(tmp_path, manifest, zero_acceptances={})
    write_report_and_gate(pending, report_path, block_path)

    assert pending["rows"][0]["status"] == "pending"
    assert json.loads(report_path.read_text(encoding="utf-8")) == pending
    assert block_path.is_file()

    _write_artifact(
        artifact,
        answers=[f"answer-{index}" for index in range(300)],
        accuracy=1 / 300,
        syntax_rate=1 / 300,
    )
    accepted = review_manifest(tmp_path, manifest, zero_acceptances={})
    write_report_and_gate(accepted, report_path, block_path)

    assert block_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8")) == accepted


def test_zero_acceptance_requires_a_written_review_reason(tmp_path: Path) -> None:
    path = tmp_path / "acceptances.json"
    path.write_text(
        json.dumps(
            {
                "acceptances": [
                    {"label": LABEL, "source_sha256": "a" * 64}
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="review reason"):
        load_zero_acceptances(path)

    path.write_text(
        json.dumps(
            {
                "acceptances": [
                    {
                        "label": LABEL,
                        "source_sha256": "a" * 64,
                        "review_reason": "300 distinct nonblank outputs; parser and logs healthy",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_zero_acceptances(path) == {LABEL: "a" * 64}


def test_manifest_and_block_are_bound_to_the_manifest_hash(tmp_path: Path) -> None:
    artifact = tmp_path / REPAIR_ROOT / "itergen.json"
    _write_artifact(
        artifact,
        answers=[f"answer-{index}" for index in range(300)],
        accuracy=1 / 300,
        syntax_rate=1 / 300,
    )
    manifest = _write_manifest(tmp_path, artifact)
    report_path = tmp_path / "monitor.json"
    block_path = tmp_path / "synthesis.blocked"

    report = review_manifest(tmp_path, manifest, zero_acceptances={})
    write_report_and_gate(report, report_path, block_path)

    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert report["manifest_sha256"] == digest
    assert json.loads(block_path.read_text(encoding="utf-8"))["manifest_sha256"] == digest


def test_manifest_rejects_duplicate_labels(tmp_path: Path) -> None:
    artifact = tmp_path / REPAIR_ROOT / "itergen.json"
    _write_artifact(artifact, answers=[f"answer-{index}" for index in range(300)])
    manifest = _write_manifest(tmp_path, artifact)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["source_exact_zero_count"] = 2
    payload["rows"].append(dict(payload["rows"][0]))
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        review_manifest(tmp_path, manifest, zero_acceptances={})


def test_manifest_rejects_tampered_frozen_source(tmp_path: Path) -> None:
    artifact = tmp_path / REPAIR_ROOT / "itergen.json"
    _write_artifact(artifact, answers=[f"answer-{index}" for index in range(300)])
    manifest = _write_manifest(tmp_path, artifact)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    source = tmp_path / payload["rows"][0]["source_artifact"]
    source.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="source SHA-256"):
        review_manifest(tmp_path, manifest, zero_acceptances={})


def test_manifest_accepts_frozen_exact_zero_system_failure(tmp_path: Path) -> None:
    artifact = tmp_path / REPAIR_ROOT / "itergen.json"
    _write_artifact(
        artifact,
        answers=[f"answer-{index}" for index in range(300)],
        accuracy=1 / 300,
        syntax_rate=1 / 300,
    )
    manifest = _write_manifest(tmp_path, artifact)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    source = tmp_path / payload["rows"][0]["source_artifact"]
    _write_artifact(source, answers=["same malformed answer"] * 300)
    payload["rows"][0]["source_sha256"] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = review_manifest(tmp_path, manifest, zero_acceptances={})

    assert report["rows"][0]["status"] == "accepted"


@pytest.mark.parametrize("invalid_zero", [False, 1e-12, "0"])
def test_manifest_requires_literal_numeric_zero_scores(
    tmp_path: Path, invalid_zero: object
) -> None:
    artifact = tmp_path / REPAIR_ROOT / "itergen.json"
    _write_artifact(
        artifact,
        answers=[f"answer-{index}" for index in range(300)],
        accuracy=1 / 300,
        syntax_rate=1 / 300,
    )
    manifest = _write_manifest(tmp_path, artifact)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    source = tmp_path / manifest_payload["rows"][0]["source_artifact"]
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    source_payload["accuracy"] = invalid_zero
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    manifest_payload["rows"][0]["source_sha256"] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="literal numeric zero"):
        review_manifest(tmp_path, manifest, zero_acceptances={})




def test_manifest_requires_the_exact_configured_repair_root(tmp_path: Path) -> None:
    artifact = tmp_path / REPAIR_ROOT / "itergen.json"
    _write_artifact(artifact, answers=[f"answer-{index}" for index in range(300)])
    manifest = _write_manifest(tmp_path, artifact)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["repair_output_root"] = "outputs/baselines"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="repair output root"):
        review_manifest(tmp_path, manifest, zero_acceptances={})


def test_manifest_rejects_source_replacement_overlap(tmp_path: Path) -> None:
    artifact = tmp_path / REPAIR_ROOT / "itergen.json"
    _write_artifact(artifact, answers=[f"bad-{index}" for index in range(300)])
    manifest = _write_manifest(tmp_path, artifact)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    row = payload["rows"][0]
    row["source_artifact"] = row["replacement_artifact"]
    row["source_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        review_manifest(tmp_path, manifest, zero_acceptances={})


def test_replacement_hash_and_scores_use_the_same_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / REPAIR_ROOT / "itergen.json"
    _write_artifact(
        artifact,
        answers=[f"answer-{index}" for index in range(300)],
        accuracy=1 / 300,
        syntax_rate=1 / 300,
    )
    original_bytes = artifact.read_bytes()
    original_digest = hashlib.sha256(original_bytes).hexdigest()
    manifest = _write_manifest(tmp_path, artifact)
    real_reader = monitor.campaign_builder._read_baseline_bytes

    def mutate_after_read(raw, *args, **kwargs):
        _write_artifact(artifact, answers=[f"changed-{index}" for index in range(300)])
        return real_reader(raw, *args, **kwargs)

    monkeypatch.setattr(monitor.campaign_builder, "_read_baseline_bytes", mutate_after_read)
    report = review_manifest(tmp_path, manifest, zero_acceptances={})
    row = report["rows"][0]

    assert row["source_sha256"] == original_digest
    assert row["num_correct"] == 1
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() != original_digest






def test_malformed_acceptance_fails_closed_and_next_poll_recovers(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / REPAIR_ROOT / "itergen.json"
    _write_artifact(
        artifact,
        answers=[f"answer-{index}" for index in range(300)],
        accuracy=1 / 300,
        syntax_rate=1 / 300,
    )
    manifest = _write_manifest(tmp_path, artifact)
    acceptance_path = tmp_path / "acceptances.json"
    acceptance_path.write_text("{", encoding="utf-8")
    report_path = tmp_path / "monitor.json"
    block_path = tmp_path / "synthesis.blocked"

    failed = monitor.poll_once(
        repo=tmp_path,
        manifest_path=manifest,
        report_path=report_path,
        block_path=block_path,
        acceptance_path=acceptance_path,
        expected_rows=1,
        pool_pid=None,
    )

    assert failed["synthesis_blocked"] is True
    assert "monitor_error" in failed
    assert block_path.is_file()

    acceptance_path.write_text(
        json.dumps({"acceptances": []}),
        encoding="utf-8",
    )
    recovered = monitor.poll_once(
        repo=tmp_path,
        manifest_path=manifest,
        report_path=report_path,
        block_path=block_path,
        acceptance_path=acceptance_path,
        expected_rows=1,
        pool_pid=None,
    )

    assert "monitor_error" not in recovered
    assert recovered["baseline_review_complete"] is True


def test_malformed_manifest_fails_closed_and_next_poll_recovers(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / REPAIR_ROOT / "itergen.json"
    _write_artifact(
        artifact,
        answers=[f"answer-{index}" for index in range(300)],
        accuracy=1 / 300,
        syntax_rate=1 / 300,
    )
    manifest = _write_manifest(tmp_path, artifact)
    valid_manifest = manifest.read_text(encoding="utf-8")
    manifest.write_text("{", encoding="utf-8")
    report_path = tmp_path / "monitor.json"
    block_path = tmp_path / "synthesis.blocked"
    acceptance_path = tmp_path / "acceptances.json"

    failed = monitor.poll_once(
        repo=tmp_path,
        manifest_path=manifest,
        report_path=report_path,
        block_path=block_path,
        acceptance_path=acceptance_path,
        expected_rows=1,
        pool_pid=None,
    )

    assert failed["synthesis_blocked"] is True
    assert "monitor_error" in failed
    assert block_path.is_file()

    manifest.write_text(valid_manifest, encoding="utf-8")
    recovered = monitor.poll_once(
        repo=tmp_path,
        manifest_path=manifest,
        report_path=report_path,
        block_path=block_path,
        acceptance_path=acceptance_path,
        expected_rows=1,
        pool_pid=None,
    )

    assert "monitor_error" not in recovered
    assert recovered["baseline_review_complete"] is True


def test_frozen_source_hash_and_scores_use_the_same_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / REPAIR_ROOT / "itergen.json"
    _write_artifact(
        artifact,
        answers=[f"answer-{index}" for index in range(300)],
        accuracy=1 / 300,
        syntax_rate=1 / 300,
    )
    manifest = _write_manifest(tmp_path, artifact)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    source = tmp_path / payload["rows"][0]["source_artifact"]
    original_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    real_reader = monitor._read_frozen_source_scores

    def mutate_after_read(raw, path, total):
        _write_artifact(
            path,
            answers=[f"changed-{index}" for index in range(300)],
            accuracy=1 / 300,
            syntax_rate=1 / 300,
        )
        return real_reader(raw, path, total)

    monkeypatch.setattr(monitor, "_read_frozen_source_scores", mutate_after_read)
    report = review_manifest(tmp_path, manifest, zero_acceptances={})

    assert report["rows"][0]["status"] == "accepted"
    assert hashlib.sha256(source.read_bytes()).hexdigest() != original_digest
