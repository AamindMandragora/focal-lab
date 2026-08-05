import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.runtime.incident_repair import finalize_exact_zero_campaign as finalizer


def _base_manifest() -> dict:
    path = Path("saved-results/2026-08-03-full-baseline-cold-manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _recovery_snapshot() -> dict:
    return {
        "cells": [
            {
                "cell_id": "gsm-qwen35-2b",
                "completed_evaluations": 3,
                "started_attempts_charged": 4,
                "remaining_attempt_cap": 36,
            },
            {
                "cell_id": "gsm-qwen35-4b",
                "completed_evaluations": 0,
                "started_attempts_charged": 1,
                "remaining_attempt_cap": 39,
            },
        ]
    }


def _pinned_csds() -> dict[str, dict[str, str]]:
    return {
        cell: {
            "heldout_csd_path": f"outputs/generated/{cell}/GeneratedCSD.py",
            "heldout_csd_sha256": hashlib.sha256(cell.encode()).hexdigest(),
        }
        for cell in finalizer.HELDOUT_ONLY_CELLS
    }


def test_queue_plan_has_approved_phases_and_exact_new_call_cap() -> None:
    manifest = finalizer.configure_queue(
        _base_manifest(),
        _recovery_snapshot(),
        _pinned_csds(),
        evidence_path=Path(
            "saved-results/2026-08-05-corrected-full-baseline-evidence.json"
        ),
    )

    phase_counts = {
        phase: sum(job["queue_phase"] == phase for job in manifest["jobs"])
        for phase in range(1, 6)
    }
    assert phase_counts == {1: 10, 2: 2, 3: 2, 4: 3, 5: 3}
    assert manifest["approved_author_call_cap"] == 675
    assert manifest["planned_author_calls"] == 675
    assert (
        sum(
            job["max_iterations"]
            for job in manifest["jobs"]
            if job["run_mode"] != "heldout_only"
        )
        == 675
    )
    finalizer.validate_queue_plan(manifest)


def test_queue_plan_preserves_only_remaining_calls_and_full_memory_gates() -> None:
    manifest = finalizer.configure_queue(
        _base_manifest(),
        _recovery_snapshot(),
        _pinned_csds(),
        evidence_path=Path("saved-results/evidence.json"),
    )
    by_cell = {job["cell_id"]: job for job in manifest["jobs"]}

    recovery = by_cell["gsm-qwen35-2b"]
    assert recovery["max_iterations"] == 36
    assert recovery["initial_attempt_offset"] == 4
    assert recovery["initial_completed_evaluations"] == 3
    assert recovery["initial_attempt_history_file"].endswith("gsm-qwen35-2b.json")
    assert by_cell["gsm-qwen35-4b"]["max_iterations"] == 39
    assert "initial_attempt_history_file" not in by_cell["gsm-qwen35-4b"]
    assert all(
        by_cell[cell]["requires_exclusive_gpu"]
        for cell in finalizer.FULL_MEMORY_RETRY_CELLS
    )


def test_queue_plan_rejects_call_cap_drift() -> None:
    manifest = finalizer.configure_queue(
        _base_manifest(),
        _recovery_snapshot(),
        _pinned_csds(),
        evidence_path=Path("saved-results/evidence.json"),
    )
    tampered = copy.deepcopy(manifest)
    changed_job = next(
        job for job in tampered["jobs"] if job["run_mode"] != "heldout_only"
    )
    changed_job["max_iterations"] += 1

    with pytest.raises(finalizer.FinalizationError, match="675"):
        finalizer.validate_queue_plan(tampered)


def test_v8_replacement_has_priority_over_v7(tmp_path: Path) -> None:
    suffix = Path("spider/qwen35-2b/itergen.json")
    v7 = tmp_path / finalizer.V7_ROOT / suffix
    v8 = tmp_path / finalizer.V8_ROOT / suffix
    v7.parent.mkdir(parents=True)
    v8.parent.mkdir(parents=True)
    v7.write_text("v7", encoding="utf-8")
    v8.write_text("v8", encoding="utf-8")

    selected = finalizer.select_replacement_path(
        tmp_path, "spider-qwen35-2b-itergen", suffix
    )

    assert selected == v8


def test_pinned_csd_resolution_uses_the_frozen_launch_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "git_commit": "a" * 40,
                "jobs": [
                    {
                        "cell_id": cell,
                        "output_name": f"old-{cell}",
                        "git_commit": "a" * 40,
                    }
                    for cell in finalizer.HELDOUT_ONLY_CELLS
                ],
            }
        ),
        encoding="utf-8",
    )
    generated = tmp_path / "compiled" / "GeneratedCSD.py"
    generated.parent.mkdir()
    generated.write_text("# accepted\n", encoding="utf-8")
    observed_launch_commits: list[str] = []

    def compiled_csd(_repo, _output_name, *, job, **_kwargs):
        observed_launch_commits.append(job["launch_commit"])
        return generated

    monkeypatch.setattr(
        finalizer, "SOURCE_MANIFEST", source_manifest.relative_to(tmp_path)
    )
    monkeypatch.setattr(finalizer.queue, "compiled_csd", compiled_csd)
    corrected_jobs = [
        {
            "cell_id": cell,
            "min_accuracy": 0.1,
            "min_syntax_rate": 0.9,
        }
        for cell in finalizer.HELDOUT_ONLY_CELLS
    ]

    finalizer._resolve_pinned_csds(
        tmp_path,
        corrected_jobs,
        launch_commit="b" * 40,
    )

    assert observed_launch_commits == ["b" * 40] * 3


def test_launch_approval_is_bound_to_the_manifest_and_evidence_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence.json"
    source = tmp_path / "source.json"
    recovery = tmp_path / "recovery.json"
    evidence.write_text('{"cells": {}}\n', encoding="utf-8")
    source.write_text('{"jobs": []}\n', encoding="utf-8")
    recovery.write_text('{"cells": []}\n', encoding="utf-8")
    monkeypatch.setattr(finalizer, "CORRECTED_EVIDENCE", Path("evidence.json"))
    monkeypatch.setattr(finalizer, "SOURCE_MANIFEST", Path("source.json"))
    monkeypatch.setattr(finalizer, "RECOVERY_SNAPSHOT", Path("recovery.json"))
    monkeypatch.setattr(finalizer, "validate_queue_plan", lambda _manifest: None)
    monkeypatch.setattr(finalizer.queue, "pinned_heldout_csd", lambda _job, _repo: None)

    manifest = {
        "campaign": "full-baseline-corrected-20260805",
        "git_commit": "a" * 40,
        "corrected_evidence_path": "evidence.json",
        "corrected_evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "recovery_snapshot_sha256": hashlib.sha256(recovery.read_bytes()).hexdigest(),
        "jobs": [
            {
                "cell_id": "heldout",
                "baseline_source": "evidence.json",
                "run_mode": "heldout_only",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    approval = {
        "decision": "approved",
        "reviewer_model": "gpt-5.6-sol",
        "git_commit": manifest["git_commit"],
        "corrected_evidence_sha256": manifest["corrected_evidence_sha256"],
        "queue_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval) + "\n", encoding="utf-8")

    validated = finalizer.validate_launch_approval(
        tmp_path,
        manifest_path,
        approval_path,
    )

    assert validated["campaign"] == "full-baseline-corrected-20260805"
    evidence.write_text('{"cells": {"tampered": {}}}\n', encoding="utf-8")
    with pytest.raises(finalizer.FinalizationError, match="evidence hash"):
        finalizer.validate_launch_approval(
            tmp_path,
            manifest_path,
            approval_path,
        )
