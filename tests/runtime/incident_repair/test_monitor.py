import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import scripts.runtime.incident_repair.monitor as monitor
from scripts.runtime.incident_repair.monitor import (
    Incident,
    RepairResult,
    build_repair_command,
    build_repair_prompt,
    changed_files,
    detect_incidents,
    is_allowed_change,
    is_protected_path,
    load_monitor_jobs,
    parse_repair_result,
    read_new_text,
    repair_can_relaunch,
    repair_environment,
    result_matches_changes,
    should_escalate_incident,
    verify_repair_account,
    write_repair_attestation,
)


def test_detects_catastrophic_errors_but_ignores_handled_rate_limits():
    text = """
HTTP 429: daily quota exceeded; retrying in 3600 seconds
Task: Unknown task
ValueError: Free memory on device cuda:0 is less than desired
"""

    incidents = detect_incidents(Path("paid_synth_worker.log"), text)

    assert [incident.rule for incident in incidents] == [
        "unknown_task",
        "gpu_memory_startup",
    ]


def test_detects_real_queue_release_and_claude_timeout_lines():
    incidents = detect_incidents(
        Path("worker.log"),
        "[warm-recovery] release cell=gsm-qwen35-2b gpu=1 status=76\n"
        "[claude] timeout request_sha256=abc retry=0 duration_seconds=900\n",
    )

    assert [incident.rule for incident in incidents] == ["claude_timeout"]


def test_detects_cold_queue_worker_failures_but_not_completed_losses():
    incidents = detect_incidents(
        Path("cold-controller.log"),
        "[coldq] release cell=bad gpu=0 status=3\n"
        "[coldq] release cell=loss gpu=1 status=75\n"
        "[coldq] release cell=good gpu=2 status=0\n",
    )

    assert [incident.rule for incident in incidents] == ["worker_failed"]


def test_monitor_loads_cold_manifest_and_its_log_paths(tmp_path):
    run_log = tmp_path / "outputs" / "generated" / "coldq_test" / "run.log"
    run_log.parent.mkdir(parents=True)
    run_log.write_text("ready\n")
    manifest = tmp_path / "manifest.json"
    job = {
        "cell_id": "test-cell",
        "task": "Solve math.",
        "dataset": "gsm_symbolic",
        "eval_model": "Qwen/Qwen3.5-2B",
        "max_iterations": 40,
        "eval_sample_size": 49,
        "min_accuracy": 0.25,
        "min_syntax_rate": 0.9,
        "eval_max_steps": 900,
        "eval_max_seconds": 600,
        "memory_reservation_mib": 16000,
        "gpu_mem_util": 0.4,
        "output_name": "coldq_test",
        "heldout_sample_size": 49,
        "heldout_split_name": "test",
        "heldout_output_json": str(tmp_path / "heldout.json"),
        "log_file": str(run_log),
    }
    manifest.write_text(json.dumps({"git_commit": "a" * 40, "jobs": [job]}))

    jobs = load_monitor_jobs(manifest)

    assert jobs == [job]
    assert monitor._log_paths(manifest, [], None) == [run_log]


def test_one_timeout_is_counted_once_even_with_related_failure_lines():
    incidents = detect_incidents(
        Path("worker.log"),
        "[claude] failure exit_status=timeout category=idle-timeout duration_seconds=900\n"
        "[claude] timeout request_sha256=abc retry=0 duration_seconds=900\n"
        "Temporary Claude provider failure: Claude Code stream idle-timeout\n",
    )

    assert [incident.rule for incident in incidents] == ["claude_timeout"]


def test_claude_timeout_escalates_only_after_three_events_in_ten_minutes():
    state = {}
    incident = Incident("claude_timeout", "worker.log", "timeout", "one")

    assert not should_escalate_incident(incident, state, now=100.0)
    assert not should_escalate_incident(
        Incident("claude_timeout", "worker.log", "timeout", "two"), state, now=200.0
    )
    assert should_escalate_incident(
        Incident("claude_timeout", "worker.log", "timeout", "three"), state, now=300.0
    )
    assert not should_escalate_incident(
        Incident("claude_timeout", "worker.log", "timeout", "late"), state, now=1_000.0
    )


def test_new_log_bootstraps_at_end_then_reads_only_appended_text(tmp_path):
    log = tmp_path / "worker.log"
    log.write_text("historical Unknown task\n")

    first_text, cursor = read_new_text(log, None, bootstrap_at_end=True)
    assert first_text == ""

    with log.open("a") as handle:
        handle.write("new catastrophic line\n")

    new_text, next_cursor = read_new_text(log, cursor, bootstrap_at_end=True)
    assert new_text == "new catastrophic line\n"
    assert next_cursor["offset"] > cursor["offset"]


def test_relaunch_requires_agent_success_external_tests_and_unchanged_protected_files():
    repaired = RepairResult(
        status="repaired",
        summary="fixed",
        files_changed=["synthesis/evaluate/feedback_loop.py"],
        tests=["pytest"],
        safe_to_relaunch=True,
    )

    assert repair_can_relaunch(repaired, agent_exit=0, verifier_exit=0, protected_changed=False)
    assert not repair_can_relaunch(repaired, agent_exit=1, verifier_exit=0, protected_changed=False)
    assert not repair_can_relaunch(repaired, agent_exit=0, verifier_exit=1, protected_changed=False)
    assert not repair_can_relaunch(repaired, agent_exit=0, verifier_exit=0, protected_changed=True)


def test_repair_prompt_contains_required_autonomy_boundaries(tmp_path):
    incident = Incident(
        rule="unknown_task",
        source="paid_synth_worker.log",
        line="Task: Unknown task",
        fingerprint="abc123",
    )
    evidence = tmp_path / "incident.json"
    evidence.write_text(json.dumps({"rule": incident.rule}))

    prompt = build_repair_prompt(incident, evidence)

    assert "reproduce the defect with a failing test" in prompt
    assert "Do not launch synthesis" in prompt
    assert "Do not call Bedrock" in prompt
    assert "grammars" in prompt
    assert "graders" in prompt
    assert "dataset splits" in prompt
    assert "warm/cold policy" in prompt
    assert str(evidence) in prompt


def test_repair_command_uses_claude_headless_with_structured_output(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text('{"type": "object"}')
    incident_dir = tmp_path / "incident"

    command = build_repair_command(
        Path("/fake/claude"), "claude-sonnet-4-6", schema, incident_dir
    )

    assert command[0] == "/fake/claude"
    assert "-p" in command
    assert command[command.index("--model") + 1] == "claude-sonnet-4-6"
    assert command[command.index("--output-format") + 1] == "json"
    assert command[command.index("--json-schema") + 1] == '{"type": "object"}'
    assert command[command.index("--permission-mode") + 1] == "acceptEdits"
    assert command[command.index("--allowedTools") + 1] == "Bash(*)"
    assert command[command.index("--add-dir") + 1] == str(incident_dir)
    assert "--dangerously-skip-permissions" not in command
    assert not any("codex" in part for part in command)


def test_parse_repair_result_reads_structured_output_envelope():
    payload = {
        "status": "repaired",
        "summary": "fixed",
        "files_changed": ["synthesis/task_context.py"],
        "tests": ["pytest -q tests/test_task.py"],
        "safe_to_relaunch": True,
    }
    envelope = json.dumps(
        {"type": "result", "subtype": "success", "is_error": False, "structured_output": payload}
    )

    result = parse_repair_result(envelope)

    assert result == RepairResult(**payload)


def test_parse_repair_result_rejects_errors_and_missing_structured_output():
    import pytest

    with pytest.raises(ValueError):
        parse_repair_result(
            json.dumps({"type": "result", "subtype": "error_during_execution", "is_error": True})
        )
    with pytest.raises(ValueError):
        parse_repair_result(json.dumps({"type": "result", "subtype": "success", "is_error": False}))
    with pytest.raises(ValueError):
        parse_repair_result("not json at all")


def test_verify_repair_account_rejects_wrong_or_non_max_accounts(monkeypatch):
    import pytest

    def fake_status(command, **_kwargs):
        assert command[1:] == ["auth", "status", "--json"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "loggedIn": True,
                    "email": "ssdear@gmail.com",
                    "authMethod": "claude.ai",
                    "apiProvider": "firstParty",
                    "subscriptionType": "max",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(monitor.subprocess, "run", fake_status)
    with pytest.raises(ValueError):
        verify_repair_account(Path("/fake/claude"), Path("/fake/config"), "aadivya@fermi.ai")


def test_verify_repair_account_accepts_the_expected_max_account(monkeypatch):
    def fake_status(command, *, env, **_kwargs):
        assert env["CLAUDE_CONFIG_DIR"] == "/fake/config"
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "loggedIn": True,
                    "email": "aadivya@fermi.ai",
                    "authMethod": "claude.ai",
                    "apiProvider": "firstParty",
                    "subscriptionType": "max",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(monitor.subprocess, "run", fake_status)
    verify_repair_account(Path("/fake/claude"), Path("/fake/config"), "aadivya@fermi.ai")


def test_protected_and_allowed_paths_are_checked_from_the_real_relative_path():
    assert is_protected_path(Path("grammars/gsm.lark"))
    assert is_protected_path(Path("synthesis/evaluate/graders/gsm.py"))
    assert is_protected_path(Path("environment/benchmark_splits/gsm.json"))
    assert is_protected_path(Path("results_matrix.md"))
    assert not is_protected_path(Path("synthesis/evaluate/feedback_loop.py"))

    assert is_allowed_change(Path("synthesis/evaluate/feedback_loop.py"))
    assert is_allowed_change(Path("tests/test_feedback_loop.py"))
    assert not is_allowed_change(Path("README.md"))
    assert not is_allowed_change(Path("grammars/gsm.lark"))
    assert not is_allowed_change(Path("synthesis/verify/library/strategy.dfy"))
    assert not is_allowed_change(Path("synthesis/evaluate/data/split.json"))


def test_changed_files_reports_edits_additions_and_deletions(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    (before / "edited.py").write_text("old")
    (after / "edited.py").write_text("new")
    (before / "deleted.py").write_text("gone")
    (after / "added.py").write_text("new")

    changes = changed_files(before, after)

    assert changes == {
        Path("added.py"): "added",
        Path("deleted.py"): "deleted",
        Path("edited.py"): "modified",
    }


def test_repair_environment_removes_credentials_and_pins_the_isolated_config_dir():
    clean = repair_environment(
        {
            "PATH": "/bin",
            "AWS_BEARER_TOKEN_BEDROCK": "secret",
            "AWS_REGION": "us-east-1",
            "OPENAI_API_KEY": "secret",
            "SAFE_VALUE": "keep",
        },
        Path("/fake/config"),
    )

    assert clean == {
        "PATH": "/bin",
        "SAFE_VALUE": "keep",
        "CLAUDE_CONFIG_DIR": "/fake/config",
    }


def test_snapshot_never_copies_nested_credentials(tmp_path):
    repo = tmp_path / "repo"
    (repo / "synthesis" / "nested").mkdir(parents=True)
    (repo / "synthesis" / "safe.py").write_text("SAFE = True\n")
    (repo / "synthesis" / "nested" / ".env").write_text("TOKEN=secret\n")
    (repo / "synthesis" / "nested" / "secrets.json").write_text("{}")
    snapshot = tmp_path / "snapshot"

    monitor._copy_snapshot(repo, snapshot)

    assert (snapshot / "synthesis" / "safe.py").is_file()
    assert not (snapshot / "synthesis" / "nested" / ".env").exists()
    assert not (snapshot / "synthesis" / "nested" / "secrets.json").exists()


def test_repair_result_must_name_the_actual_changes_and_tests():
    result = RepairResult(
        status="repaired",
        summary="fixed",
        files_changed=["synthesis/task_context.py"],
        tests=["pytest -q tests/test_task_context.py"],
        safe_to_relaunch=True,
    )
    changes = {Path("synthesis/task_context.py"): "modified"}

    assert result_matches_changes(result, changes)
    assert not result_matches_changes(
        RepairResult(**{**result.__dict__, "files_changed": []}), changes
    )
    assert not result_matches_changes(
        RepairResult(**{**result.__dict__, "tests": []}), changes
    )


def test_incident_service_is_separate_and_visible_in_combined_paid_logs():
    unit = (
        Path(__file__).parents[3]
        / "deploy"
        / "focal"
        / "systemd"
        / "csd-codex-incident-monitor.service"
    ).read_text()

    assert "incident_repair/monitor.py" in unit
    assert "Restart=always" in unit
    assert "--claude-executable /home/aadivyar/.local/bin/claude" in unit
    assert "--claude-model claude-sonnet-4-6" in unit
    assert "--claude-config-dir /home/aadivyar/.claude-csd-synthesis" in unit
    assert "--claude-expected-account aadivya@fermi.ai" in unit
    assert "--codex" not in unit
    assert "codex-cli" not in unit
    assert "paid_synth_codex_incident_monitor.log" in unit
    assert "2026-07-19-exhaustive-cold-queue-manifest.json" in unit
    assert "outputs/generated/coldq_*/run.log" in unit
    assert "--recovery-service csd-cold-synthesis-queue.service" in unit
    assert "--repair-attestation" in unit
    assert "--recovery-service csd-gsm14b-claude-durable.service" not in unit
    assert "AWS_BEARER_TOKEN_BEDROCK" not in unit


def test_result_schema_is_flat_for_structured_output_validation():
    schema = json.loads(
        (
            Path(__file__).parents[3]
            / "scripts"
            / "runtime"
            / "incident_repair"
            / "result.schema.json"
        ).read_text()
    )

    assert "allOf" not in schema
    assert "if" not in json.dumps(schema)
    assert set(schema["required"]) == {
        "status",
        "summary",
        "files_changed",
        "tests",
        "safe_to_relaunch",
    }


def _repair_args(tmp_path, repo, manifest, schema):
    return SimpleNamespace(
        repo=repo,
        manifest=manifest,
        state_dir=tmp_path / "state",
        claude_executable=Path("/fake/claude"),
        claude_model="claude-sonnet-4-6",
        claude_config_dir=Path("/fake/config"),
        claude_expected_account="aadivya@fermi.ai",
        claude_timeout_seconds=60.0,
        python=Path("/fake/python"),
        result_schema=schema,
        repair_attestation=tmp_path / "approved-repair.json",
    )


def _fake_repair_envelope(payload):
    return json.dumps(
        {"type": "result", "subtype": "success", "is_error": False, "structured_output": payload}
    )


def test_repair_attestation_records_exact_verified_live_file_hashes(tmp_path):
    repo = tmp_path / "repo"
    changed = repo / "synthesis" / "task_context.py"
    changed.parent.mkdir(parents=True)
    changed.write_text("TASK = 'gsm_symbolic'\n")
    target = tmp_path / "approved-repair.json"

    write_repair_attestation(
        target,
        repo=repo,
        base_commit="a" * 40,
        changes={Path("synthesis/task_context.py"): "modified"},
        incident_fingerprint="abc123",
    )

    payload = json.loads(target.read_text())
    assert payload["base_commit"] == "a" * 40
    assert payload["verifier_exit"] == 0
    assert payload["incident_fingerprint"] == "abc123"
    assert payload["files"] == {
        "synthesis/task_context.py": hashlib.sha256(changed.read_bytes()).hexdigest()
    }


def test_fake_unknown_task_is_repaired_verified_deployed_and_relaunched(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    source = repo / "synthesis" / "task_context.py"
    source.parent.mkdir(parents=True)
    source.write_text("TASK = 'Unknown task'\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_task.py").write_text("def test_placeholder(): pass\n")
    manifest = repo / "manifest.json"
    manifest.write_text(json.dumps({"git_commit": "a" * 40, "jobs": []}))
    evidence = tmp_path / "incident.json"
    evidence.write_text("{}")
    schema = tmp_path / "schema.json"
    schema.write_text("{}")
    commands = []

    def fake_capture(_repo, _incident, incident_dir):
        incident_dir.mkdir(parents=True)
        return evidence

    monkeypatch.setattr(monitor, "_capture_evidence", fake_capture)
    monkeypatch.setattr(monitor, "_stop_recovery", lambda *_: commands.append("stopped"))
    monkeypatch.setattr(monitor, "_verify", lambda *_: 0)
    monkeypatch.setattr(monitor, "verify_repair_account", lambda *_: None)
    monkeypatch.setattr(
        monitor,
        "_run",
        lambda command, **_: commands.append(command) or 0,
    )

    def fake_claude(command, *, cwd, env, input, text, capture_output, timeout):
        assert "AWS_BEARER_TOKEN_BEDROCK" not in env
        assert env["CLAUDE_CONFIG_DIR"] == "/fake/config"
        assert command[command.index("--permission-mode") + 1] == "acceptEdits"
        assert capture_output and text
        assert timeout == 60.0
        assert "Do not launch synthesis" in input
        (cwd / "synthesis" / "task_context.py").write_text("TASK = 'gsm_symbolic'\n")
        return SimpleNamespace(
            returncode=0,
            stdout=_fake_repair_envelope(
                {
                    "status": "repaired",
                    "summary": "restored task context",
                    "files_changed": ["synthesis/task_context.py"],
                    "tests": ["pytest -q tests/test_task.py"],
                    "safe_to_relaunch": True,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(monitor.subprocess, "run", fake_claude)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "must-not-reach-the-repair-agent")
    args = _repair_args(tmp_path, repo, manifest, schema)
    incident = Incident("unknown_task", "worker.log", "Task: Unknown task", "abc123")

    assert monitor.repair_incident(args, incident)
    assert source.read_text() == "TASK = 'gsm_symbolic'\n"
    assert commands[0] == "stopped"
    assert ["systemctl", "--user", "start", "csd-warm-recovery.service"] in commands
    assert commands[-1] == [
        "systemctl", "--user", "is-active", "--quiet", "csd-warm-recovery.service"
    ]
    attestation = json.loads(args.repair_attestation.read_text())
    assert set(attestation["files"]) == {"synthesis/task_context.py"}


def test_failed_relaunch_rolls_back_the_deployed_repair(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    source = repo / "synthesis" / "task_context.py"
    source.parent.mkdir(parents=True)
    source.write_text("ORIGINAL = True\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_task.py").write_text("def test_placeholder(): pass\n")
    manifest = repo / "manifest.json"
    manifest.write_text(json.dumps({"git_commit": "a" * 40, "jobs": []}))
    evidence = tmp_path / "incident.json"
    evidence.write_text("{}")
    schema = tmp_path / "schema.json"
    schema.write_text("{}")

    def fake_capture(_repo, _incident, incident_dir):
        incident_dir.mkdir(parents=True)
        return evidence

    def fake_claude(command, *, cwd, env, input, text, capture_output, timeout):
        (cwd / "synthesis" / "task_context.py").write_text("REPAIRED = True\n")
        return SimpleNamespace(
            returncode=0,
            stdout=_fake_repair_envelope(
                {
                    "status": "repaired",
                    "summary": "fixed",
                    "files_changed": ["synthesis/task_context.py"],
                    "tests": ["pytest -q tests/test_task.py"],
                    "safe_to_relaunch": True,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(monitor, "_capture_evidence", fake_capture)
    monkeypatch.setattr(monitor, "_stop_recovery", lambda *_: None)
    monkeypatch.setattr(monitor, "_verify", lambda *_: 0)
    monkeypatch.setattr(monitor, "verify_repair_account", lambda *_: None)
    monkeypatch.setattr(monitor.subprocess, "run", fake_claude)
    monkeypatch.setattr(monitor, "_run", lambda *_args, **_kwargs: 1)
    args = _repair_args(tmp_path, repo, manifest, schema)

    assert not monitor.repair_incident(
        args, Incident("unknown_task", "worker.log", "Unknown task", "rollback")
    )
    assert source.read_text() == "ORIGINAL = True\n"
    assert not args.repair_attestation.exists()


def test_rejected_repair_restarts_the_original_stopped_service(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    source = repo / "synthesis" / "task_context.py"
    source.parent.mkdir(parents=True)
    source.write_text("ORIGINAL = True\n")
    manifest = repo / "manifest.json"
    manifest.write_text(json.dumps({"git_commit": "a" * 40, "jobs": []}))
    schema = tmp_path / "schema.json"
    schema.write_text("{}")
    commands = []

    def fake_capture(_repo, _incident, incident_dir):
        incident_dir.mkdir(parents=True)
        evidence = incident_dir / "incident.json"
        evidence.write_text("{}")
        return evidence

    monkeypatch.setattr(monitor, "_capture_evidence", fake_capture)
    monkeypatch.setattr(monitor, "_stop_recovery", lambda *_: commands.append("stopped"))
    monkeypatch.setattr(monitor, "verify_repair_account", lambda *_: None)
    monkeypatch.setattr(monitor, "_run", lambda command, **_: commands.append(command) or 0)
    monkeypatch.setattr(
        monitor.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="not-json", stderr=""),
    )
    args = _repair_args(tmp_path, repo, manifest, schema)

    assert not monitor.repair_incident(
        args, Incident("unknown_task", "worker.log", "Unknown task", "rejected")
    )
    assert commands[0] == "stopped"
    assert ["systemctl", "--user", "start", "csd-warm-recovery.service"] in commands


def test_blocked_repair_is_retried_when_the_same_incident_returns(tmp_path, monkeypatch):
    calls = []
    incident = Incident("worker_failed", "worker.log", "failed", "retry-me")
    args = SimpleNamespace(recovery_service=["csd-cold-synthesis-queue.service"])
    seen = set()
    timeout_state = {}
    monkeypatch.setattr(monitor, "repair_incident", lambda *_: calls.append(1) or False)

    monitor._process_incidents(args, [incident], seen, timeout_state)
    monitor._process_incidents(args, [incident], seen, timeout_state)

    assert calls == [1, 1]
    assert incident.fingerprint not in seen


def test_monitor_state_recovers_from_truncated_json_and_replaces_atomically(
    tmp_path, monkeypatch
):
    state = tmp_path / "seen.json"
    state.write_text("{", encoding="utf-8")
    assert monitor._load_json_state(state, []) == []
    state.write_text("[]", encoding="utf-8")
    assert monitor._load_json_state(state, {}) == {}

    state.write_text('["old"]\n', encoding="utf-8")
    original_dump = monitor.json.dump

    def interrupted_dump(payload, handle, **kwargs):
        handle.write("[")
        raise OSError("interrupted")

    monkeypatch.setattr(monitor.json, "dump", interrupted_dump)
    try:
        monitor._write_json_atomic(state, ["new"])
    except OSError:
        pass
    else:
        raise AssertionError("the simulated interrupted write must fail")
    assert json.loads(state.read_text()) == ["old"]
    monkeypatch.setattr(monitor.json, "dump", original_dump)
    monitor._write_json_atomic(state, ["new"])
    assert json.loads(state.read_text()) == ["new"]


def test_stop_recovery_requires_the_service_to_be_inactive(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"jobs": []}', encoding="utf-8")
    monkeypatch.setattr(monitor, "load_monitor_jobs", lambda *_: [])
    monkeypatch.setattr(monitor, "recovery_processes", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        monitor,
        "_run",
        lambda command, **_: 0,
    )

    try:
        monitor._stop_recovery(
            tmp_path, manifest, ["csd-cold-synthesis-queue.service"]
        )
    except RuntimeError as error:
        assert "still active" in str(error)
    else:
        raise AssertionError("an active service must block live code replacement")
