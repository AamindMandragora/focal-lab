from pathlib import Path
from types import SimpleNamespace

import json
import hashlib
import subprocess
import sys

from scripts.runtime import run_cold_synthesis_queue as queue


EXPECTED_CRANE_GSM_TASK = (
    "You are an expert in solving grade school math tasks. "
    "You will be presented with a grade-school math word problem with symbolic variables and be asked to solve it.\n\n"
    "Before answering you should reason about the problem (using the <reasoning> field in the response described below). "
    "Intermediate symbolic expressions generated during reasoning should be wrapped in << >>.\n\n"
    "Then, output the symbolic expression wrapped in << >> that answers the question. "
    "The expressions must use numbers as well as the variables defined in the question. "
    "You are only allowed to use the following operations: +, -, /, //, %, (), and int().\n\n"
    "You will always respond in the format described below: \n"
    "Let's think step by step. <reasoning> The final answer is <<symbolic expression>>"
)


def _job(dataset="gsm_symbolic"):
    return {
        "cell_id": "gsm-qwen35-2b",
        "task": "Solve symbolic math.",
        "dataset": dataset,
        "eval_model": "Qwen/Qwen3.5-2B",
        "max_iterations": 80,
        "eval_sample_size": 49,
        "min_accuracy": 0.265306122449,
        "min_syntax_rate": 0.9,
        "eval_max_steps": 900,
        "eval_max_seconds": 600,
        "memory_reservation_mib": 16384,
        "gpu_mem_util": 0.8,
        "output_name": "coldq_gsm-qwen35-2b_0719",
        "heldout_sample_size": 49,
        "heldout_split_file": "/repo/split.json",
        "heldout_split_name": "test",
        "heldout_output_json": "/repo/heldout.json",
    }


def _matching_run_configuration(job):
    prefix = "gsm" if job["dataset"] == "gsm_symbolic" else "spider"
    split = {
        "gsm_split_file": None,
        "gsm_split_name": None,
        "spider_split_file": None,
        "spider_split_name": None,
        "bar_split_name": "train",
    }
    if job["dataset"] in {"gsm_symbolic", "spider"}:
        split[f"{prefix}_split_file"] = str(
            Path("/repo")
            / (
                "gsm_symbolic_crane_proportional_49x49_seed123.json"
                if prefix == "gsm"
                else "spider_dev_proportional_300x300_seed334.json"
            )
        )
        split[f"{prefix}_split_name"] = "train"
    return {
        "task_description": job["task"],
        "output_name": job["output_name"],
        "git_commit": job.get("launch_commit", job["git_commit"]),
        "max_iterations": job["max_iterations"],
        "thresholds": {
            "min_accuracy": job["min_accuracy"],
            "min_syntax_rate": job["min_syntax_rate"],
        },
        "author_model": {
            "backend": "claude-bedrock",
            "model": "us.anthropic.claude-sonnet-4-6",
            "max_new_tokens": 8192,
            "reasoning_budget_tokens": 4096,
            "anthropic_thinking": "always-on",
            "anthropic_effort": "high",
            "anthropic_thinking_display": "summarized",
        },
        "evaluation": {
            "dataset": job["dataset"],
            "eval_model": job["eval_model"],
            "eval_sample_size": job["eval_sample_size"],
            "eval_max_steps": job["eval_max_steps"],
            "eval_step_token_budget": 1,
            "eval_max_seconds_per_example": job["eval_max_seconds"],
            "smiles_classes": (
                [job["smiles_class"]] if job.get("smiles_class") else None
            ),
            "split_provenance": split,
        },
    }


def test_gsm_task_matches_crane_task_and_cot_contract() -> None:
    assert queue.GSM_TASK == EXPECTED_CRANE_GSM_TASK
    assert {
        config["task"]
        for config in queue.EXPECTED_CELLS.values()
        if config["dataset"] == "gsm_symbolic"
    } == {EXPECTED_CRANE_GSM_TASK}


def test_synthesis_command_is_cold_and_uses_large_bedrock_author():
    command = queue.synthesis_command(_job(), Path("/env/python"))

    assert command[:3] == ["/env/python", "-m", "synthesis.run_synthesis"]
    assert command[command.index("--generation-backend") + 1] == "claude-bedrock"
    assert command[command.index("--generation-model") + 1] == (
        "us.anthropic.claude-sonnet-4-6"
    )
    assert command[command.index("--synthesizer-reasoning-budget") + 1] == "4096"
    assert command[command.index("--max-iterations") + 1] == "80"
    assert "--bar-split-name" not in command
    assert not any(flag.startswith("--initial-") for flag in command)


def test_synthesis_command_only_uses_run_synthesis_cli_flags():
    command = queue.synthesis_command(_job(), Path(sys.executable))
    help_result = subprocess.run(
        [sys.executable, "-m", "synthesis.run_synthesis", "--help"],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=True,
    )

    unsupported = [
        argument
        for argument in command
        if argument.startswith("--") and argument not in help_result.stdout
    ]

    assert unsupported == []


def test_synthesis_environment_names_the_isolated_cold_output():
    env = queue.synthesis_environment(_job(), 2, {"PATH": "/bin"}, Path("/repo"))

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["CSD_OUTPUT_NAME"] == "coldq_gsm-qwen35-2b_0719"
    assert env["CSD_OUTPUT_DIR"] == "/repo/outputs/generated/coldq_gsm-qwen35-2b_0719"


def test_heldout_command_binds_result_to_cell_commit_and_strategy():
    job = _job()
    job["git_commit"] = "a" * 40
    csd = Path("/repo/GeneratedCSD.py")

    command = queue.heldout_command(job, Path("/env/python"), csd)

    assert command[command.index("--provenance-cell-id") + 1] == job["cell_id"]
    assert command[command.index("--provenance-manifest-commit") + 1] == job["git_commit"]


def test_heldout_environment_removes_paid_author_credentials():
    env = queue.author_free_environment(
        {"PATH": "/bin", "AWS_BEARER_TOKEN_BEDROCK": "secret", "OPENAI_API_KEY": "secret"},
        1,
    )

    assert env == {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "1"}


def test_dispatch_preserves_manifest_priority_on_one_gpu():
    first = _job()
    first["cell_id"] = "first"
    second = _job()
    second["cell_id"] = "second"
    started = []

    def worker(job, gpu):
        started.append((job["cell_id"], gpu))
        return 0

    queue.dispatch(
        [first, second],
        snapshot=lambda: {0: {"used_mib": 0, "total_mib": 48_000}},
        worker=worker,
        poll_seconds=0.001,
    )

    assert started == [("first", 0), ("second", 0)]


def test_synthesis_reservation_uses_the_fixed_eighty_percent_runtime_setting():
    job = _job()
    job["gpu_mem_util"] = 0.4

    assert queue.synthesis_required_memory_mib(job, 48_000) == 38_400


def test_compiled_csd_uses_best_threshold_candidate_after_exhaustion(tmp_path):
    output_name = "coldq_gsm-qwen35-2b_0719"
    run_dir = tmp_path / "outputs" / "generated" / output_name / "failed-run"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    (tmp_path / "outputs" / "generated" / output_name / "latest_run.txt").write_text(
        str(run_dir), encoding="utf-8"
    )
    weaker = run_dir / "compiled-weaker"
    closest = run_dir / "compiled-closest"
    weaker.mkdir()
    closest.mkdir()
    (weaker / "GeneratedCSD.py").write_text("# weaker\n", encoding="utf-8")
    (closest / "GeneratedCSD.py").write_text("# closest\n", encoding="utf-8")
    job = _job()
    job["git_commit"] = "a" * 40
    report = {
        "total_attempts": job["max_iterations"],
        "run_configuration": _matching_run_configuration(job),
        "attempts": [
            {
                "attempt_number": 1,
                "compilation": {"success": True, "output_dir": str(weaker)},
                "evaluation": {"accuracy": 0.6, "syntax_rate": 0.7, "num_examples": 49},
            },
            {
                "attempt_number": 2,
                "compilation": {"success": True, "output_dir": str(closest)},
                "evaluation": {"accuracy": 0.58, "syntax_rate": 0.9, "num_examples": 49},
            },
        ]
    }
    (results_dir / "failure_report.json").write_text(json.dumps(report), encoding="utf-8")

    selected = queue.compiled_csd(
        tmp_path, output_name, min_accuracy=0.65, min_syntax_rate=0.9, job=job
    )

    assert selected == closest / "GeneratedCSD.py"


def test_compiled_csd_treats_a_truncated_report_as_incomplete(tmp_path):
    output_name = "coldq_gsm-qwen35-2b_0719"
    run_dir = tmp_path / "outputs" / "generated" / output_name / "interrupted-run"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    (tmp_path / "outputs" / "generated" / output_name / "latest_run.txt").write_text(
        str(run_dir), encoding="utf-8"
    )
    (results_dir / "success_report.json").write_text("{", encoding="utf-8")

    assert queue.compiled_csd(
        tmp_path,
        output_name,
        min_accuracy=0.65,
        min_syntax_rate=0.9,
        job={**_job(), "git_commit": "a" * 40},
    ) is None


def test_exhaustion_report_must_match_the_exact_cold_invocation():
    job = _job()
    job["git_commit"] = "a" * 40
    job["launch_commit"] = "b" * 40
    report = {
        "total_attempts": job["max_iterations"],
        "run_configuration": _matching_run_configuration(job),
    }

    assert queue.report_matches_job(report, job, require_exhausted=True)

    report["run_configuration"]["evaluation"]["eval_model"] = "wrong-model"
    assert not queue.report_matches_job(report, job, require_exhausted=True)
    report["run_configuration"]["evaluation"]["eval_model"] = job["eval_model"]
    report["total_attempts"] -= 1
    assert not queue.report_matches_job(report, job, require_exhausted=True)

    report["total_attempts"] = job["max_iterations"]
    report["run_configuration"]["author_model"]["reasoning_budget_tokens"] = 2048
    assert not queue.report_matches_job(report, job, require_exhausted=True)

    smiles_job = _job("smiles")
    smiles_job.update(
        {
            "git_commit": "a" * 40,
            "smiles_class": "isocyanates",
            "task": queue.SMILES_TASK,
        }
    )
    smiles_report = {
        "total_attempts": smiles_job["max_iterations"],
        "run_configuration": _matching_run_configuration(smiles_job),
    }
    assert queue.report_matches_job(smiles_report, smiles_job, require_exhausted=True)
    smiles_report["run_configuration"]["evaluation"]["smiles_classes"] = [
        "acrylates"
    ]
    assert not queue.report_matches_job(smiles_report, smiles_job, require_exhausted=True)


def test_exhausted_synthesis_still_runs_heldout_for_best_attempt(tmp_path, monkeypatch):
    job = _job()
    job["git_commit"] = "a" * 40
    job["heldout_output_json"] = str(tmp_path / "heldout.json")
    best_csd = tmp_path / "GeneratedCSD.py"
    best_csd.write_text("# best failed attempt\n", encoding="utf-8")
    selected = iter([None, best_csd])
    calls = []
    monkeypatch.setattr(queue, "compiled_csd", lambda *args, **kwargs: next(selected))
    monkeypatch.setattr(queue, "synthesis_was_exhausted", lambda *args, **kwargs: True)
    monkeypatch.setattr(queue, "heldout_is_complete", lambda *args: len(calls) >= 2)

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    monkeypatch.setattr(queue.subprocess, "run", run)

    status = queue.run_job(
        job,
        0,
        repo=tmp_path,
        python=Path("/env/python"),
        state_dir=tmp_path / "state",
    )

    assert status == 0
    assert calls[1][1:3] == ["-m", "synthesis.scripts.reevaluate_compiled_csd"]
    state = json.loads((tmp_path / "state" / "gsm-qwen35-2b.json").read_text())
    assert state["status"] == "complete_loss"


def test_unexpected_exit_one_is_an_error_not_a_scientific_loss(tmp_path, monkeypatch):
    job = _job()
    job["git_commit"] = "a" * 40
    job["heldout_output_json"] = str(tmp_path / "heldout.json")
    monkeypatch.setattr(queue, "compiled_csd", lambda *args, **kwargs: None)
    monkeypatch.setattr(queue, "synthesis_was_exhausted", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        queue.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )

    status = queue.run_job(
        job,
        0,
        repo=tmp_path,
        python=Path("/env/python"),
        state_dir=tmp_path / "state",
    )

    assert status == 1
    state = json.loads((tmp_path / "state" / "gsm-qwen35-2b.json").read_text())
    assert state["status"] == "error"


def test_existing_heldout_must_be_complete_before_restart_skips_cell(tmp_path, monkeypatch):
    job = _job()
    job["git_commit"] = "a" * 40
    heldout = tmp_path / "heldout.json"
    job["heldout_output_json"] = str(heldout)
    heldout.write_text("{", encoding="utf-8")
    csd = tmp_path / "GeneratedCSD.py"
    csd.write_text("# strategy\n", encoding="utf-8")
    monkeypatch.setattr(queue, "compiled_csd", lambda *args, **kwargs: csd)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        heldout.write_text(
            json.dumps(
                {
                    "accuracy": 0.5,
                    "syntax_rate": 1.0,
                    "metrics": {"num_examples": 49},
                    "answers": [{} for _ in range(49)],
                    "eval_split": {
                        "gsm_split_file": job["heldout_split_file"],
                        "gsm_split_name": "test",
                        "spider_split_file": None,
                        "spider_split_name": None,
                        "bar_split_name": None,
                    },
                    "reevaluation_provenance": {
                        "cell_id": job["cell_id"],
                        "manifest_commit": job["git_commit"],
                        "dataset": job["dataset"],
                        "eval_model": job["eval_model"],
                        "compiled_csd_path": str(csd.resolve()),
                        "compiled_csd_sha256": hashlib.sha256(csd.read_bytes()).hexdigest(),
                        "sample_size": job["heldout_sample_size"],
                        "max_steps": job["eval_max_steps"],
                        "step_token_budget": 1,
                        "smiles_class": None,
                    },
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(queue.subprocess, "run", run)

    assert queue.run_job(
        job, 0, repo=tmp_path, python=Path("/env/python"), state_dir=tmp_path / "state"
    ) == 0
    assert len(calls) == 1
    assert queue.heldout_is_complete(heldout, job)
    payload = json.loads(heldout.read_text())
    payload["reevaluation_provenance"]["eval_model"] = "wrong-model"
    heldout.write_text(json.dumps(payload), encoding="utf-8")
    assert not queue.heldout_is_complete(heldout, job)


def test_restart_preserves_exhausted_state_while_running_heldout(tmp_path, monkeypatch):
    job = _job()
    job["git_commit"] = "a" * 40
    job["heldout_output_json"] = str(tmp_path / "heldout.json")
    best_csd = tmp_path / "GeneratedCSD.py"
    best_csd.write_text("# best failed attempt\n", encoding="utf-8")
    monkeypatch.setattr(queue, "compiled_csd", lambda *args, **kwargs: best_csd)
    monkeypatch.setattr(queue, "synthesis_was_exhausted", lambda *args: True)
    calls = []
    monkeypatch.setattr(queue, "heldout_is_complete", lambda *args: len(calls) >= 1)

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(queue.subprocess, "run", run)

    status = queue.run_job(
        job,
        0,
        repo=tmp_path,
        python=Path("/env/python"),
        state_dir=tmp_path / "state",
    )

    assert status == 0
    assert len(calls) == 1
    state = json.loads((tmp_path / "state" / "gsm-qwen35-2b.json").read_text())
    assert state["status"] == "complete_loss"


def test_cold_queue_service_uses_work_env_and_kills_its_process_group():
    unit = (
        Path(__file__).parents[2]
        / "deploy"
        / "focal"
        / "systemd"
        / "csd-cold-synthesis-queue.service"
    ).read_text()

    assert (
        "ExecStart=/apps/conda/aadivyar/envs/csd/bin/python "
        "-m scripts.runtime.run_cold_synthesis_queue "
    ) in unit
    assert "/scripts/runtime/run_cold_synthesis_queue.py" not in unit
    assert "EnvironmentFile=/home/aadivyar/csd-generation/.env" in unit
    assert "2026-07-19-exhaustive-cold-queue-manifest.json" in unit
    assert "KillMode=control-group" in unit
    assert "Restart=on-failure" in unit
    assert "initial-strategy" not in unit
    assert "warm" not in unit.lower()


def test_repo_version_allows_manifest_only_commit_but_rejects_code_drift(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    script = tmp_path / "scripts" / "runtime" / "worker.py"
    script.parent.mkdir(parents=True)
    script.write_text("VERSION = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "scripts/runtime/worker.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "code"], cwd=tmp_path, check=True)
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True, capture_output=True, check=True
    ).stdout.strip()
    saved = tmp_path / "saved-results" / "manifest.json"
    saved.parent.mkdir()
    saved.write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "add", "saved-results/manifest.json"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "manifest"], cwd=tmp_path, check=True)

    queue.verify_repo_version(tmp_path, code_commit)

    script.write_text("VERSION = 2\n", encoding="utf-8")
    try:
        queue.verify_repo_version(tmp_path, code_commit)
    except queue.ConfigError as error:
        assert "uncommitted code changes" in str(error)
    else:
        raise AssertionError("dirty runtime code must block launch")

    script.write_text("VERSION = 1\n", encoding="utf-8")
    untracked = tmp_path / "synthesis" / "new_path.py"
    untracked.parent.mkdir()
    untracked.write_text("ENABLED = True\n", encoding="utf-8")
    try:
        queue.verify_repo_version(tmp_path, code_commit)
    except queue.ConfigError as error:
        assert "untracked code changes" in str(error)
    else:
        raise AssertionError("untracked synthesis code must block launch")

    untracked.unlink()
    gsm_source = tmp_path / "legacy" / "CRANE" / "src" / "gsm_symbolic" / "parser.py"
    gsm_source.parent.mkdir(parents=True)
    gsm_source.write_text("CHANGED = True\n", encoding="utf-8")
    try:
        queue.verify_repo_version(tmp_path, code_commit)
    except queue.ConfigError as error:
        assert "untracked code changes" in str(error)
    else:
        raise AssertionError("untracked GSM source code must block launch")


def test_verified_repair_attestation_allows_only_its_exact_dirty_code(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    script = tmp_path / "scripts" / "runtime" / "worker.py"
    script.parent.mkdir(parents=True)
    script.write_text("VERSION = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", str(script.relative_to(tmp_path))], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "code"], cwd=tmp_path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True, capture_output=True, check=True
    ).stdout.strip()
    script.write_text("VERSION = 2\n", encoding="utf-8")
    attestation = tmp_path / "repair.json"
    attestation.write_text(
        json.dumps(
            {
                "base_commit": commit,
                "files": {
                    "scripts/runtime/worker.py": hashlib.sha256(script.read_bytes()).hexdigest()
                },
                "verifier_exit": 0,
            }
        ),
        encoding="utf-8",
    )

    queue.verify_repo_version(tmp_path, commit, attestation)

    script.write_text("VERSION = 3\n", encoding="utf-8")
    try:
        queue.verify_repo_version(tmp_path, commit, attestation)
    except queue.ConfigError as error:
        assert "repair attestation" in str(error)
    else:
        raise AssertionError("code changed after attestation must block launch")


def test_saved_exhaustive_manifest_matches_the_approved_call_budget():
    repo = Path(__file__).parents[2]
    manifest = (
        repo / "saved-results" / "2026-07-19-exhaustive-cold-queue-manifest.json"
    )

    commit, jobs = queue.load_manifest(manifest)
    queue.validate_exhaustive_campaign(jobs)

    assert len(commit) == 40
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repo,
        check=True,
    )
    assert len(jobs) == 11
    assert sum(job["max_iterations"] for job in jobs) == 472
    assert sum(job["interrupted_author_calls"] for job in jobs) == 8
    assert sum(
        job["max_iterations"] + job["interrupted_author_calls"] for job in jobs
    ) == queue.APPROVED_AUTHOR_CALL_CAP == 480


def test_exhaustive_campaign_requires_all_eleven_exact_cells_and_unique_outputs(
    tmp_path, monkeypatch
):
    expected_ids = {
        "gsm-qwen25-1p5b", "gsm-qwen25-7b", "gsm-qwen25-14b",
        "gsm-qwen35-2b", "gsm-qwen35-4b", "gsm-qwen35-9b",
        "spider-qwen25-7b", "spider-qwen35-4b", "spider-qwen35-9b",
        "smiles-qwen35-4b-acrylates", "smiles-qwen35-9b-isocyanates",
    }
    assert set(queue.EXPECTED_CELLS) == expected_ids
    interrupted = {
        cell: spec["interrupted_author_calls"]
        for cell, spec in queue.EXPECTED_CELLS.items()
        if spec["interrupted_author_calls"]
    }
    assert interrupted == {
        "gsm-qwen25-1p5b": 2,
        "gsm-qwen25-7b": 2,
        "gsm-qwen25-14b": 2,
        "gsm-qwen35-2b": 2,
    }
    assert sum(
        spec["max_iterations"] for spec in queue.EXPECTED_CELLS.values()
    ) == 472
    assert sum(
        spec["max_iterations"] + spec["interrupted_author_calls"]
        for spec in queue.EXPECTED_CELLS.values()
    ) == queue.APPROVED_AUTHOR_CALL_CAP == 480
    jobs = []
    for cell, spec in queue.EXPECTED_CELLS.items():
        job = _job(spec["dataset"])
        job.update(spec)
        job.update(queue.EXPECTED_RUNTIME_BY_MODEL[job["eval_model"]])
        job["eval_max_seconds"] = 600
        job["cell_id"] = cell
        job["baseline_num_correct"] = 0
        job["baseline_num_examples"] = job["eval_sample_size"]
        job["baseline_source"] = f"outputs/baselines/{cell}.json"
        job["min_accuracy"] = 1 / job["baseline_num_examples"]
        job["output_name"] = f"coldq_{cell}_0719"
        job["log_file"] = f"outputs/generated/{job['output_name']}/run.log"
        job["heldout_output_json"] = f"outputs/reeval/exhaustive_0719/{cell}.json"
        if job["dataset"] == "gsm_symbolic":
            job["heldout_split_file"] = "/repo/gsm_symbolic_crane_proportional_49x49_seed123.json"
        elif job["dataset"] == "spider":
            job["heldout_split_file"] = "/repo/spider_dev_proportional_300x300_seed334.json"
        else:
            job.pop("heldout_split_file", None)
        jobs.append(job)

    queue.validate_exhaustive_campaign(jobs)

    monkeypatch.setattr(queue, "APPROVED_AUTHOR_CALL_CAP", 479)
    try:
        queue.validate_exhaustive_campaign(jobs)
    except queue.ConfigError as error:
        assert "author-call accounting must total 479, got 480" in str(error)
    else:
        raise AssertionError("a campaign above the approved call cap must be rejected")
    monkeypatch.setattr(queue, "APPROVED_AUTHOR_CALL_CAP", 480)

    try:
        queue.validate_exhaustive_campaign(jobs, repo=tmp_path)
    except queue.ConfigError as error:
        assert "baseline_source file is missing" in str(error)
    else:
        raise AssertionError("missing baseline evidence files must block launch")

    try:
        queue.validate_exhaustive_campaign(jobs[:-1])
    except queue.ConfigError as error:
        assert "exactly the 11 approved cells" in str(error)
    else:
        raise AssertionError("a missing cell must block the exhaustive launch")

    jobs[-1]["heldout_output_json"] = jobs[0]["heldout_output_json"]
    try:
        queue.validate_exhaustive_campaign(jobs)
    except queue.ConfigError as error:
        assert "heldout_output_json values must be unique" in str(error)
    else:
        raise AssertionError("shared held-out outputs must block launch")


def test_heldout_split_must_resolve_to_the_canonical_repo_file(tmp_path):
    canonical = (
        tmp_path
        / "environment"
        / "benchmark_splits"
        / "gsm_symbolic_crane_proportional_49x49_seed123.json"
    )
    canonical.parent.mkdir(parents=True)
    canonical.write_text('{"train": [], "test": []}\n', encoding="utf-8")
    job = _job()
    job["heldout_split_file"] = str(canonical)

    queue.validate_heldout_split(job["cell_id"], job, tmp_path)

    wrong = tmp_path / "other" / canonical.name
    wrong.parent.mkdir()
    wrong.write_text(canonical.read_text(), encoding="utf-8")
    job["heldout_split_file"] = str(wrong)
    try:
        queue.validate_heldout_split(job["cell_id"], job, tmp_path)
    except queue.ConfigError as error:
        assert "canonical heldout split" in str(error)
    else:
        raise AssertionError("a same-named split outside the repo path must be rejected")


def test_baseline_evidence_binds_counts_model_train_side_and_source_hash(tmp_path):
    split_file = (
        tmp_path
        / "environment"
        / "benchmark_splits"
        / "gsm_symbolic_crane_proportional_49x49_seed123.json"
    )
    split_file.parent.mkdir(parents=True)
    split_file.write_text('{"train": [], "test": []}\n', encoding="utf-8")
    raw_artifact = tmp_path / "raw-baseline.json"
    raw_artifact.write_text('{"accuracy": 0.4081632653}\n', encoding="utf-8")
    artifact = tmp_path / "baseline.json"
    job = _job()
    job["baseline_num_correct"] = 20
    job["baseline_num_examples"] = 49
    normalized = {
        "dataset": job["dataset"],
        "eval_model": job["eval_model"],
        "split_name": "train",
        "baseline_strategy": "crane",
        "num_correct": 20,
        "num_examples": 49,
        "split_file": str(split_file.relative_to(tmp_path)),
        "split_file_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "raw_source_artifact": str(raw_artifact),
        "raw_source_sha256": hashlib.sha256(raw_artifact.read_bytes()).hexdigest(),
    }
    artifact.write_text(json.dumps(normalized), encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    job["baseline_source"] = str(evidence)
    evidence.write_text(
        json.dumps(
            {
                "cells": {
                    job["cell_id"]: {
                        "dataset": job["dataset"],
                        "eval_model": job["eval_model"],
                        "split_name": "train",
                        "baseline_strategy": "crane",
                        "num_correct": 20,
                        "num_examples": 49,
                        "source_artifact": str(artifact),
                        "source_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    queue.validate_baseline_evidence(job["cell_id"], job, tmp_path)

    payload = json.loads(evidence.read_text())
    normalized["baseline_strategy"] = "itergen"
    artifact.write_text(json.dumps(normalized), encoding="utf-8")
    payload["cells"][job["cell_id"]]["baseline_strategy"] = "itergen"
    payload["cells"][job["cell_id"]]["source_sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    try:
        queue.validate_baseline_evidence(job["cell_id"], job, tmp_path)
    except queue.ConfigError as error:
        assert "wrong baseline strategy" in str(error)
    else:
        raise AssertionError("the wrong dataset comparator must block launch")

    normalized["baseline_strategy"] = "crane"
    wrong_split = tmp_path / "other-train.json"
    wrong_split.write_text('{"train": []}\n', encoding="utf-8")
    normalized["split_file"] = str(wrong_split.relative_to(tmp_path))
    normalized["split_file_sha256"] = hashlib.sha256(
        wrong_split.read_bytes()
    ).hexdigest()
    artifact.write_text(json.dumps(normalized), encoding="utf-8")
    payload["cells"][job["cell_id"]]["baseline_strategy"] = "crane"
    payload["cells"][job["cell_id"]]["source_sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    try:
        queue.validate_baseline_evidence(job["cell_id"], job, tmp_path)
    except queue.ConfigError as error:
        assert "wrong canonical train split" in str(error)
    else:
        raise AssertionError("a different train split must block launch")

    normalized["split_file"] = str(split_file.relative_to(tmp_path))
    normalized["split_file_sha256"] = hashlib.sha256(split_file.read_bytes()).hexdigest()
    normalized["num_correct"] = 19
    artifact.write_text(json.dumps(normalized), encoding="utf-8")
    payload["cells"][job["cell_id"]]["source_sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    try:
        queue.validate_baseline_evidence(job["cell_id"], job, tmp_path)
    except queue.ConfigError as error:
        assert "source artifact does not support manifest measurement" in str(error)
    else:
        raise AssertionError("unsupported baseline counts in source must block launch")

    normalized["num_correct"] = 20
    artifact.write_text(json.dumps(normalized), encoding="utf-8")
    payload["cells"][job["cell_id"]]["source_sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    payload["cells"][job["cell_id"]]["num_correct"] = 19
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    try:
        queue.validate_baseline_evidence(job["cell_id"], job, tmp_path)
    except queue.ConfigError as error:
        assert "does not match manifest counts" in str(error)
    else:
        raise AssertionError("baseline evidence with different counts must block launch")
