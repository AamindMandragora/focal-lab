from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_dashboard_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "experiment_dashboard.py"
    spec = importlib.util.spec_from_file_location("experiment_dashboard", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_summary_includes_failure_attempt_metrics_and_run_metadata(tmp_path):
    dashboard = load_dashboard_module()
    dashboard.ROOT_DIR = tmp_path

    report_path = tmp_path / "outputs" / "generated" / "run" / "results" / "failure_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "total_attempts": 40,
                "timestamp": "2026-05-25T07:30:00Z",
                "run_configuration": {
                    "output_name": "metadecode_spider_Qwen_Qwen2.5_1.5B_Instruct_opus4.7_iter40_tb1_ms900",
                    "max_iterations": 40,
                    "thresholds": {
                        "min_accuracy": 0.537,
                        "min_syntax_rate": 0.9,
                        "require_delimiters": True,
                    },
                    "author_model": {
                        "backend": "anthropic",
                        "model": "claude-opus-4-7",
                        "anthropic_thinking": "adaptive",
                        "anthropic_effort": "xhigh",
                    },
                    "evaluation": {
                        "dataset": "spider",
                        "eval_model": "Qwen/Qwen2.5-1.5B-Instruct",
                        "eval_backend": "vllm",
                        "eval_sample_size": 50,
                        "eval_max_steps": 900,
                        "eval_step_token_budget": 1,
                    },
                    "synthesis_controls": {
                        "helper_selection_policy": "bandit",
                        "adaptive_helper_mask": True,
                        "refinement_beam_size": 2,
                    },
                },
                "attempts": [
                    {
                        "attempt_number": 1,
                        "evaluation": {"accuracy": 0.1, "syntax_rate": 0.8, "num_correct": 5, "num_examples": 50},
                    },
                    {
                        "attempt_number": 2,
                        "evaluation": {"accuracy": 0.32, "syntax_rate": 0.92, "num_correct": 16, "num_examples": 50},
                    },
                ],
            }
        )
    )

    summary = dashboard.report_summary(report_path)

    assert summary["accuracy"] == 0.32
    assert summary["syntax_rate"] == 0.92
    assert summary["metric_source"] == "best attempt 2"
    assert summary["dataset"] == "spider"
    assert summary["eval_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert summary["author_model"] == "claude-opus-4-7"
    assert summary["min_accuracy"] == 0.537
    assert summary["min_syntax_rate"] == 0.9
    assert summary["max_iterations"] == 40
    assert summary["eval_max_steps"] == 900
    assert summary["eval_step_token_budget"] == 1
    assert summary["helper_selection_policy"] == "bandit"
    assert summary["refinement_beam_size"] == 2
    assert summary["reported_at"] == "2026-05-25T07:30:00Z"


def test_process_title_exposes_matrix_ablation_metadata():
    dashboard = load_dashboard_module()

    process = dashboard.classify_run(
        {
            "pid": "123",
            "ppid": "1",
            "elapsed_seconds": 90,
            "cmd": (
                "python run_all_tests.py --reuse-baselines --accuracy-win-margin 0.03 "
                "--benchmarks gsm,spider --skip-main --ablation-sections A,B "
                "--main-synthesis-iterations 40 "
                "--generated-output-dir outputs/generated/matrix_ablation_AB_gpu1"
            ),
        }
    )

    assert process["ablation_sections"] == "A,B"
    assert process["accuracy_win_margin"] == "0.03"
    assert process["max_iterations"] == "40"
    assert process["skip_main"] is True
    assert dashboard.process_title(process) == "ablation A,B / gsm,spider / model=default / iter=40 / margin=0.03"


def test_dashboard_html_uses_per_gpu_queue_and_failure_kind_coloring():
    dashboard = load_dashboard_module()

    assert "<section><h2>Queue</h2>" not in dashboard.HTML
    assert "Per-GPU Queue" in dashboard.HTML
    assert "Runtime Alerts" in dashboard.HTML
    assert "renderRuntimeAlerts(data)" in dashboard.HTML
    assert "r.kind === 'failure' ? 'bad'" in dashboard.HTML
    assert "r.reported_at" in dashboard.HTML


def test_runtime_alerts_extract_api_credit_and_traceback_failures():
    dashboard = load_dashboard_module()

    alerts = dashboard.runtime_alerts_from_lines(
        [
            "normal progress",
            "[api-retry] anthropic HTTP 429; retry 1/5 after 20.0s",
            '"message": "Your prepayment credits are depleted."',
            "Traceback (most recent call last):",
            "RuntimeError: provider failed",
        ]
    )

    assert [alert["kind"] for alert in alerts] == [
        "api_retry",
        "credits",
        "traceback",
        "traceback",
    ]
    assert alerts[1]["line"] == 3
    assert "credits are depleted" in alerts[1]["message"]


def test_collect_runtime_alerts_sorts_newest_first():
    dashboard = load_dashboard_module()

    logs = [
        {"path": "old.log", "mtime": 1, "alerts": [{"kind": "traceback", "line": 5, "message": "old"}]},
        {"path": "new.log", "mtime": 2, "alerts": [{"kind": "api_retry", "line": 1, "message": "new"}]},
    ]

    alerts = dashboard.collect_runtime_alerts(logs)

    assert alerts[0]["path"] == "new.log"
    assert alerts[1]["path"] == "old.log"
