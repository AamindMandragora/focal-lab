from argparse import Namespace
import json

from synthesis.evaluate import run_legacy_fixed_strategy as fixed


def test_crane_delimited_start_grammar_wraps_default_start_rule():
    grammar = "\n".join(
        [
            "syncode: start",
            "",
            "start: sql_stmt",
            'csd_start: sql_stmt ">>"',
            'sql_stmt: "SELECT"',
        ]
    )

    wrapped = fixed._crane_delimited_start_grammar(grammar)

    assert wrapped.startswith('start: "<<" crane_body ">>"\n')
    assert "crane_body: sql_stmt" in wrapped
    assert 'csd_start: sql_stmt ">>"' in wrapped
    assert "\nstart: sql_stmt" not in wrapped


def test_crane_strategy_dispatches_to_shared_evaluator_adapter(monkeypatch):
    calls = []

    def fake_shared_adapter(args, dataset):
        calls.append((args.strategy, dataset))
        return 13

    monkeypatch.setattr(fixed, "_crane_via_adaptive_syncode", fake_shared_adapter)

    for dataset in ["gsm_symbolic", "spider", "smiles"]:
        args = Namespace(strategy="crane", dataset=dataset)
        assert fixed.run_crane_legacy_adapter(args) == 13

    assert calls == [
        ("crane", "gsm_symbolic"),
        ("crane", "spider"),
        ("crane", "smiles"),
    ]


def test_minimal_json_can_mark_shared_crane_adapter(tmp_path):
    out = tmp_path / "baseline.json"

    fixed._build_minimal_json(
        [{"question": "q", "llm_response": "a", "correct": True, "syntax_valid": True}],
        out,
        extra_metrics={"adapter": "crane_shared_evaluator"},
    )

    payload = json.loads(out.read_text())
    assert payload["accuracy"] == 1.0
    assert payload["syntax_rate"] == 1.0
    assert payload["metrics"]["adapter"] == "crane_shared_evaluator"
