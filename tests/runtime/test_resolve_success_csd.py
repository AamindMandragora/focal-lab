import importlib.util
import json
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).parents[2] / ".context/resolve_success_csd.py"
SPEC = importlib.util.spec_from_file_location("resolve_success_csd", MODULE_PATH)
resolver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = resolver
SPEC.loader.exec_module(resolver)


def test_resolve_uses_latest_success_report(tmp_path, monkeypatch):
    output_name = "post14b_rebar_gsm-qwen35-9b_0711"
    run_dir = tmp_path / "run"
    compiled = run_dir / "compiled"
    compiled.mkdir(parents=True)
    (compiled / "GeneratedCSD.py").write_text("# compiled")
    (run_dir / "results").mkdir()
    (run_dir / "results/success_report.json").write_text(
        json.dumps({"compiled_dir": str(compiled)})
    )
    latest = tmp_path / "outputs/generated" / output_name / "latest_run.txt"
    latest.parent.mkdir(parents=True)
    latest.write_text(str(run_dir))

    assert resolver.resolve(tmp_path, "gsm-qwen35-9b") == compiled / "GeneratedCSD.py"
