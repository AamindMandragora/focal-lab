import json

from synthesis.evaluate import feedback_loop


def test_atomic_report_write_preserves_the_previous_file_if_interrupted(
    tmp_path, monkeypatch
):
    report = tmp_path / "success_report.json"
    report.write_text('{"status": "old"}\n', encoding="utf-8")

    def interrupted_dump(payload, handle, **kwargs):
        handle.write("{")
        raise OSError("interrupted")

    monkeypatch.setattr(feedback_loop.json, "dump", interrupted_dump)
    try:
        feedback_loop._write_json_atomic(report, {"status": "new"})
    except OSError:
        pass
    else:
        raise AssertionError("the simulated interrupted write must fail")

    assert json.loads(report.read_text()) == {"status": "old"}
    assert list(tmp_path.glob("*.tmp")) == []
