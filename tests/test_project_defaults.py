from __future__ import annotations

from pathlib import Path

from synthesis.project_defaults import default_dafny_path, repo_root


def test_default_dafny_path_prefers_repo_binary(monkeypatch):
    monkeypatch.delenv("DAFNY_PATH", raising=False)
    repo_dafny = repo_root() / "dafny" / "dafny"
    assert repo_dafny.is_file(), "expected vendored dafny/dafny in repository"
    assert default_dafny_path() == str(repo_dafny)


def test_default_dafny_path_honors_dafny_path_env(monkeypatch, tmp_path: Path):
    override = tmp_path / "custom-dafny"
    override.write_text("#!/bin/sh\n", encoding="utf-8")
    override.chmod(0o755)
    monkeypatch.setenv("DAFNY_PATH", str(override))
    assert default_dafny_path() == str(override)
