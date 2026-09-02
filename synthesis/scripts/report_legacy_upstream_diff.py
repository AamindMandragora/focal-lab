#!/usr/bin/env python3
"""Compare legacy/ baseline trees against pristine upstream clones.

Upstream URLs default from environment/legacy/repos.json. Run with --help for examples.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "synthesis" / "run_synthesis.py").is_file():
            return parent
    raise RuntimeError(f"Could not find repo root from {start}")


def _load_manifest(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "environment" / "legacy" / "repos.json"
    return json.loads(path.read_text())


def _run_diff(left: Path, right: Path) -> tuple[int, str]:
    """Recursive diff excluding .git; returns (returncode, stdout+stderr)."""
    proc = subprocess.run(
        ["diff", "-ruN", "--exclude=.git", str(left), str(right)],
        capture_output=True,
        text=True,
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def _clone_manifest_entries(
    repo_root: Path,
    manifest: dict[str, Any],
    tmp: Path,
    shallow: bool,
) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    for _key, meta in manifest.items():
        if not isinstance(meta, dict):
            continue
        url = meta.get("git_url")
        sub = meta.get("legacy_subdir")
        if not url or not sub:
            continue
        dest = tmp / sub
        if dest.exists():
            shutil.rmtree(dest)
        cmd = ["git", "clone"]
        if shallow:
            cmd.extend(["--depth", "1"])
        cmd.extend([url, str(dest)])
        subprocess.run(cmd, check=True, cwd=str(tmp))
    return tmp


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python synthesis/scripts/report_legacy_upstream_diff.py --fetch-upstream\n"
            "  python synthesis/scripts/report_legacy_upstream_diff.py \\\n"
            "      --upstream-base /data/pristine_legacy --json-out /tmp/legacy_diff.json\n"
            "\n"
            "Exit status is 1 when a tree is missing or file contents differ from upstream; "
            "0 when every compared tree matches.\n"
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (auto-detected when omitted)",
    )
    parser.add_argument(
        "--upstream-base",
        type=Path,
        default=None,
        help="Directory containing CRANE/, itergen/, cars/ pristine clones",
    )
    parser.add_argument(
        "--fetch-upstream",
        action="store_true",
        help="Clone manifest URLs into a temporary directory and diff against that",
    )
    parser.add_argument(
        "--no-shallow",
        action="store_true",
        help="When using --fetch-upstream, clone full history (default: shallow)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write structured summary JSON to this path",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve()
    repo_root = args.repo_root or _repo_root(here.parents[2])
    manifest = _load_manifest(repo_root)
    legacy_root = repo_root / "legacy"

    upstream_base = args.upstream_base
    tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.fetch_upstream:
        tmp_dir = tempfile.TemporaryDirectory(prefix="vas-legacy-upstream-")
        upstream_base = _clone_manifest_entries(
            repo_root,
            manifest,
            Path(tmp_dir.name),
            shallow=not args.no_shallow,
        )
    elif upstream_base is None:
        print(
            "error: provide --upstream-base or --fetch-upstream",
            file=sys.stderr,
        )
        return 2

    report: dict[str, Any] = {"repos": {}, "legacy_root": str(legacy_root)}
    any_diff = False
    lines_out: list[str] = []

    for key, meta in manifest.items():
        if not isinstance(meta, dict):
            continue
        sub = meta.get("legacy_subdir")
        url = meta.get("git_url")
        if not sub:
            continue
        left = legacy_root / sub
        right = upstream_base / sub
        if not left.is_dir():
            msg = f"[{sub}] missing local tree: {left}"
            lines_out.append(msg)
            report["repos"][sub] = {"status": "missing_local", "url": url}
            any_diff = True
            continue
        if not right.is_dir():
            msg = f"[{sub}] missing upstream tree: {right}"
            lines_out.append(msg)
            report["repos"][sub] = {"status": "missing_upstream", "url": url}
            any_diff = True
            continue

        code, diff_text = _run_diff(right, left)
        # diff returns 1 when files differ; 2 means trouble.
        if code not in (0, 1):
            lines_out.append(f"[{sub}] diff failed (exit {code}):\n{diff_text}")
            report["repos"][sub] = {"status": "diff_error", "exit_code": code}
            any_diff = True
            continue

        if code == 0:
            lines_out.append(f"[{sub}] no file-level differences vs upstream clone ({url})")
            report["repos"][sub] = {
                "status": "identical",
                "url": url,
                "upstream_path": str(right),
            }
        else:
            lines_out.append(
                f"[{sub}] differences vs upstream ({url}); showing unified diff:\n{diff_text}"
            )
            report["repos"][sub] = {
                "status": "different",
                "url": url,
                "upstream_path": str(right),
                "diff": diff_text,
            }
            any_diff = True

    if tmp_dir is not None:
        tmp_dir.cleanup()

    text = "\n".join(lines_out) + "\n"
    sys.stdout.write(text)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")

    return 1 if any_diff else 0


if __name__ == "__main__":
    raise SystemExit(main())
