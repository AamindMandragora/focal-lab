#!/usr/bin/env python3
"""Run one handoff-sized slice of the generalization experiments.

The full matrix is intentionally large. This launcher exposes named pieces that
can be handed to different people or machines without accidentally running the
whole cross product.

Default behavior is parallel-friendly: it passes --no-kill-vllm-before-cells to
the master matrix runner so one slice does not kill another slice's active
workers. Run `cleanup-vllm --yes` once before starting parallel jobs if you need
to clear stale workers.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "generated-csd"
DEFAULT_PYTHON = "/opt/anaconda/bin/python"

QWEN_MODELS = {
    "qwen15": "qwen25_coder_1p5b_instruct",
    "qwen7": "qwen25_coder_7b_instruct",
    "qwen14": "qwen25_coder_14b_instruct",
}

GENERATION_MODELS = {
    "gpt54": "gpt54",
    "opus47": "opus47",
    "gemini31pro": "gemini31pro",
}


@dataclass(frozen=True)
class Part:
    name: str
    description: str
    command: tuple[str, ...]


def quote_cmd(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'").strip('"')
    return env


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.load_env:
        env.update(load_env_file(PROJECT_ROOT / ".env"))
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env


def master_cmd(
    args: argparse.Namespace,
    *,
    part_name: str,
    datasets: str,
    methods: str,
    models: str,
    generation_models: str = "gpt54",
    include_ablations: bool = False,
) -> tuple[str, ...]:
    run_name = f"{args.run_name_prefix}_{part_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    cmd = [
        args.python,
        "scripts/master_experiment_matrix.py",
        "--run-name",
        run_name,
        "--output-dir",
        str(args.output_dir),
        "--datasets",
        datasets,
        "--methods",
        methods,
        "--models",
        models,
        "--generation-models",
        generation_models,
        "--no-kill-vllm-before-cells",
    ]
    if not include_ablations:
        cmd.append("--no-include-ablations")
    return tuple(cmd)


def ablation_cmd(
    args: argparse.Namespace,
    *,
    part_name: str,
    dataset: str,
    sweep: str,
) -> tuple[str, ...]:
    run_name = f"{args.run_name_prefix}_{part_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    env_parts = [
        f"DATASET_VALUES={shlex.quote(dataset)}",
        f"RUN_NAME={shlex.quote(run_name)}",
        f"OUTPUT_DIR={shlex.quote(str(args.output_dir))}",
        f"ABLATION_SWEEP={shlex.quote(sweep)}",
        "GENERATION_MODEL=gpt-5.4",
        "GENERATION_BACKEND=openai",
        "KILL_VLLM_WORKERS=0",
    ]
    if sweep == "maxsteps":
        env_parts.append("FIXED_SYNTHESIS_ITERATIONS=20")
    elif sweep == "iterations":
        env_parts.append("FIXED_MAX_STEPS=512")
    return (
        "bash",
        "-lc",
        " ".join([*env_parts, "scripts/run_generalization_ablation_grid.sh"]),
    )


def build_parts(args: argparse.Namespace) -> dict[str, Part]:
    parts: dict[str, Part] = {}

    for short, model in QWEN_MODELS.items():
        name = f"baselines_{short}"
        parts[name] = Part(
            name=name,
            description=f"CRANE, IterGen, CARS, and unconstrained on GSM/Spider/SMILES for {model}.",
            command=master_cmd(
                args,
                part_name=name,
                datasets="gsm,spider,smiles",
                methods="crane,itergen,cars,unconstrained",
                models=model,
            ),
        )

    for short, model in QWEN_MODELS.items():
        name = f"metadecode_gpt54_{short}"
        parts[name] = Part(
            name=name,
            description=f"metaDecode with GPT-5.4 synthesis on GSM/Spider/SMILES for {model}.",
            command=master_cmd(
                args,
                part_name=name,
                datasets="gsm,spider,smiles",
                methods="metadecode",
                models=model,
                generation_models="gpt54",
            ),
        )

    for gen in ("opus47", "gemini31pro"):
        name = f"metadecode_{gen}_qwen7"
        parts[name] = Part(
            name=name,
            description=f"metaDecode with {gen} synthesis on GSM/Spider/SMILES for Qwen2.5-Coder-7B only.",
            command=master_cmd(
                args,
                part_name=name,
                datasets="gsm,spider,smiles",
                methods="metadecode",
                models=QWEN_MODELS["qwen7"],
                generation_models=GENERATION_MODELS[gen],
            ),
        )

    for dataset in ("gsm", "spider", "smiles"):
        name = f"ablation_maxsteps_{dataset}"
        parts[name] = Part(
            name=name,
            description=(
                f"metaDecode maxSteps-only ablation for {dataset}: "
                "maxSteps 256/512/1024 with synthesis iterations fixed at 20."
            ),
            command=ablation_cmd(args, part_name=name, dataset=dataset, sweep="maxsteps"),
        )

        name = f"ablation_iterations_{dataset}"
        parts[name] = Part(
            name=name,
            description=(
                f"metaDecode synthesis-iterations-only ablation for {dataset}: "
                "iterations 5/10/15/20 with maxSteps fixed at 512."
            ),
            command=ablation_cmd(args, part_name=name, dataset=dataset, sweep="iterations"),
        )

    return parts


def print_parts(parts: dict[str, Part]) -> None:
    for part in parts.values():
        print(f"{part.name}")
        print(f"  {part.description}")
        print(f"  {quote_cmd(part.command)}")


def launch_part(args: argparse.Namespace, part: Part) -> int:
    log_dir = args.output_dir / "logs" / "experiment_parts"
    log_path = log_dir / f"{part.name}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    print(f"[part] {part.name}")
    print(f"[description] {part.description}")
    print(f"[command] {quote_cmd(part.command)}")
    print(f"[log] {log_path}")

    if args.print_only:
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    env = base_env(args)

    if args.background:
        with log_path.open("w") as log_file:
            proc = subprocess.Popen(
                list(part.command),
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        metadata_path = log_path.with_suffix(".json")
        metadata_path.write_text(json.dumps({
            "part": part.name,
            "description": part.description,
            "pid": proc.pid,
            "command": list(part.command),
            "log_path": str(log_path),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, indent=2))
        print(f"[background] pid={proc.pid}")
        print(f"[metadata] {metadata_path}")
        return 0

    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            list(part.command),
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        return proc.wait()


def cleanup_vllm(args: argparse.Namespace) -> int:
    patterns = ("vllm", "VLLM", "multiproc_worker_utils", "VllmWorkerProcess")
    matches: list[tuple[int, str]] = []
    for pattern in patterns:
        proc = subprocess.run(
            ["pgrep", "-a", "-u", str(os.getuid()), "-f", pattern],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        for line in proc.stdout.splitlines():
            parts = line.split(maxsplit=1)
            if not parts:
                continue
            pid = int(parts[0])
            command = parts[1] if len(parts) > 1 else ""
            if pid == os.getpid() or "run_experiment_part.py" in command:
                continue
            matches.append((pid, command))

    deduped = sorted(dict(matches).items())
    if not deduped:
        print("[cleanup-vllm] no owned vLLM worker processes found")
        return 0

    for pid, command in deduped:
        print(f"{pid} {command}")

    if not args.yes:
        print("[cleanup-vllm] dry-run only; pass --yes to terminate these processes")
        return 1

    for pid, _ in deduped:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(2)
    for pid, _ in deduped:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    print(f"[cleanup-vllm] terminated {len(deduped)} process(es)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--run-name-prefix", default="handoff")
    parser.add_argument("--no-load-env", dest="load_env", action="store_false", default=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List named experiment parts and commands.")
    list_parser.add_argument("--names-only", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run or print one named experiment part.")
    run_parser.add_argument("part")
    run_parser.add_argument("--background", action="store_true", help="Launch detached and print pid/log.")
    run_parser.add_argument("--print-only", action="store_true", help="Print command without running.")

    cleanup_parser = subparsers.add_parser("cleanup-vllm", help="List or kill owned vLLM worker processes.")
    cleanup_parser.add_argument("--yes", action="store_true", help="Actually terminate matched processes.")

    args = parser.parse_args()

    if args.command == "cleanup-vllm":
        return cleanup_vllm(args)

    parts = build_parts(args)
    if args.command == "list":
        if args.names_only:
            print("\n".join(parts))
        else:
            print_parts(parts)
        return 0

    if args.part not in parts:
        print(f"Unknown part: {args.part}", file=sys.stderr)
        print("Available parts:", file=sys.stderr)
        for name in parts:
            print(f"  {name}", file=sys.stderr)
        return 2
    return launch_part(args, parts[args.part])


if __name__ == "__main__":
    raise SystemExit(main())
