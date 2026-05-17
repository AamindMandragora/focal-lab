#!/usr/bin/env python3
"""Python launcher for the full CSD experiment matrix.

This mirrors run_all_tests.sh while keeping the control flow easier to inspect
and extend. It intentionally preserves the bash runner's public CLI, output
layout, matrix order, cache semantics, and MetaDecode target selection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
CALLER_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
DEFAULT_GSM_SPLIT_FILE = (
    ROOT_DIR / "environment" / "benchmark_splits" / "gsm_symbolic_crane_proportional.json"
)
DEFAULT_SPIDER_SPLIT_FILE = ROOT_DIR / "environment" / "benchmark_splits" / "spider_dev_proportional.json"

DEFAULT_MODELS = (
    "Qwen/Qwen2.5-Coder-1.5B-Instruct,"
    "Qwen/Qwen2.5-Coder-7B-Instruct,"
    "Qwen/Qwen2.5-Coder-14B-Instruct,"
    "meta-llama/Llama-3.1-8B-Instruct"
)
DEFAULT_BENCHMARKS = "gsm,spider,smiles"
DEFAULT_STRATEGIES = "unconstrained,gcd,crane,itergen,cars,metadecode"
DEFAULT_TOKEN_BUDGETS = "1,2,4"
DEFAULT_SYNTH_ITERS = "3,5,10"
DEFAULT_GEN_MODELS = "gpt5.4,opus4.7"
DEFAULT_STEP_BUDGETS = "256,512,1024"
DEFAULT_SMILES_CLASSES = "acrylates,chain_extenders,isocyanates"
CSD_TARGET_STRATEGIES = ("crane", "itergen", "cars")
OOM_RE = re.compile(
    r"out of memory|OutOfMemoryError|CUDA out of memory|"
    r"CUDA error: out of memory|torch\.cuda\.OutOfMemoryError|"
    r"cumemAllocator|RESOURCE_EXHAUSTED|"
    r"Free memory on device|desired GPU memory utilization",
    re.IGNORECASE,
)


from synthesis.env_utils import load_env_file
from synthesis.evaluate.benchmarks.smiles.dataset import normalize_smiles_classes


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def slugify(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_").replace("-", "_")


def normalize_benchmark(value: str) -> str:
    return "gsm_symbolic" if value == "gsm" else value


def command_text(cmd: list[str]) -> str:
    return shlex.join([str(part) for part in cmd])


def line_count(path: Path) -> int:
    try:
        return len(path.read_text().splitlines())
    except Exception:
        return 0


def canonical(path: Path) -> str:
    return str(path.resolve())


@dataclass
class Config:
    models: list[str]
    benchmarks: list[str]
    strategies: list[str]
    token_budgets: list[str]
    synth_iters: list[str]
    gen_models: list[str]
    step_budgets: list[str]
    smiles_classes: list[str]
    eval_backend: str
    device: str
    generation_sample_size: str
    eval_sample_size: str
    gsm_generation_sample_size: str
    gsm_eval_sample_size: str
    eval_max_steps: str
    eval_max_seconds_per_example: str
    eval_min_examples_before_threshold_stop: str
    vllm_gpu_memory_utilization: str
    vllm_tensor_parallel_size: int
    dafny_path: str
    generated_output_dir: Path
    baseline_output_dir: Path
    ablation_output_dir: Path
    baseline_cache_mode: str
    gsm_split_file: str = ""
    spider_split_file: str = ""
    dry_run: bool = False
    skip_main: bool = False
    skip_ablations: bool = False
    conda_env_path: Path = Path("/apps/conda/advayth2/envs/advayth2")
    cuda_devices: str = "auto"
    cuda_oom_fallback: str = "auto"
    free_gpu_max_used_mb: int = 1024
    gpu_wait_seconds: int = 60
    gpu_wait_timeout_seconds: int = 0

@dataclass
class Runner:
    config: Config
    env: dict[str, str]
    prepared_baselines: set[tuple[str, str, str, str, str]] = field(default_factory=set)

    def configure_cuda_devices(self) -> bool:
        from synthesis.evaluate.benchmarks.common.model_utils import limit_cuda_visible_devices

        selected = self.resolve_cuda_visible_devices("primary", ())
        if selected:
            selected = limit_cuda_visible_devices(selected) or selected
            self.env["CUDA_VISIBLE_DEVICES"] = selected
            os.environ["CUDA_VISIBLE_DEVICES"] = selected
            if self.config.cuda_devices == "auto" and not CALLER_CUDA_VISIBLE_DEVICES:
                print(
                    f"[env] CUDA_VISIBLE_DEVICES={selected} "
                    f"(auto-selected; max used <= {self.config.free_gpu_max_used_mb} MiB)"
                )
            else:
                print(f"[env] CUDA_VISIBLE_DEVICES={selected}")
        else:
            self.env.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            if self.config.dry_run:
                print("[env] CUDA_VISIBLE_DEVICES=<auto> (dry-run; no idle GPU selected)")
            else:
                print("[error] Could not select a CUDA device.", file=sys.stderr)
                return False

        if self.config.cuda_oom_fallback:
            print(
                f"[env] RUN_ALL_TESTS_CUDA_OOM_FALLBACK={self.config.cuda_oom_fallback} "
                "(OOM retry; set empty to disable)"
            )
        else:
            print("[env] RUN_ALL_TESTS_CUDA_OOM_FALLBACK unset/disabled")
        return True

    def cuda_free_gpu_candidates(self) -> list[str]:
        nvidia_smi = shutil.which("nvidia-smi", path=self.env.get("PATH"))
        if not nvidia_smi:
            return []
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            env=self.env,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            return []

        candidates: list[tuple[int, int, int, str]] = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 4 or not all(fields[:4]):
                continue
            try:
                index = fields[0]
                used = int(float(fields[1]))
                free = int(float(fields[2]))
                util = int(float(fields[3]))
            except ValueError:
                continue
            if used <= self.config.free_gpu_max_used_mb:
                candidates.append((used, util, 1_000_000 - free, index))
        return [candidate[3] for candidate in sorted(candidates)]

    def select_free_cuda_device(self, skip: tuple[str, ...]) -> str | None:
        skip_set = set(skip)
        for gpu in self.cuda_free_gpu_candidates():
            if gpu not in skip_set:
                return gpu
        return None

    def wait_for_free_cuda_device(self, skip: tuple[str, ...]) -> str | None:
        start = time.monotonic()
        while True:
            selected = self.select_free_cuda_device(skip)
            if selected:
                return selected
            if self.config.dry_run:
                return None
            if not shutil.which("nvidia-smi", path=self.env.get("PATH")):
                print(
                    "[error] RUN_ALL_TESTS_CUDA_DEVICES=auto requires nvidia-smi; "
                    "set RUN_ALL_TESTS_CUDA_DEVICES explicitly to override.",
                    file=sys.stderr,
                )
                return None
            if self.config.gpu_wait_timeout_seconds > 0:
                elapsed = int(time.monotonic() - start)
                if elapsed >= self.config.gpu_wait_timeout_seconds:
                    print(
                        f"[error] No GPU became idle within "
                        f"{self.config.gpu_wait_timeout_seconds}s.",
                        file=sys.stderr,
                    )
                    return None
            print(
                f"[env] No GPU with <= {self.config.free_gpu_max_used_mb} MiB used; "
                f"waiting {self.config.gpu_wait_seconds}s...",
                file=sys.stderr,
            )
            time.sleep(self.config.gpu_wait_seconds)

    def resolve_cuda_visible_devices(self, role: str, skip: tuple[str, ...]) -> str | None:
        requested = (
            self.config.cuda_oom_fallback
            if role == "fallback"
            else self.config.cuda_devices
        )
        if not requested:
            return None
        if requested != "auto":
            return requested
        if role != "fallback" and CALLER_CUDA_VISIBLE_DEVICES:
            from synthesis.evaluate.benchmarks.common.model_utils import limit_cuda_visible_devices

            return limit_cuda_visible_devices(CALLER_CUDA_VISIBLE_DEVICES)
        selected = self.wait_for_free_cuda_device(skip)
        if selected:
            from synthesis.evaluate.benchmarks.common.model_utils import limit_cuda_visible_devices

            return limit_cuda_visible_devices(selected)
        return None

    def generation_sample_size(self, benchmark: str) -> str:
        if benchmark == "gsm_symbolic":
            return self.config.gsm_generation_sample_size
        return self.config.generation_sample_size

    def evaluation_sample_size(self, benchmark: str) -> str:
        if benchmark == "gsm_symbolic":
            return self.config.gsm_eval_sample_size
        return self.config.eval_sample_size

    def ensure_split_manifests(self) -> None:
        """Require tracked stratified manifests under environment/benchmark_splits/."""
        normalized = {normalize_benchmark(benchmark) for benchmark in self.config.benchmarks}
        if not self.config.skip_ablations:
            normalized.update({"gsm_symbolic", "spider", "smiles"})

        if "gsm_symbolic" in normalized:
            gsm_path = Path(self.config.gsm_split_file)
            if not gsm_path.is_file():
                raise SystemExit(
                    f"GSM split manifest not found: {gsm_path}\n"
                    "Regenerate tracked splits with:\n"
                    "  python -m synthesis.evaluate.benchmarks.write_fixed_benchmark_splits"
                )

        if "spider" in normalized:
            spider_path = Path(self.config.spider_split_file)
            if not spider_path.is_file():
                raise SystemExit(
                    f"Spider split manifest not found: {spider_path}\n"
                    "Regenerate tracked splits with:\n"
                    "  python -m synthesis.evaluate.benchmarks.write_fixed_benchmark_splits"
                )

    def gsm_split_name_for_role(self, role: str) -> str:
        """
        Map generation/evaluation roles to manifest keys.

        The default CRANE proportional manifest has train_size=0; use the eval
        pool for synthesis as well so metadecode tunes on the same fixed subset.
        """
        path = Path(self.config.gsm_split_file)
        if not path.is_file():
            return "eval" if role != "train" else "train"
        manifest = json.loads(path.read_text())
        if role == "train" and not manifest.get("train_indices"):
            return "eval"
        return "train" if role == "train" else "eval"

    def add_generation_split_flags(self, cmd: list[str], benchmark: str) -> None:
        if self.config.gsm_split_file and benchmark == "gsm_symbolic":
            cmd += [
                "--gsm-split-file",
                self.config.gsm_split_file,
                "--gsm-split-name",
                self.gsm_split_name_for_role("train"),
            ]
        if self.config.spider_split_file and benchmark == "spider":
            cmd += ["--spider-split-file", self.config.spider_split_file, "--spider-split-name", "train"]

    def add_evaluation_split_flags(self, cmd: list[str], benchmark: str) -> None:
        if self.config.gsm_split_file and benchmark == "gsm_symbolic":
            cmd += [
                "--gsm-split-file",
                self.config.gsm_split_file,
                "--gsm-split-name",
                self.gsm_split_name_for_role("eval"),
            ]
        if self.config.spider_split_file and benchmark == "spider":
            cmd += ["--spider-split-file", self.config.spider_split_file, "--spider-split-name", "eval"]

    def add_vllm_parallel_flags(self, cmd: list[str]) -> None:
        if self.config.eval_backend != "vllm":
            return
        cmd += [
            "--vllm-tensor-parallel-size",
            str(self.config.vllm_tensor_parallel_size),
        ]

    def run_cmd(self, cmd: list[str]) -> bool:
        if self.config.dry_run:
            print(f"[dry-run] {command_text(cmd)}")
            return True

        primary = self.resolve_cuda_visible_devices("primary", ())
        if not primary:
            print(f"[error] Could not select a CUDA device for command: {command_text(cmd)}", file=sys.stderr)
            return False

        if not self.config.cuda_oom_fallback:
            print(f"[run] {command_text(cmd)}")
            return subprocess.run(cmd, env={**self.env, "CUDA_VISIBLE_DEVICES": primary}).returncode == 0

        print(f"[run] CUDA_VISIBLE_DEVICES={primary} {command_text(cmd)}")
        with tempfile.NamedTemporaryFile("w+", prefix="run_all_tests_cuda_try.", delete=False) as log:
            log_path = Path(log.name)
        try:
            proc = subprocess.Popen(
                cmd,
                env={**self.env, "CUDA_VISIBLE_DEVICES": primary},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            with log_path.open("w") as log_file:
                for line in proc.stdout:
                    print(line, end="")
                    log_file.write(line)
            return_code = proc.wait()
            if return_code == 0:
                return True
            log_text = log_path.read_text(errors="ignore")
            if not OOM_RE.search(log_text):
                return False
        finally:
            try:
                log_path.unlink()
            except FileNotFoundError:
                pass

        fallback = self.resolve_cuda_visible_devices("fallback", (primary,))
        if not fallback:
            print(
                f"[warn] CUDA OOM on CUDA_VISIBLE_DEVICES={primary}; "
                "no fallback CUDA device available",
                file=sys.stderr,
            )
            return False

        print(
            f"[warn] CUDA OOM on CUDA_VISIBLE_DEVICES={primary}; "
            f"retrying with CUDA_VISIBLE_DEVICES={fallback}",
            file=sys.stderr,
        )
        print(f"[run] CUDA_VISIBLE_DEVICES={fallback} {command_text(cmd)}")
        return subprocess.run(cmd, env={**self.env, "CUDA_VISIBLE_DEVICES": fallback}).returncode == 0

    def metadecode_task(self, benchmark: str) -> str:
        if benchmark == "gsm_symbolic":
            return "Solve math word problems step by step, writing each arithmetic computation inside << >> delimiters."
        if benchmark == "spider":
            return "Generate a single valid SQL query that answers each question using the provided schema context."
        if benchmark == "smiles":
            return "Generate valid SMILES strings that match the requested molecular class while maintaining parser-valid output."
        return "Generate parser-valid benchmark answers."

    def resolve_gen_profile(self, profile: str) -> tuple[str, str]:
        bedrock_model = self.env.get(
            "BEDROCK_GENERATION_MODEL",
            self.env.get("AWS_BEDROCK_GENERATION_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        )
        opus_model = self.env.get(
            "BEDROCK_OPUS_MODEL",
            self.env.get("BEDROCK_PROFILE_OPUS", "us.anthropic.claude-opus-4-1-20250514-v1:0"),
        )
        openai_gpt = self.env.get("OPENAI_GENERATION_MODEL", "gpt-5.4")
        if profile == "gpt5.4":
            return "openai", openai_gpt
        if profile == "opus4.7":
            return "bedrock", opus_model
        if profile == "gemini-pro":
            model = self.env.get("GEMINI_BEDROCK_MODEL")
            if not model:
                raise SystemExit("Set GEMINI_BEDROCK_MODEL for gemini-pro (partner-owned wiring)")
            return "bedrock", model
        if profile == "bedrock":
            return "bedrock", bedrock_model
        if profile.startswith("bedrock:"):
            return "bedrock", profile.removeprefix("bedrock:")
        return "bedrock", profile

    def baseline_case_key(
        self,
        strategy: str,
        model_slug: str,
        benchmark_key: str,
        token_budget: str,
        max_steps: str,
    ) -> tuple[str, str, str, str, str]:
        return strategy, model_slug, benchmark_key, token_budget, max_steps

    def baseline_json_complete(self, path: Path) -> bool:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return False
        answers = payload.get("answers")
        if not isinstance(answers, list) or not answers:
            return False
        return all(isinstance(row, dict) and "generated_answer" in row for row in answers)

    def baseline_json_matches_strategy(self, path: Path, strategy: str) -> bool:
        if strategy != "crane":
            return True
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return False
        adapter = payload.get("metrics", {}).get("adapter")
        return adapter != "crane_shared_evaluator"

    def baseline_json_usable(self, path: Path, strategy: str) -> bool:
        return (
            path.is_file()
            and path.stat().st_size > 20
            and self.baseline_json_complete(path)
            and self.baseline_json_matches_strategy(path, strategy)
        )

    def benchmark_key(self, benchmark: str, smiles_class: str = "") -> str:
        if benchmark == "smiles":
            return f"{benchmark}__class_{slugify(smiles_class)}"
        return benchmark

    def fixed_baseline_path(
        self,
        strategy: str,
        eval_model: str,
        benchmark: str,
        token_budget: str,
        max_steps: str,
        smiles_class: str = "",
    ) -> Path:
        model_slug = slugify(eval_model)
        key = self.benchmark_key(benchmark, smiles_class)
        return (
            self.config.baseline_output_dir
            / strategy
            / model_slug
            / f"{key}__tb{token_budget}__ms{max_steps}.json"
        )

    def best_csd_baseline_targets(
        self,
        benchmark: str,
        eval_model: str,
        token_budget: str,
        max_steps: str,
        smiles_class: str = "",
    ) -> tuple[float, str, str, str, float, str, str, str]:
        best_accuracy: tuple[float, str, str, str] | None = None
        best_syntax: tuple[float, str, str, str] | None = None
        for strategy in CSD_TARGET_STRATEGIES:
            path = self.fixed_baseline_path(
                strategy, eval_model, benchmark, token_budget, max_steps, smiles_class
            )
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            answers = payload.get("answers")
            if not isinstance(answers, list) or not answers:
                continue
            if not all(isinstance(row, dict) and "generated_answer" in row for row in answers):
                continue
            if strategy == "crane" and payload.get("metrics", {}).get("adapter") == "crane_shared_evaluator":
                continue
            accuracy = payload.get("accuracy")
            if isinstance(accuracy, (int, float)):
                candidate = (float(accuracy), strategy, str(path), f"{float(accuracy):.1%}")
                if best_accuracy is None or candidate[0] > best_accuracy[0]:
                    best_accuracy = candidate
            syntax_rate = payload.get("syntax_rate")
            if isinstance(syntax_rate, (int, float)):
                candidate = (float(syntax_rate), strategy, str(path), f"{float(syntax_rate):.1%}")
                if best_syntax is None or candidate[0] > best_syntax[0]:
                    best_syntax = candidate

        if best_accuracy is None:
            best_accuracy = (0.0, "none", "", "0.0%")
        if best_syntax is None:
            best_syntax = (0.0, "none", "", "0.0%")
        return (*best_accuracy, *best_syntax)

    def ensure_csd_target_baselines(
        self,
        benchmark: str,
        eval_model: str,
        token_budget: str,
        max_steps: str,
        smiles_class: str = "",
    ) -> None:
        for strategy in CSD_TARGET_STRATEGIES:
            ok = self.run_fixed_strategy_case(
                strategy, benchmark, eval_model, token_budget, max_steps, smiles_class
            )
            if not ok:
                print(
                    f"[warn] Could not prepare {strategy} baseline for "
                    f"benchmark={benchmark} eval_model={eval_model} "
                    f"token_budget={token_budget} max_steps={max_steps} "
                    f"smiles_class={smiles_class or '<none>'}",
                    file=sys.stderr,
                )

    def run_fixed_strategy_case(
        self,
        strategy: str,
        benchmark: str,
        eval_model: str,
        token_budget: str,
        max_steps: str,
        smiles_class: str = "",
    ) -> bool:
        if benchmark == "smiles" and not smiles_class:
            print("Internal error: SMILES fixed-strategy run requires a class.", file=sys.stderr)
            return False

        model_slug = slugify(eval_model)
        key = self.benchmark_key(benchmark, smiles_class)
        out_json = self.fixed_baseline_path(
            strategy, eval_model, benchmark, token_budget, max_steps, smiles_class
        )
        out_json.parent.mkdir(parents=True, exist_ok=True)
        case_key = self.baseline_case_key(strategy, model_slug, key, token_budget, max_steps)
        allow_cache_reuse = (
            self.config.baseline_cache_mode == "reuse" or case_key in self.prepared_baselines
        )

        if allow_cache_reuse and self.baseline_json_usable(out_json, strategy):
            self.prepared_baselines.add(case_key)
            print(f"[skip] {out_json} already exists ({line_count(out_json)} lines). Delete it to re-run.")
            return True
        if out_json.exists():
            if self.config.baseline_cache_mode == "refresh" and case_key not in self.prepared_baselines:
                print(f"[rerun] {out_json} exists but --recompute-baselines was requested.")
            else:
                print(f"[rerun] {out_json} exists but is incomplete, corrupt, or from an obsolete adapter.")

        cmd = [
            "python",
            "-m",
            "synthesis.evaluate.run_legacy_fixed_strategy",
            "--strategy",
            strategy,
            "--dataset",
            benchmark,
            "--eval-model",
            eval_model,
            "--eval-backend",
            self.config.eval_backend,
            "--device",
            self.config.device,
            "--eval-sample-size",
            self.evaluation_sample_size(benchmark),
            "--eval-max-steps",
            max_steps,
            "--eval-step-token-budget",
            token_budget,
            "--vllm-gpu-memory-utilization",
            self.config.vllm_gpu_memory_utilization,
            "--output-json",
            str(out_json),
        ]
        self.add_vllm_parallel_flags(cmd)
        if self.config.dafny_path:
            cmd += ["--dafny-path", self.config.dafny_path]
        self.add_evaluation_split_flags(cmd, benchmark)
        if benchmark == "smiles":
            cmd += [
                "--smiles-classes",
                smiles_class,
                "--smiles-samples-per-class",
                self.evaluation_sample_size(benchmark),
            ]

        if self.run_cmd(cmd):
            self.prepared_baselines.add(case_key)
            return True
        return False

    def run_fixed_strategy_cases(
        self,
        strategy: str,
        benchmark: str,
        eval_model: str,
        token_budget: str,
        max_steps: str,
    ) -> None:
        if benchmark == "smiles":
            for smiles_class in self.config.smiles_classes:
                self.run_fixed_strategy_case(strategy, benchmark, eval_model, token_budget, max_steps, smiles_class)
            return
        self.run_fixed_strategy_case(strategy, benchmark, eval_model, token_budget, max_steps)

    def metadecode_final_eval_command(
        self,
        compiled_module: Path,
        out_json: Path,
        benchmark: str,
        eval_model: str,
        token_budget: str,
        max_steps: str,
        smiles_class: str = "",
    ) -> list[str]:
        cmd = [
            "python",
            "-m",
            "synthesis.scripts.reevaluate_compiled_csd",
            str(compiled_module),
            "--dataset",
            benchmark,
            "--eval-model",
            eval_model,
            "--eval-backend",
            self.config.eval_backend,
            "--device",
            self.config.device,
            "--sample-size",
            self.evaluation_sample_size(benchmark),
            "--max-steps",
            max_steps,
            "--step-token-budget",
            token_budget,
            "--vllm-gpu-memory-utilization",
            self.config.vllm_gpu_memory_utilization,
            "--output-json",
            str(out_json),
        ]
        self.add_vllm_parallel_flags(cmd)
        self.add_evaluation_split_flags(cmd, benchmark)
        if benchmark == "smiles":
            cmd += ["--smiles-classes", smiles_class]
        return cmd

    def run_metadecode_case(
        self,
        benchmark: str,
        eval_model: str,
        token_budget: str,
        synth_iter: str,
        gen_profile: str,
        max_steps: str,
        smiles_class: str = "",
    ) -> bool:
        if benchmark == "smiles" and not smiles_class:
            print("Internal error: SMILES metadecode run requires a class.", file=sys.stderr)
            return False

        backend, generation_model = self.resolve_gen_profile(gen_profile)
        model_slug = slugify(eval_model)
        gen_slug = slugify(gen_profile)
        class_suffix = f"_class_{slugify(smiles_class)}" if benchmark == "smiles" else ""
        key = self.benchmark_key(benchmark, smiles_class)
        run_name = (
            f"metadecode_{benchmark}_{model_slug}_{gen_slug}_"
            f"iter{synth_iter}_tb{token_budget}_ms{max_steps}{class_suffix}"
        )
        task = self.metadecode_task(benchmark)

        self.ensure_csd_target_baselines(benchmark, eval_model, token_budget, max_steps, smiles_class)
        (
            target_accuracy,
            target_strategy,
            _target_path,
            target_percent,
            target_syntax,
            target_syntax_strategy,
            _target_syntax_path,
            target_syntax_percent,
        ) = self.best_csd_baseline_targets(benchmark, eval_model, token_budget, max_steps, smiles_class)

        if target_strategy == "none" and target_syntax_strategy == "none":
            print(
                f"[target] metadecode {key}/{model_slug} tb{token_budget} ms{max_steps}: "
                "no valid CRANE/IterGen/CARS baseline found; passing --min-accuracy 0.0 --min-syntax-rate 0.0"
            )
        else:
            print(
                f"[target] metadecode {key}/{model_slug} tb{token_budget} ms{max_steps}: "
                f"best CSD baseline accuracy {target_strategy}={target_percent}, "
                f"syntax {target_syntax_strategy}={target_syntax_percent}; "
                f"passing --min-accuracy {target_accuracy:.12g} --min-syntax-rate {target_syntax:.12g}"
            )

        cmd = [
            "python",
            "-m",
            "synthesis.run_synthesis",
            "--task",
            task,
            "--dataset",
            benchmark,
            "--generation-model",
            generation_model,
            "--generation-backend",
            backend,
            "--eval-model",
            eval_model,
            "--eval-backend",
            self.config.eval_backend,
            "--max-iterations",
            synth_iter,
            "--output-name",
            run_name,
            "--min-accuracy",
            f"{target_accuracy:.12g}",
            "--min-syntax-rate",
            f"{target_syntax:.12g}",
            "--eval-sample-size",
            self.generation_sample_size(benchmark),
            "--eval-max-steps",
            max_steps,
            "--eval-step-token-budget",
            token_budget,
            "--eval-max-seconds-per-example",
            self.config.eval_max_seconds_per_example,
            "--eval-min-examples-before-threshold-stop",
            self.config.eval_min_examples_before_threshold_stop,
            "--vllm-gpu-memory-utilization",
            self.config.vllm_gpu_memory_utilization,
            "--device",
            self.config.device,
            "--output-dir",
            str(self.config.generated_output_dir),
        ]
        self.add_vllm_parallel_flags(cmd)
        self.add_generation_split_flags(cmd, benchmark)
        if benchmark == "smiles":
            cmd += [
                "--smiles-samples-per-class",
                self.generation_sample_size(benchmark),
                "--smiles-classes",
                smiles_class,
            ]
        if self.config.dafny_path:
            cmd += ["--dafny-path", self.config.dafny_path]

        if not self.run_cmd(cmd):
            print(
                f"[warn] Metadecode synthesis failed for benchmark={benchmark} "
                f"eval_model={eval_model} token_budget={token_budget} iter={synth_iter} "
                f"gen={gen_profile} max_steps={max_steps}",
                file=sys.stderr,
            )
            return True

        out_json = (
            self.config.baseline_output_dir
            / "metadecode"
            / model_slug
            / f"{key}__tb{token_budget}__ms{max_steps}__gen{gen_slug}__iter{synth_iter}.json"
        )
        out_json.parent.mkdir(parents=True, exist_ok=True)

        if self.config.dry_run:
            final_cmd = self.metadecode_final_eval_command(
                Path(f"<{self.config.generated_output_dir}/.../python/{run_name}/GeneratedCSD.py>"),
                out_json,
                benchmark,
                eval_model,
                token_budget,
                max_steps,
                smiles_class,
            )
            print(f"[dry-run] {command_text(final_cmd)}")
            return True

        latest_file = self.config.generated_output_dir / "latest_run.txt"
        if not latest_file.is_file():
            print(f"[warn] No latest run file found after synthesis: {latest_file}", file=sys.stderr)
            return True
        run_dir = Path(latest_file.read_text().strip())
        success_report = run_dir / "results" / "success_report.json"
        if not success_report.is_file():
            print(f"[warn] No success report found for run: {run_dir}", file=sys.stderr)
            return True
        try:
            report = json.loads(success_report.read_text())
        except Exception as exc:
            print(f"[warn] Could not read success report {success_report}: {exc}", file=sys.stderr)
            return True
        compiled_dir = report.get("compiled_dir")
        if not compiled_dir:
            print(f"[warn] Success report does not contain compiled_dir: {success_report}", file=sys.stderr)
            return True
        compiled_module = Path(compiled_dir) / "GeneratedCSD.py"
        if not compiled_module.is_file():
            print(f"[warn] Compiled GeneratedCSD.py not found: {compiled_module}", file=sys.stderr)
            return True
        self.run_cmd(
            self.metadecode_final_eval_command(
                compiled_module,
                out_json,
                benchmark,
                eval_model,
                token_budget,
                max_steps,
                smiles_class,
            )
        )
        return True

    def run_metadecode_cases(
        self,
        benchmark: str,
        eval_model: str,
        token_budget: str,
        synth_iter: str,
        gen_profile: str,
        max_steps: str,
    ) -> None:
        if benchmark == "smiles":
            for smiles_class in self.config.smiles_classes:
                self.run_metadecode_case(
                    benchmark, eval_model, token_budget, synth_iter, gen_profile, max_steps, smiles_class
                )
            return
        self.run_metadecode_case(benchmark, eval_model, token_budget, synth_iter, gen_profile, max_steps)

    def print_matrix_header(self) -> None:
        print("=== run_all_tests matrix ===")
        print(f"models: {' '.join(self.config.models)}")
        print(f"benchmarks: {' '.join(self.config.benchmarks)}")
        print(f"strategies: {' '.join(self.config.strategies)}")
        print(f"token budgets: {' '.join(self.config.token_budgets)}")
        print(f"step budgets (ablation): {' '.join(self.config.step_budgets)}")
        print(f"synthesis iters (metadecode): {' '.join(self.config.synth_iters)}")
        print(f"generation models (metadecode): {' '.join(self.config.gen_models)}")
        print(f"SMILES classes: {' '.join(self.config.smiles_classes)}")
        print(f"eval max steps (main): {self.config.eval_max_steps}")
        print(
            "split policy: "
            f"GSM generation={self.config.gsm_generation_sample_size}/eval={self.config.gsm_eval_sample_size}; "
            f"other generation={self.config.generation_sample_size}/eval={self.config.eval_sample_size}"
        )
        if self.config.gsm_split_file:
            print(f"GSM split file: {self.config.gsm_split_file}")
        if self.config.spider_split_file:
            print(f"Spider split file: {self.config.spider_split_file}")
        print(
            f"baseline cache mode: {self.config.baseline_cache_mode} "
            "(reuse=skip complete JSONs, refresh=recompute fixed baselines)"
        )
        print("")

    def run_main_matrix(self) -> None:
        if self.config.skip_main:
            return
        print("=== Phase 1: Main experiment matrix ===")
        for eval_model in self.config.models:
            for raw_benchmark in self.config.benchmarks:
                benchmark = normalize_benchmark(raw_benchmark)
                for strategy in self.config.strategies:
                    if strategy == "metadecode":
                        self.run_metadecode_cases(
                            benchmark,
                            eval_model,
                            self.config.token_budgets[0],
                            self.config.synth_iters[-1],
                            self.config.gen_models[0],
                            self.config.eval_max_steps,
                        )
                    else:
                        self.run_fixed_strategy_cases(
                            strategy,
                            benchmark,
                            eval_model,
                            self.config.token_budgets[0],
                            self.config.eval_max_steps,
                        )
        print("=== Phase 1 complete ===")

    def run_ablation_e_case(
        self,
        benchmark: str,
        eval_model: str,
        beam_size: str,
        mask_flag: str,
        policy: str,
        smiles_class: str = "",
    ) -> None:
        task = self.metadecode_task(benchmark)
        class_suffix = f"_class_{slugify(smiles_class)}" if benchmark == "smiles" else ""
        run_name = f"ablat_beam{beam_size}_{'mask_off' if mask_flag == '--no-adaptive-helper-mask' else 'mask_on'}_{policy}_{benchmark}{class_suffix}"
        backend, generation_model = self.resolve_gen_profile("gpt5.4")
        token_budget = self.config.token_budgets[0]
        self.ensure_csd_target_baselines(benchmark, eval_model, token_budget, self.config.eval_max_steps, smiles_class)
        (
            target_accuracy,
            target_strategy,
            _target_path,
            target_percent,
            target_syntax,
            target_syntax_strategy,
            _target_syntax_path,
            target_syntax_percent,
        ) = self.best_csd_baseline_targets(benchmark, eval_model, token_budget, self.config.eval_max_steps, smiles_class)

        if target_strategy == "none" and target_syntax_strategy == "none":
            print(
                f"[target] metadecode {benchmark}{class_suffix}/{slugify(eval_model)} "
                f"tb{token_budget} ms{self.config.eval_max_steps}: no valid CRANE/IterGen/CARS baseline found; "
                "passing --min-accuracy 0.0 --min-syntax-rate 0.0"
            )
        else:
            print(
                f"[target] metadecode {benchmark}{class_suffix}/{slugify(eval_model)} "
                f"tb{token_budget} ms{self.config.eval_max_steps}: best CSD baseline accuracy "
                f"{target_strategy}={target_percent}, syntax {target_syntax_strategy}={target_syntax_percent}; "
                f"passing --min-accuracy {target_accuracy:.12g} --min-syntax-rate {target_syntax:.12g}"
            )

        cmd = [
            "python",
            "-m",
            "synthesis.run_synthesis",
            "--task",
            task,
            "--dataset",
            benchmark,
            "--generation-backend",
            backend,
            "--generation-model",
            generation_model,
            "--eval-model",
            eval_model,
            "--eval-backend",
            self.config.eval_backend,
            "--max-iterations",
            self.config.synth_iters[-1],
            "--output-name",
            run_name,
            "--min-accuracy",
            f"{target_accuracy:.12g}",
            "--min-syntax-rate",
            f"{target_syntax:.12g}",
            "--eval-sample-size",
            self.generation_sample_size(benchmark),
            "--eval-max-steps",
            self.config.eval_max_steps,
            "--eval-step-token-budget",
            token_budget,
            "--eval-max-seconds-per-example",
            self.config.eval_max_seconds_per_example,
            "--eval-min-examples-before-threshold-stop",
            self.config.eval_min_examples_before_threshold_stop,
            "--vllm-gpu-memory-utilization",
            self.config.vllm_gpu_memory_utilization,
            "--device",
            self.config.device,
            "--output-dir",
            str(self.config.generated_output_dir),
            "--refinement-beam-size",
            beam_size,
            mask_flag,
            "--helper-selection-policy",
            policy,
        ]
        self.add_vllm_parallel_flags(cmd)
        self.add_generation_split_flags(cmd, benchmark)
        if benchmark == "smiles":
            cmd += [
                "--smiles-samples-per-class",
                self.generation_sample_size(benchmark),
                "--smiles-classes",
                smiles_class,
            ]
        if self.config.dafny_path:
            cmd += ["--dafny-path", self.config.dafny_path]
        self.run_cmd(cmd)

    def run_ablations(self) -> None:
        if self.config.skip_ablations:
            return
        print("")
        print("=== Phase 2: Ablation studies ===")
        ablation_model = "Qwen/Qwen2.5-Coder-7B-Instruct"

        print("--- Ablation A: Step budget ---")
        for raw_benchmark in ("gsm", "spider", "smiles"):
            benchmark = normalize_benchmark(raw_benchmark)
            for step_budget in self.config.step_budgets:
                for strategy in ("gcd", "crane", "itergen", "cars", "metadecode"):
                    if strategy == "metadecode":
                        self.run_metadecode_cases(
                            benchmark,
                            ablation_model,
                            self.config.token_budgets[0],
                            self.config.synth_iters[-1],
                            self.config.gen_models[0],
                            step_budget,
                        )
                    else:
                        self.run_fixed_strategy_cases(
                            strategy, benchmark, ablation_model, self.config.token_budgets[0], step_budget
                        )

        print("--- Ablation B: Synthesis iterations ---")
        for raw_benchmark in ("gsm", "spider", "smiles"):
            benchmark = normalize_benchmark(raw_benchmark)
            for synth_iter in self.config.synth_iters:
                self.run_metadecode_cases(
                    benchmark,
                    ablation_model,
                    self.config.token_budgets[0],
                    synth_iter,
                    self.config.gen_models[0],
                    self.config.eval_max_steps,
                )

        print("--- Ablation C: Synthesizer model ---")
        for raw_benchmark in ("gsm", "spider", "smiles"):
            benchmark = normalize_benchmark(raw_benchmark)
            for gen_profile in self.config.gen_models:
                self.run_metadecode_cases(
                    benchmark,
                    ablation_model,
                    self.config.token_budgets[0],
                    self.config.synth_iters[-1],
                    gen_profile,
                    self.config.eval_max_steps,
                )

        print("--- Ablation D: Per-step token budget ---")
        for raw_benchmark in ("gsm", "spider", "smiles"):
            benchmark = normalize_benchmark(raw_benchmark)
            for token_budget in self.config.token_budgets:
                for strategy in ("gcd", "crane", "itergen", "cars", "metadecode"):
                    if strategy == "metadecode":
                        self.run_metadecode_cases(
                            benchmark,
                            ablation_model,
                            token_budget,
                            self.config.synth_iters[-1],
                            self.config.gen_models[0],
                            self.config.eval_max_steps,
                        )
                    else:
                        self.run_fixed_strategy_cases(
                            strategy, benchmark, ablation_model, token_budget, self.config.eval_max_steps
                        )

        print("--- Ablation E: Beam refinement x adaptive helper masking x helper selection policy ---")
        for raw_benchmark in ("gsm", "spider", "smiles"):
            benchmark = normalize_benchmark(raw_benchmark)
            for beam_size in ("1", "2", "4"):
                for mask_flag in ("--adaptive-helper-mask", "--no-adaptive-helper-mask"):
                    for policy in ("utility", "bandit"):
                        smiles_classes = self.config.smiles_classes if benchmark == "smiles" else [""]
                        for smiles_class in smiles_classes:
                            self.run_ablation_e_case(
                                benchmark, ablation_model, beam_size, mask_flag, policy, smiles_class
                            )
        print("=== Phase 2 complete ===")

    def run(self) -> int:
        self.config.generated_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.baseline_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.ablation_output_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_split_manifests()
        if not self.configure_cuda_devices():
            return 1
        self.print_matrix_header()
        self.run_main_matrix()
        self.run_ablations()
        print("")
        print("All requested matrix jobs completed.")
        return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strategy x model x benchmark matrix, plus ablations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Outputs:\n"
            "- Synthesis runs: generated output dir\n"
            "- Baseline JSONs: baseline output dir\n"
            "- Ablation JSONs: ablation output dir\n"
        ),
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--strategies", default=DEFAULT_STRATEGIES)
    parser.add_argument("--token-budgets", default=DEFAULT_TOKEN_BUDGETS)
    parser.add_argument("--step-budgets", default=DEFAULT_STEP_BUDGETS)
    parser.add_argument("--synthesis-iterations", default=DEFAULT_SYNTH_ITERS)
    parser.add_argument("--generation-models", default=DEFAULT_GEN_MODELS)
    parser.add_argument("--smiles-classes", "--smiles-class", default=DEFAULT_SMILES_CLASSES)
    parser.add_argument("--eval-backend", default="vllm")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--generation-sample-size", default="50")
    parser.add_argument("--eval-sample-size", default="100")
    parser.add_argument("--gsm-generation-sample-size", default="100")
    parser.add_argument("--gsm-eval-sample-size", default="100")
    parser.add_argument("--eval-max-steps", default="900")
    parser.add_argument(
        "--eval-max-seconds-per-example",
        default="90",
        help="Per-example wall-clock timeout for synthesis evaluation (seconds). "
        "Wired through to synthesis.run_synthesis. Default: 90.",
    )
    parser.add_argument(
        "--eval-min-examples-before-threshold-stop",
        default="25",
        help="Minimum number of evaluated examples before threshold-impossible "
        "early stops can fire during synthesis. Default: 25.",
    )
    parser.add_argument(
        "--gsm-split-file",
        default=os.environ.get("CSD_GSM_SPLIT_FILE", str(DEFAULT_GSM_SPLIT_FILE)),
        help="Stratified GSM-Symbolic manifest (default: environment/benchmark_splits/gsm_symbolic_crane_proportional.json)",
    )
    parser.add_argument(
        "--spider-split-file",
        default=os.environ.get("CSD_SPIDER_SPLIT_FILE", str(DEFAULT_SPIDER_SPLIT_FILE)),
        help="Stratified Spider manifest (default: environment/benchmark_splits/spider_dev_proportional.json)",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        default=os.environ.get("VAS_VLLM_GPU_MEMORY_UTILIZATION", "0.80"),
        help="vLLM GPU memory fraction (default: 0.80; override via VAS_VLLM_GPU_MEMORY_UTILIZATION)",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=int(os.environ.get("VAS_VLLM_TENSOR_PARALLEL_SIZE", "1")),
        help="vLLM tensor parallel size (default: 1; capped by VAS_MAX_CUDA_DEVICES)",
    )
    parser.add_argument("--dafny-path", default=os.environ.get("DAFNY_PATH", ""))
    parser.add_argument("--generated-output-dir", default=os.environ.get("CSD_OUTPUT_DIR", "outputs/generated"))
    parser.add_argument("--baseline-output-dir", default=os.environ.get("CSD_BASELINE_OUTPUT_DIR", "outputs/baselines"))
    parser.add_argument("--ablation-output-dir", default=os.environ.get("CSD_ABLATION_OUTPUT_DIR", "outputs/ablations"))
    parser.add_argument(
        "--recompute-baselines",
        dest="baseline_cache_mode",
        action="store_const",
        const="refresh",
    )
    parser.add_argument(
        "--reuse-baselines",
        dest="baseline_cache_mode",
        action="store_const",
        const="reuse",
    )
    parser.set_defaults(baseline_cache_mode="reuse")
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def normalize_smiles_classes_for_cli(raw: str) -> list[str]:
    try:
        return normalize_smiles_classes(
            ",".join(csv_list(raw)),
            dedupe=True,
            require_non_empty=True,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def configure_conda_environment(root: Path) -> tuple[Path, dict[str, str]]:
    default_env = Path("/apps/conda/advayth2/envs/advayth2")
    conda_env_path = Path(
        os.environ.get("VAS_CONDA_ENV")
        or os.environ.get("VAS_RDKIT_CONDA_ENV")
        or str(default_env)
    )
    python_path = conda_env_path / "bin" / "python"
    if not python_path.exists():
        print(f"conda environment python not found: {python_path}", file=sys.stderr)
        raise SystemExit(1)

    env = os.environ.copy()
    env["CONDA_PREFIX"] = str(conda_env_path)
    env["PATH"] = f"{conda_env_path / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    lib_dir = conda_env_path / "lib"
    if lib_dir.is_dir():
        env["LD_LIBRARY_PATH"] = f"{lib_dir}{os.pathsep}{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else str(lib_dir)
    env["PYTHONUNBUFFERED"] = "1"

    rdkit_check = subprocess.run(
        [str(python_path), "-c", "import rdkit"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )
    if rdkit_check.returncode != 0:
        sys.stderr.write(rdkit_check.stderr)
        print(f"failed to import rdkit in conda environment: {conda_env_path}", file=sys.stderr)
        raise SystemExit(rdkit_check.returncode)

    print(f"[env] using conda environment: {conda_env_path}")
    return conda_env_path, env


def build_config(args: argparse.Namespace, conda_env_path: Path) -> Config:
    from synthesis.evaluate.benchmarks.common.model_utils import resolve_vllm_tensor_parallel_size

    dafny_path = args.dafny_path
    if not dafny_path and (ROOT_DIR / "dafny" / "dafny").is_file():
        dafny_path = str(ROOT_DIR / "dafny" / "dafny")

    baseline_cache_mode = args.baseline_cache_mode
    if baseline_cache_mode not in {"reuse", "refresh"}:
        raise SystemExit(f"Invalid baseline cache mode: {baseline_cache_mode} (expected reuse or refresh)")

    return Config(
        models=csv_list(args.models),
        benchmarks=csv_list(args.benchmarks),
        strategies=csv_list(args.strategies),
        token_budgets=csv_list(args.token_budgets),
        synth_iters=csv_list(args.synthesis_iterations),
        gen_models=csv_list(args.generation_models),
        step_budgets=csv_list(args.step_budgets),
        smiles_classes=normalize_smiles_classes_for_cli(args.smiles_classes),
        eval_backend=args.eval_backend,
        device=args.device,
        generation_sample_size=str(args.generation_sample_size),
        eval_sample_size=str(args.eval_sample_size),
        gsm_generation_sample_size=str(args.gsm_generation_sample_size),
        gsm_eval_sample_size=str(args.gsm_eval_sample_size),
        eval_max_steps=str(args.eval_max_steps),
        eval_max_seconds_per_example=str(args.eval_max_seconds_per_example),
        eval_min_examples_before_threshold_stop=str(args.eval_min_examples_before_threshold_stop),
        vllm_gpu_memory_utilization=str(args.vllm_gpu_memory_utilization),
        vllm_tensor_parallel_size=resolve_vllm_tensor_parallel_size(args.vllm_tensor_parallel_size),
        dafny_path=dafny_path,
        generated_output_dir=Path(args.generated_output_dir),
        baseline_output_dir=Path(args.baseline_output_dir),
        ablation_output_dir=Path(args.ablation_output_dir),
        baseline_cache_mode=baseline_cache_mode,
        gsm_split_file=args.gsm_split_file,
        spider_split_file=args.spider_split_file,
        dry_run=args.dry_run,
        skip_main=args.skip_main,
        skip_ablations=args.skip_ablations,
        conda_env_path=conda_env_path,
        cuda_devices=os.environ.get("RUN_ALL_TESTS_CUDA_DEVICES", "auto"),
        cuda_oom_fallback=os.environ.get("RUN_ALL_TESTS_CUDA_OOM_FALLBACK", "auto"),
        free_gpu_max_used_mb=int(os.environ.get("RUN_ALL_TESTS_FREE_GPU_MAX_USED_MB", "1024")),
        gpu_wait_seconds=int(os.environ.get("RUN_ALL_TESTS_GPU_WAIT_SECONDS", "60")),
        gpu_wait_timeout_seconds=int(os.environ.get("RUN_ALL_TESTS_GPU_WAIT_TIMEOUT_SECONDS", "0")),
    )


def main(argv: list[str] | None = None) -> int:
    os.chdir(ROOT_DIR)
    load_env_file(ROOT_DIR / "synthesis" / ".env")
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.eval_backend == "vllm":
        from synthesis.evaluate.benchmarks.common.model_utils import configure_vllm_multiprocessing

        configure_vllm_multiprocessing()
    conda_env_path, env = configure_conda_environment(ROOT_DIR)
    config = build_config(args, conda_env_path)
    return Runner(config=config, env=env).run()


if __name__ == "__main__":
    raise SystemExit(main())
