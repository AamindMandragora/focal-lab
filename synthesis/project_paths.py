"""Canonical artifact paths grouped by model, benchmark, then strategy."""

from __future__ import annotations

import shutil
from pathlib import Path


def slugify(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_").replace("-", "_")


def normalize_benchmark(value: str) -> str:
    return "gsm_symbolic" if value == "gsm" else value


def benchmark_key(benchmark: str, smiles_class: str = "") -> str:
    bench = normalize_benchmark(benchmark)
    if bench == "smiles" and smiles_class:
        return f"smiles__class_{smiles_class}"
    return bench


def synthesis_strategy_from_output_name(output_name: str) -> str:
    name = (output_name or "").strip().lower()
    if name.startswith("metadecode") or name.startswith("ablat_"):
        return "metadecode"
    return "synthesis"


def baseline_params_stem(
    token_budget: str,
    max_steps: str,
    *,
    strategy: str,
    benchmark: str = "",
    rs_search_steps: str = "",
    cars_search_steps: str = "",
    gen_profile: str = "",
    synth_iter: str = "",
) -> str:
    if strategy == "metadecode":
        return (
            f"gen{slugify(gen_profile)}__iter{synth_iter}"
            f"__tb{token_budget}__ms{max_steps}"
        )
    stem = f"tb{token_budget}__ms{max_steps}"
    if strategy == "rs" and rs_search_steps:
        stem += f"__rs{rs_search_steps}"
    elif strategy == "cars" and normalize_benchmark(benchmark) == "smiles" and cars_search_steps:
        stem += f"__cs{cars_search_steps}"
    return stem


def baseline_json_path(
    baseline_root: Path,
    *,
    eval_model: str,
    benchmark: str,
    strategy: str,
    token_budget: str,
    max_steps: str,
    smiles_class: str = "",
    rs_search_steps: str = "",
    cars_search_steps: str = "",
    gen_profile: str = "",
    synth_iter: str = "",
) -> Path:
    """``outputs/baselines/<model>/<benchmark>/<strategy>/<params>.json``."""
    params = baseline_params_stem(
        token_budget,
        max_steps,
        strategy=strategy,
        benchmark=benchmark,
        rs_search_steps=rs_search_steps,
        cars_search_steps=cars_search_steps,
        gen_profile=gen_profile,
        synth_iter=synth_iter,
    )
    return (
        baseline_root
        / slugify(eval_model)
        / benchmark_key(benchmark, smiles_class)
        / strategy
        / f"{params}.json"
    )


def legacy_baseline_json_stem(
    benchmark: str,
    token_budget: str,
    max_steps: str,
    *,
    strategy: str,
    smiles_class: str = "",
    rs_search_steps: str = "",
    cars_search_steps: str = "",
    gen_profile: str = "",
    synth_iter: str = "",
) -> str:
    """Pre-reorg filename stem: ``<benchmark_key>__tb*__ms*[suffix]``."""
    key = benchmark_key(benchmark, smiles_class)
    stem = f"{key}__tb{token_budget}__ms{max_steps}"
    if strategy == "rs" and rs_search_steps:
        stem += f"__rs{rs_search_steps}"
    elif strategy == "cars" and normalize_benchmark(benchmark) == "smiles" and cars_search_steps:
        stem += f"__cs{cars_search_steps}"
    elif strategy == "metadecode" and gen_profile:
        stem += f"__gen{slugify(gen_profile)}__iter{synth_iter}"
    return stem


def legacy_baseline_json_path(
    baseline_root: Path,
    *,
    eval_model: str,
    benchmark: str,
    strategy: str,
    token_budget: str,
    max_steps: str,
    smiles_class: str = "",
    rs_search_steps: str = "",
    cars_search_steps: str = "",
    gen_profile: str = "",
    synth_iter: str = "",
) -> Path:
    """Legacy layout: ``outputs/baselines/<strategy>/<model>/<stem>.json``."""
    if strategy == "metadecode":
        return (
            baseline_root
            / "metadecode"
            / slugify(eval_model)
            / f"{legacy_baseline_json_stem(benchmark, token_budget, max_steps, strategy=strategy, smiles_class=smiles_class, gen_profile=gen_profile, synth_iter=synth_iter)}.json"
        )
    stem = legacy_baseline_json_stem(
        benchmark,
        token_budget,
        max_steps,
        strategy=strategy,
        smiles_class=smiles_class,
        rs_search_steps=rs_search_steps,
        cars_search_steps=cars_search_steps,
    )
    return baseline_root / strategy / slugify(eval_model) / f"{stem}.json"


def resolve_baseline_json_path(
    baseline_root: Path,
    *,
    eval_model: str,
    benchmark: str,
    strategy: str,
    token_budget: str,
    max_steps: str,
    smiles_class: str = "",
    rs_search_steps: str = "",
    cars_search_steps: str = "",
    gen_profile: str = "",
    synth_iter: str = "",
) -> Path:
    """Return the canonical new path, falling back to legacy when only that exists."""
    new_path = baseline_json_path(
        baseline_root,
        eval_model=eval_model,
        benchmark=benchmark,
        strategy=strategy,
        token_budget=token_budget,
        max_steps=max_steps,
        smiles_class=smiles_class,
        rs_search_steps=rs_search_steps,
        cars_search_steps=cars_search_steps,
        gen_profile=gen_profile,
        synth_iter=synth_iter,
    )
    if new_path.is_file():
        return new_path
    legacy_path = legacy_baseline_json_path(
        baseline_root,
        eval_model=eval_model,
        benchmark=benchmark,
        strategy=strategy,
        token_budget=token_budget,
        max_steps=max_steps,
        smiles_class=smiles_class,
        rs_search_steps=rs_search_steps,
        cars_search_steps=cars_search_steps,
        gen_profile=gen_profile,
        synth_iter=synth_iter,
    )
    if legacy_path.is_file():
        return legacy_path
    return new_path


def generated_run_dir(
    generated_root: Path,
    *,
    eval_model: str,
    benchmark: str,
    strategy: str,
    output_name: str,
    run_id: str,
) -> Path:
    """``outputs/generated/<model>/<benchmark>/<strategy>/<output_name>_<run_id>/``."""
    return (
        generated_root
        / slugify(eval_model)
        / benchmark_key(benchmark)
        / strategy
        / f"{output_name}_{run_id}"
    )


def legacy_generated_run_dir(
    generated_root: Path,
    *,
    output_name: str,
    run_id: str,
) -> Path:
    return generated_root / f"{output_name}_{run_id}"


def prompt_log_dir(
    logs_root: Path,
    *,
    eval_model: str,
    benchmark: str,
    strategy: str,
    output_name: str,
    run_id: str,
) -> Path:
    """``logs/<model>/<benchmark>/<strategy>/<output_name>_<run_id>/``."""
    return (
        logs_root
        / slugify(eval_model)
        / benchmark_key(benchmark)
        / strategy
        / f"{output_name}_{run_id}"
    )


def iter_generated_run_dirs(
    generated_root: Path,
    *,
    eval_model: str | None = None,
    benchmark: str | None = None,
    strategy: str | None = None,
    name_prefix: str | None = None,
) -> list[Path]:
    """Discover run directories in the new nested layout (and legacy flat dirs)."""
    found: list[Path] = []

    def _maybe_add(path: Path) -> None:
        if not path.is_dir():
            return
        if name_prefix and not path.name.startswith(name_prefix):
            return
        found.append(path)

    if generated_root.is_dir():
        for entry in generated_root.iterdir():
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            # Legacy flat run dir at generated root.
            if (entry / "results").is_dir() or (entry / "dafny").is_dir():
                if eval_model is None and benchmark is None and strategy is None:
                    _maybe_add(entry)
                continue
            # New layout: model / benchmark / strategy / run
            if eval_model and entry.name != slugify(eval_model):
                continue
            for bench_dir in entry.iterdir():
                if not bench_dir.is_dir():
                    continue
                if benchmark and bench_dir.name != benchmark_key(benchmark):
                    continue
                for strat_dir in bench_dir.iterdir():
                    if not strat_dir.is_dir():
                        continue
                    if strategy and strat_dir.name != strategy:
                        continue
                    for run_dir in strat_dir.iterdir():
                        _maybe_add(run_dir)

    return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)


def migrate_baseline_file(old_path: Path, new_path: Path, *, dry_run: bool = False) -> bool:
    if not old_path.is_file() or new_path.is_file():
        return False
    if dry_run:
        return True
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_path), str(new_path))
    return True


def migrate_legacy_baseline_tree(baseline_root: Path, *, dry_run: bool = False) -> int:
    """Move ``strategy/model/*.json`` and ``metadecode/model/*.json`` into the new tree."""
    moved = 0
    if not baseline_root.is_dir():
        return moved

    for strategy_dir in baseline_root.iterdir():
        if not strategy_dir.is_dir():
            continue
        strategy = strategy_dir.name
        if strategy not in {
            "unconstrained",
            "gcd",
            "crane",
            "itergen",
            "cars",
            "rs",
            "metadecode",
        }:
            continue
        for model_dir in strategy_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_slug = model_dir.name
            for json_path in model_dir.glob("*.json"):
                stem = json_path.stem
                if strategy == "metadecode":
                    bench_key, token_budget, max_steps, rs_steps, cs_steps = _parse_legacy_stem(stem)
                    gen_profile = ""
                    synth_iter = ""
                    for part in stem.split("__"):
                        if part.startswith("gen"):
                            gen_profile = part[3:]
                        elif part.startswith("iter"):
                            synth_iter = part[4:]
                    if not bench_key or not token_budget or not max_steps or not gen_profile or not synth_iter:
                        continue
                    smiles_class = ""
                    if bench_key.startswith("smiles__class_"):
                        smiles_class = bench_key.split("smiles__class_", 1)[1]
                        benchmark = "smiles"
                    elif bench_key == "gsm_symbolic":
                        benchmark = "gsm_symbolic"
                    elif bench_key == "spider":
                        benchmark = "spider"
                    else:
                        continue
                    eval_model = _infer_model_from_slug(model_slug)
                    new_path = baseline_json_path(
                        baseline_root,
                        eval_model=eval_model,
                        benchmark=benchmark,
                        strategy="metadecode",
                        token_budget=token_budget,
                        max_steps=max_steps,
                        smiles_class=smiles_class,
                        gen_profile=gen_profile,
                        synth_iter=synth_iter,
                    )
                else:
                    bench_key, token_budget, max_steps, rs_steps, cs_steps = _parse_legacy_stem(
                        stem
                    )
                    if not bench_key:
                        continue
                    smiles_class = ""
                    if bench_key.startswith("smiles__class_"):
                        smiles_class = bench_key.split("smiles__class_", 1)[1]
                        benchmark = "smiles"
                    elif bench_key == "gsm_symbolic":
                        benchmark = "gsm_symbolic"
                    elif bench_key == "spider":
                        benchmark = "spider"
                    else:
                        continue
                    eval_model = _infer_model_from_slug(model_slug)
                    new_path = baseline_json_path(
                        baseline_root,
                        eval_model=eval_model,
                        benchmark=benchmark,
                        strategy=strategy,
                        token_budget=token_budget,
                        max_steps=max_steps,
                        smiles_class=smiles_class,
                        rs_search_steps=rs_steps or "",
                        cars_search_steps=cs_steps or "",
                    )
                if migrate_baseline_file(json_path, new_path, dry_run=dry_run):
                    moved += 1

    _prune_empty_dirs(baseline_root, dry_run=dry_run)
    return moved


def _parse_legacy_stem(stem: str) -> tuple[str, str, str, str, str]:
    parts = stem.split("__")
    token_budget = ""
    max_steps = ""
    rs_steps = ""
    cs_steps = ""
    param_indices: list[int] = []
    for index, part in enumerate(parts):
        if part.startswith("tb"):
            token_budget = part[2:]
            param_indices.append(index)
        elif part.startswith("ms"):
            max_steps = part[2:]
            param_indices.append(index)
        elif part.startswith("rs"):
            rs_steps = part[2:]
            param_indices.append(index)
        elif part.startswith("cs"):
            cs_steps = part[2:]
            param_indices.append(index)
    if not token_budget or not max_steps or not param_indices:
        return "", "", "", "", ""
    bench_key = "__".join(parts[: min(param_indices)])
    return bench_key, token_budget, max_steps, rs_steps, cs_steps


def _infer_model_from_slug(model_slug: str) -> str:
    """Best-effort inverse of :func:`slugify` for known matrix eval models."""
    known = {
        "Qwen_Qwen2.5_1.5B_Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen_Qwen2.5_Coder_1.5B_Instruct": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen_Qwen2.5_Coder_7B_Instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen_Qwen2.5_Coder_14B_Instruct": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "meta_llama_Llama_3.1_8B_Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    }
    return known.get(model_slug, model_slug.replace("_", "/"))


def migrate_legacy_generated_tree(generated_root: Path, *, dry_run: bool = False) -> int:
    """Move flat ``<output_name>_<run_id>/`` dirs into model/benchmark/strategy."""
    import re

    run_dir_re = re.compile(
        r"^(?P<output_name>.+)_(?P<run_id>\d{8}_\d{6}_[0-9a-f]{6})$"
    )
    moved = 0
    if not generated_root.is_dir():
        return moved

    for entry in list(generated_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in {"latest"}:
            continue
        if not ((entry / "results").is_dir() or (entry / "dafny").is_dir()):
            continue
        match = run_dir_re.match(entry.name)
        if not match:
            continue
        output_name = match.group("output_name")
        run_id = match.group("run_id")
        eval_model, benchmark, strategy = _infer_generated_run_axes(output_name)
        if not eval_model or not benchmark:
            continue
        new_dir = generated_run_dir(
            generated_root,
            eval_model=eval_model,
            benchmark=benchmark,
            strategy=strategy,
            output_name=output_name,
            run_id=run_id,
        )
        if new_dir == entry or new_dir.is_dir():
            continue
        if dry_run:
            moved += 1
            continue
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(entry), str(new_dir))
        moved += 1
    return moved


def _infer_generated_run_axes(output_name: str) -> tuple[str, str, str]:
    strategy = synthesis_strategy_from_output_name(output_name)
    if output_name.startswith("metadecode_"):
        rest = output_name[len("metadecode_") :]
        benchmark = ""
        smiles_class = ""
        if rest.startswith("gsm_symbolic_"):
            benchmark = "gsm_symbolic"
            rest = rest[len("gsm_symbolic_") :]
        elif rest.startswith("spider_"):
            benchmark = "spider"
            rest = rest[len("spider_") :]
        elif rest.startswith("smiles_"):
            benchmark = "smiles"
            rest = rest[len("smiles_") :]
            if rest.startswith("class_"):
                class_parts = rest.split("_")
                if len(class_parts) >= 2:
                    smiles_class = class_parts[1]
                    rest = "_".join(class_parts[2:])
        else:
            return "", "", strategy
        model_slug = _first_model_slug_segment(rest)
        eval_model = _infer_model_from_slug(model_slug)
        if benchmark == "smiles" and smiles_class:
            benchmark = benchmark_key("smiles", smiles_class)
        return eval_model, benchmark, strategy
    return "", "", strategy


def _first_model_slug_segment(rest: str) -> str:
    known_prefixes = (
        "Qwen_Qwen2.5_Coder_7B_Instruct",
        "Qwen_Qwen2.5_Coder_14B_Instruct",
        "Qwen_Qwen2.5_Coder_1.5B_Instruct",
        "Qwen_Qwen2.5_1.5B_Instruct",
        "meta_llama_Llama_3.1_8B_Instruct",
    )
    for prefix in known_prefixes:
        if rest.startswith(prefix):
            return prefix
    return rest.split("_iter", 1)[0]


def _prune_empty_dirs(root: Path, *, dry_run: bool = False) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if not path.is_dir():
            continue
        if path == root:
            continue
        if any(path.iterdir()):
            continue
        if not dry_run:
            path.rmdir()
