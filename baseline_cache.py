"""Mandatory train-baseline cache helpers for experiment workflows."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable


Validator = Callable[[dict[str, Any]], bool]


def file_digest(path: Path | str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def baseline_cache_key(dataset: str, method: str, identity: dict[str, Any]) -> str:
    payload = {
        "dataset": dataset,
        "method": method,
        "identity": _canonical(identity),
        "schema_version": 1,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def baseline_cache_path(
    output_dir: Path,
    dataset: str,
    method: str,
    identity: dict[str, Any],
) -> Path:
    key = baseline_cache_key(dataset, method, identity)
    safe_dataset = dataset.replace("/", "_")
    safe_method = method.replace("/", "_")
    return output_dir / "benchmarks" / "baseline_cache" / safe_dataset / f"{safe_method}_{key}.json"


def accuracy_syntax_validator(payload: dict[str, Any]) -> bool:
    return "accuracy" in payload and "syntax_rate" in payload


def json_validator(payload: dict[str, Any]) -> bool:
    return bool(payload)


def _load_valid(path: Path, validator: Validator | None) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if validator is not None and not validator(payload):
        return None
    return payload


_MISSING = object()


def _lookup(payload: dict[str, Any], key: str) -> Any:
    config = payload.get("config")
    if isinstance(config, dict) and key in config:
        return config[key]
    cache = payload.get("baseline_cache")
    if isinstance(cache, dict):
        identity = cache.get("identity")
        if isinstance(identity, dict) and key in identity:
            return identity[key]
    if key in payload:
        return payload[key]
    return _MISSING


def _same_value(expected: Any, actual: Any) -> bool:
    expected = _canonical(expected)
    actual = _canonical(actual)
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return abs(float(expected) - float(actual)) < 1e-12
        except Exception:
            return False
    return expected == actual


def _payload_matches_identity(payload: dict[str, Any], identity: dict[str, Any]) -> bool:
    matched = 0
    for key, expected in identity.items():
        if key in {"split_digest", "cache_note"}:
            continue
        actual = _lookup(payload, key)
        if actual is _MISSING:
            continue
        if not _same_value(expected, actual):
            return False
        matched += 1
    return matched >= 4


def _candidate_paths(output_dir: Path, method: str) -> list[Path]:
    benchmarks_dir = output_dir / "benchmarks"
    if not benchmarks_dir.exists():
        return []
    method_l = method.lower()
    candidates = [
        path
        for path in benchmarks_dir.rglob("*.json")
        if "baseline_cache" not in path.parts and method_l in path.name.lower() and "train" in path.name.lower()
    ]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _write_with_cache_metadata(
    payload: dict[str, Any],
    path: Path,
    *,
    dataset: str,
    method: str,
    identity: dict[str, Any],
    cache_path: Path,
    source_path: Path,
    cache_hit: bool,
) -> None:
    payload = dict(payload)
    payload["baseline_cache"] = {
        "dataset": dataset,
        "method": method,
        "key": baseline_cache_key(dataset, method, identity),
        "identity": _canonical(identity),
        "cache_path": str(cache_path),
        "source_path": str(source_path),
        "cache_hit": cache_hit,
        "updated_at_unix": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def reuse_cached_baseline(
    *,
    output_dir: Path,
    dataset: str,
    method: str,
    identity: dict[str, Any],
    output_path: Path,
    label: str,
    validator: Validator | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Copy a matching cached baseline to output_path, or report a required miss."""
    cache_path = baseline_cache_path(output_dir, dataset, method, identity)
    result = {
        "hit": False,
        "cache_path": str(cache_path),
        "source_path": None,
        "output_path": str(output_path),
    }
    if dry_run:
        print(f"[baseline-cache] DRY-RUN {label}: would use {cache_path}")
        return result

    payload = _load_valid(cache_path, validator)
    source_path = cache_path if payload is not None else None
    if payload is None:
        for candidate in _candidate_paths(output_dir, method):
            candidate_payload = _load_valid(candidate, validator)
            if candidate_payload is None:
                continue
            if _payload_matches_identity(candidate_payload, identity):
                payload = candidate_payload
                source_path = candidate
                break

    if payload is None or source_path is None:
        print(f"[baseline-cache] MISS {label}: {cache_path}")
        return result

    if source_path != cache_path:
        _write_with_cache_metadata(
            payload,
            cache_path,
            dataset=dataset,
            method=method,
            identity=identity,
            cache_path=cache_path,
            source_path=source_path,
            cache_hit=False,
        )
        payload = json.loads(cache_path.read_text())
        print(f"[baseline-cache] PROMOTED {label}: {source_path} -> {cache_path}")

    _write_with_cache_metadata(
        payload,
        output_path,
        dataset=dataset,
        method=method,
        identity=identity,
        cache_path=cache_path,
        source_path=cache_path,
        cache_hit=True,
    )
    print(f"[baseline-cache] HIT {label}: reused {cache_path}")
    result.update({"hit": True, "source_path": str(cache_path)})
    return result


def store_cached_baseline(
    *,
    output_dir: Path,
    dataset: str,
    method: str,
    identity: dict[str, Any],
    output_path: Path,
    label: str,
    validator: Validator | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    cache_path = baseline_cache_path(output_dir, dataset, method, identity)
    result = {
        "stored": False,
        "cache_path": str(cache_path),
        "output_path": str(output_path),
    }
    if dry_run:
        return result
    payload = _load_valid(output_path, validator)
    if payload is None:
        raise RuntimeError(f"Cannot cache invalid baseline output for {label}: {output_path}")
    _write_with_cache_metadata(
        payload,
        output_path,
        dataset=dataset,
        method=method,
        identity=identity,
        cache_path=cache_path,
        source_path=output_path,
        cache_hit=False,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, cache_path)
    print(f"[baseline-cache] STORED {label}: {cache_path}")
    result["stored"] = True
    return result
