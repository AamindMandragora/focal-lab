"""
One-off diagnostic (NOT a standing script): does CRANE-style span-body MASKING
hold up on GSM-1.5B, or does it dead-end?

Inputs:
  - compiled_module: the WINNING GSM-7B strategy's GeneratedCSD.py (att14,
    0.36/0.915 on 7B). It provably masks span bodies via ConstrainedStep.
  - eval_model: Qwen/Qwen2.5-1.5B-Instruct (the weaker target).
Output:
  - JSON with accuracy, syntax_rate, contains_delimiters, and a per-example
    dead-end audit parsed from sample_outputs[].helper_trace.
Algorithm:
  1. Build Evaluator on gsm_symbolic / 1.5B / vllm.
  2. evaluate_sample(winning_7B_module, n).  <-- runs masking on the 1.5B
  3. Scan each sample's helper_trace for DeadEnd/Rollback signals; tally
     syntax pass/fail and accuracy.
This directly answers the user's chosen question: "masking alone viable, or
grammar must change?"

NOTE: body is under `if __name__ == "__main__"` so vLLM's multiprocessing
engine children (which re-import this module) don't re-run it.
"""
import json
import sys
from pathlib import Path


def main() -> None:
    from synthesis.evaluate.evaluator import Evaluator

    win_7b = Path(
        "outputs/generated/validation_3changes_metadecode_gsm_7b_opus47_iter15_20260520_224040_78a274"
        "/python/validation_3changes_metadecode_gsm_7b_opus47_iter15_20260521_001829_35d746"
        "/GeneratedCSD.py"
    )
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    gpu_mem = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/diag_mask_1p5b_result.json")

    assert win_7b.is_file(), f"winning 7B module not found: {win_7b}"

    ev = Evaluator(
        dataset_name="gsm_symbolic",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        backend="vllm",
        device="cuda",  # CUDA_VISIBLE_DEVICES pins the GPU; create_vllm_lm rejects "auto"
        sample_size=n,
        max_steps=900,
        step_token_budget=1,
        vllm_gpu_memory_utilization=gpu_mem,
    )
    try:
        res = ev.evaluate_sample(win_7b, sample_size=n)
    finally:
        ev.unload_runtime()

    if not getattr(res, "success", True) or res.num_examples == 0:
        print("EVAL FAILED: success=", getattr(res, "success", None),
              "num_examples=", res.num_examples, "error=", getattr(res, "error", None))
        raise SystemExit(1)

    samples = res.sample_outputs or []
    deadend_hits = 0
    syntax_fail = 0
    audit = []
    for i, s in enumerate(samples):
        trace = s.get("helper_trace") or ""
        trace_str = trace if isinstance(trace, str) else json.dumps(trace)
        low = trace_str.lower()
        has_deadend = ("deadend" in low) or ("dead_end" in low) or ("rollback" in low)
        if has_deadend:
            deadend_hits += 1
        if not s.get("is_syntax_valid", True):
            syntax_fail += 1
        audit.append({
            "i": i,
            "syntax_ok": bool(s.get("is_syntax_valid", True)),
            "deadend": has_deadend,
            "num_visible_spans": s.get("num_visible_spans"),
            "num_valid_visible_spans": s.get("num_valid_visible_spans"),
            "answer_source": s.get("answer_source"),
            "expected": s.get("expected"),
            "actual": s.get("actual"),
        })

    payload = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "strategy": "GSM-7B winner (masking) run unchanged on 1.5B",
        "n": res.num_examples,
        "accuracy": float(res.accuracy),
        "syntax_rate": float(res.syntax_rate),
        "contains_delimiters": res.contains_delimiters,
        "num_correct": res.num_correct,
        "deadend_examples": deadend_hits,
        "syntax_fail_examples": syntax_fail,
        "audit": audit,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "audit"}, indent=2))
    print(f"\nWrote full audit to {out}")


if __name__ == "__main__":
    main()
