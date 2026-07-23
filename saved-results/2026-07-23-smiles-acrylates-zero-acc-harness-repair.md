# SMILES acrylates zero-acc harness repair

**Date:** 2026-07-23  
**What for:** Incident `smiles-acrylates-qwen25-1p5b:8:harness:1784850078`  
(same harness gap as attempt-5 `…:5:harness:1784848009` / PR #5)  
**Worktree:** `/home/aadivyar/csd-generation-babysitter-repair`  
**Branch:** `babysitter-fix/smiles-acrylates-qwen25-1p5b-8-harness-1784850078`  
**Broken SHA:** `3b7c917bcb76d98bd02f97fdcd9d4bb3da16dd6c`

## Symptom

Cold-queue cell `smiles-acrylates-qwen25-1p5b` hit **Accuracy 0.0%**
(unique-valid collapse). Babysitter classified `path_kind=harness`.

Evidence:

- Live tree HEAD was still `3b7c917b` (behind `origin/synthesis-snapshot-20260622`
  which already contains PR #5). In-flight attempts therefore never saw the
  pilot-parity env exports.
- Live process env for the cell had **no** `CSD_CONSTRAINED_TEMPERATURE`
  (defaults to `0.0` in `synthesis/evaluate/benchmarks/common/model_utils.py`).
- Synthesis also ignored the job's `gpu_mem_util` and used the global vLLM
  default (~0.8 / 0.81) instead of the per-job bar.
- Train log also shows `tiny_span_dominant: 100%` / membership 0% (often `"C"` /
  ethane-like early closes). That is a real strategy failure mode, but it is
  separate from the missing pilot-parity harness knobs.

## Root cause (harness gap)

Inputs → Outputs → Algorithm:

1. **Inputs:** cold queue launches `synthesis.run_synthesis` for SMILES without
   setting span-sampling temperature; evaluator uses argmax.
2. **Outputs:** identical / tiny SMILES across the train bar → unique-valid
   rate 0 → Accuracy 0.0%.
3. **Algorithm:** `ChooseNextToken` reads `CSD_CONSTRAINED_TEMPERATURE`
   (default `0.0`). Pilots export `0.7`; the cold queue on this broken SHA did
   not.

Secondary: synthesis ignored `job["gpu_mem_util"]`.

## Residual risk (verified)

Post-PR #5 smoke on the same cell
(`logs/zero_acc_babysitter/smoke_smiles-acrylates-qwen25-1p5b_20260723T232808Z`)
still reported Acc=0 / UV=0 with 1-token `"C"` outputs, while
`smoke_verdict.json` marked the harness smoke **passed**. So this repair
restores pilot-parity wiring; it does **not** by itself prove Acc>0 on
existing tiny-span strategies. After merge, live must **pull + restart** the
cell so new env vars apply.

## Fix (minimal)

- `scripts/runtime/run_cold_synthesis_queue.py`
  - SMILES synthesis + held-out envs set `CSD_CONSTRAINED_TEMPERATURE=0.7`.
  - All synthesis envs set `CSD_VLLM_GPU_MEMORY_UTILIZATION` from
    `job["gpu_mem_util"]`.
- `synthesis/run_synthesis.py`
  - Read `CSD_VLLM_GPU_MEMORY_UTILIZATION` when constructing generator/evaluator.

## Tests

```bash
cd /home/aadivyar/csd-generation-babysitter-repair
python -m pytest \
  tests/runtime/test_cold_synthesis_queue.py::test_smiles_synthesis_environment_enables_constrained_sampling_and_job_gpu_util \
  tests/runtime/test_cold_synthesis_queue.py::test_non_smiles_synthesis_environment_does_not_force_constrained_temperature \
  tests/runtime/test_cold_synthesis_queue.py::test_smiles_heldout_environment_enables_constrained_sampling \
  tests/runtime/test_cold_synthesis_queue.py::test_resolve_vllm_gpu_memory_utilization_prefers_env_override \
  tests/runtime/test_cold_synthesis_queue.py::test_heldout_environment_removes_paid_author_credentials \
  tests/runtime/test_cold_synthesis_queue.py::test_synthesis_environment_names_the_isolated_cold_output \
  tests/runtime/test_cold_synthesis_queue.py::test_poolable_synthesis_environment_uses_the_reserved_two_gpu_bundle \
  -q
```

Result: **7 passed**.

## Sibling search

- Searched `CSD_CONSTRAINED_TEMPERATURE` / `CSD_VLLM_GPU_MEMORY_UTILIZATION` /
  `author_free_environment` under `scripts/` and `synthesis/run_synthesis.py`.
- Cold queue is the only synthesis launcher that needed the SMILES T=0.7 export.
- `scripts/runtime/run_warm_task_recovery_queue.py` has its own
  `author_free_environment` and does **not** set SMILES T=0.7. Out of scope
  for this cold-queue incident; fix if warm SMILES held-out is relaunched.
- Env tests that still passed 2 GPUs were updated to 3 to match
  `POOLABLE_GPU_COUNT=3` already on the broken SHA (otherwise
  `synthesis_environment` raises before assertions).

## Reuse / deploy

After merge into the live snapshot, **pull on the live tree and restart** the
cold-queue cell (or whole controller) so new env vars apply. Existing
in-flight attempts will not pick them up mid-process. Until live leaves
`3b7c917b`, babysitter will keep reopening the same harness incident on later
attempts.
