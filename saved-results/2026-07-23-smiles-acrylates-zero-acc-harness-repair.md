# SMILES acrylates zero-acc harness repair

**Date:** 2026-07-23  
**What for:** Incident `smiles-acrylates-qwen25-1p5b:5:harness:1784848009`  
**Worktree:** `/home/aadivyar/csd-generation-babysitter-repair`  
**Branch:** `babysitter-fix/smiles-acrylates-qwen25-1p5b-5-harness-1784848009`  
**Broken SHA:** `3b7c917bcb76d98bd02f97fdcd9d4bb3da16dd6c`

## Symptom

Cold-queue cell `smiles-acrylates-qwen25-1p5b` finished attempts 2/4/5 at
**Accuracy 0.0%** (unique-valid 0/50). Babysitter classified path_kind=`harness`.

Evidence from `outputs/generated/coldq_smiles-acrylates-qwen25-1p5b_20260724/run.log`:

- Attempt 2: RDKit validity 100%, membership 0%, many **1-token** generations.
- Attempt 5: every example emitted the same string `CC(=O)OCC1=CC`
  (argmax collapse; RDKit "unclosed ring").
- Live process env for pid running the cell had **no**
  `CSD_CONSTRAINED_TEMPERATURE` (defaults to `0.0` in
  `synthesis/evaluate/benchmarks/common/model_utils.py`).
- Attempt 1 also failed vLLM engine init: free VRAM was only
  ~4 GiB on the assigned GPU, so even the job's `gpu_mem_util=0.4`
  (~15.8 GiB request) could not start. Separately, synthesis was
  still using the global default util (~0.8) instead of the job bar,
  which makes crowded-GPU startups worse. Forwarding
  `CSD_VLLM_GPU_MEMORY_UTILIZATION` restores the job contract; it does
  not by itself create free VRAM when another process owns the card.

## Root cause

Inputs → Outputs → Algorithm:

1. **Inputs:** cold queue launches `synthesis.run_synthesis` for SMILES without
   setting span-sampling temperature; evaluator uses argmax.
2. **Outputs:** identical / tiny SMILES across the 50-example train bar →
   unique-valid rate 0 → Accuracy 0.0%.
3. **Algorithm:** `ChooseNextToken` reads
   `CSD_CONSTRAINED_TEMPERATURE` (default `0.0`). Pilots
   (`pilot_smiles_uv*.sh`) export `0.7`; the cold queue did not.

Secondary: synthesis ignored the job's `gpu_mem_util` and used the global
vLLM default (~0.8). That is a real contract hole. Attempt 1's engine-init
failure, though, was dominated by **contention** (~4 GiB free); retries at
util 0.4 still needed ~15.8 GiB and failed until the GPU freed up.

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
  tests/runtime/test_cold_synthesis_queue.py::test_heldout_environment_removes_paid_author_credentials \
  tests/runtime/test_cold_synthesis_queue.py::test_synthesis_environment_names_the_isolated_cold_output \
  tests/runtime/test_cold_synthesis_queue.py::test_poolable_synthesis_environment_uses_the_reserved_two_gpu_bundle \
  -q
```

Result: **6 passed**.

## Sibling note

`scripts/runtime/run_warm_task_recovery_queue.py` has its own
`author_free_environment` and does not set SMILES T=0.7. Out of scope for
this cold-queue incident; fix if warm SMILES held-out is relaunched.

## Reuse / deploy

After merge into the live snapshot, restart the cold-queue cell (or whole
controller) so new env vars apply. Existing in-flight attempt will not pick
them up mid-process.
