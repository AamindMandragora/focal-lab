# AGENTS.md — `scripts/runtime/`

## Scope

Cold-queue and babysitter runtime scripts. Repo-wide rules live in `../../AGENTS.md`.

## Cold synthesis queue (`run_cold_synthesis_queue.py`)

- SMILES cells must export `CSD_CONSTRAINED_TEMPERATURE=0.7` in synthesis and
  held-out envs. Default argmax (`0.0`) collapses every example to the same
  tiny SMILES and drives unique-valid accuracy to zero.
- Always forward the job's `gpu_mem_util` via
  `CSD_VLLM_GPU_MEMORY_UTILIZATION` (consumed by `synthesis.run_synthesis`).
  Do not hard-code the global `VLLM_GPU_MEMORY_UTILIZATION` for cold jobs.
- Do not put strategy coaching into `--task` strings (Critical Prompting Rule).
- Claude Code synthesis queue commands use the fixed `claude-opus-5` model ID;
  direct Anthropic and Bedrock model IDs are separate routes and should not be
  changed by this contract.
- The `full-baseline-20260803` profile must contain exactly 20 cold cells and
  800 author attempts. Its manifest must bind the `aadivya@fermi.ai` Max
  profile and all five raw baseline hashes before dispatch.
- For that profile, use one exact example above the maximum baseline accuracy,
  cap the maximum syntax rate at 90%, and use the explicitly labeled 95%
  exception only when the maximum baseline accuracy is 100%.

## Zero-acc babysitter

Repair agents run in the sibling worktree
(`~/csd-generation-babysitter-repair`), never by checking out branches on the
live cold-queue tree.
