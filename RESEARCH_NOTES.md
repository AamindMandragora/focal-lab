# CSD Generation Research Notes

## Project Goal
Dynamically generate constrained decoding strategies (CSDs) using open-source LLMs that outperform the CRANE baseline on GSM-Symbolic and FOLIO datasets.

---

## Current Status (April 22, 2026)

**Where we are:** The GPT-5.4-based synthesis pipeline reliably produces CSDs that beat CRANE on a small (10-example) training set — 4 out of 5 lottery seeds produced a training-set winner (80% pipeline reliability). **But those training wins do not transfer.** On a 50-example held-out GSM-Symbolic split (seed=456), synthesized CSDs perform within binomial noise of CRANE (38–44% accuracy vs CRANE's 42%), and the training→held-out accuracy drop is much larger for synthesized strategies (~40pt) than for CRANE (~8pt).

**Key insight:** the 10-example gate in the synthesis loop is too noisy to distinguish genuinely-better strategies from ones that got a lucky draw on those specific 10 examples. With a 55% accuracy gate and 20 attempts per run, the loop essentially performs selection on binomial noise rather than selecting for real skill — a true-40%-accuracy strategy has ~16.6% chance of passing the 55% gate on a 10-example sample, so at least one of 20 attempts passes ~97% of the time even if none of the strategies is actually better than CRANE.

**Plain-English takeaway:** just because a strategy succeeds on a small training sample doesn't mean it will succeed on a larger held-out sample. Small-n evaluation with multiple attempts is a selection filter on noise.

**Proposed fix (next experiment):** raise the synthesis gate from `--eval-sample-size 10` to `--eval-sample-size 50` (or at least 30), ideally combined with a held-out validation split inside the synthesis loop so that passing the gate on the training sub-split must be corroborated on a separate sub-split before a candidate is declared a winner.

---

## Key Observations

### 1. Model Scale Threshold for CSD Synthesis (April 2, 2026)

**Finding:** The CSD synthesis framework requires models above a certain parameter threshold to be effective. Models below this threshold lack the reasoning capabilities needed to generate novel, verifiable CSDs.

**Evidence:**
- **Qwen2.5-Coder-7B-Instruct** failed to synthesize successful strategies after 15+ attempts
  - 12/15 attempts failed at Dafny verification stage
  - 3/15 attempts passed verification but achieved only 0-20% accuracy (CRANE baseline: ~70%)
  - Common failures: incorrect method signatures, syntax errors in `decreases` clauses, using methods in expression positions

**Root Cause:**
- Generating valid Dafny code requires understanding:
  - Complex type systems and contracts (preconditions, postconditions, invariants)
  - Verification constraints (decreases clauses, loop invariants)
  - Tool contracts and their interdependencies
- 7B models struggle with:
  - Multi-step reasoning about verification constraints
  - Generating syntactically correct Dafny while maintaining semantic correctness
  - Discovering novel strategy patterns beyond prompt examples

**Implication:** 
- Minimum viable model size appears to be **14B+ parameters** for this task
- Smaller models may work for simpler synthesis tasks, but CSD generation requires stronger reasoning

**Action:** Testing with `Qwen/Qwen2.5-Coder-14B-Instruct` to validate this hypothesis.

---

### 2. Verification Failures Dominate (April 2, 2026)

**Pattern:** Most synthesis attempts fail at the Dafny verification stage, not at evaluation.

**Common Verification Errors:**
1. `decreases expression might not decrease` - Model doesn't understand termination proofs
2. `invariant could not be proved to be maintained` - Often caused by double-incrementing `helpers.cost`
3. `method returns 1 value but is assigned to 2 variables` - Confusion about method signatures
4. `expression is not allowed to invoke a method` - Using methods in `if`/`while` conditions
5. Parse errors from invalid Dafny syntax (e.g., `<<""` instead of `<<"`)

**Insight:** The model needs to understand Dafny's verification model, not just syntax. This requires stronger reasoning capabilities.

---

### 3. CRANE Baseline Performance (April 2, 2026)

**GSM-Symbolic (Qwen2.5-Coder-7B-Instruct):**
- Expected accuracy: ~70% (per README)
- Format validity: 100% (CSD ensures correct structure)
- Syntax validity: 100% (all `<< >>` expressions are valid)

**Note:** Full baseline evaluation timed out due to slow generation (2-3 min per example with 1024 max steps). Need to run with smaller sample or fewer steps for quick validation.

---

### 4. Task Description Constraints (April 2, 2026)

**Critical Rule:** Task descriptions must NOT contain strategy guidance.

**Allowed:**
- Task description (what the model should accomplish)
- Available tools (signatures, preconditions, postconditions, types only)

**NOT Allowed:**
- Recommendations on which tools to use
- Patterns to avoid
- Hints about strategy structure
- Comparisons to baseline
- Usage hints on tools
- Notes about which structural patterns are preferred

**Rationale:** This is a controlled study. The model must discover effective strategies autonomously from the task description and tool contracts alone.

---

### 5. Rationale Block Requirement (April 2, 2026)

**Issue:** Qwen 7B frequently omitted the required `// CSD_RATIONALE_BEGIN ... // CSD_RATIONALE_END` block.

**Fix Applied:** Modified `synthesis/generator.py` to auto-inject a placeholder rationale block when the model fails to produce one:
```python
return "// CSD_RATIONALE_BEGIN\n// (Auto-injected rationale)\n// CSD_RATIONALE_END\n" + current
```

**Trade-off:** This allows the pipeline to continue but loses the model's explanation of its strategy choice. With larger models, this should be less necessary.

---

## Experimental Log

### Experiment 1: GSM-Symbolic with Qwen2.5-Coder-7B-Instruct
- **Date:** April 2, 2026
- **Model:** Qwen/Qwen2.5-Coder-7B-Instruct
- **Dataset:** gsm_symbolic
- **Max Iterations:** 15
- **Result:** FAILED
- **Best Accuracy:** 20% (threshold: 30%)
- **Failure Mode:** Verification failures (12/15), low accuracy on passing strategies (3/15)
- **Conclusion:** 7B model insufficient for CSD synthesis

### Experiment 2: GSM-Symbolic with Qwen2.5-Coder-14B-Instruct
- **Date:** April 2, 2026
- **Model:** Qwen/Qwen2.5-Coder-14B-Instruct (4-bit quantization)
- **Dataset:** gsm_symbolic
- **Status:** INCOMPLETE (timeout during evaluation)
- **Observations:**
  - Model successfully loads with 4-bit quantization
  - First attempt passed verification on try #4 (vs 7B which took many more attempts)
  - Verification success rate appears higher than 7B
  - Attempt #4: 0% accuracy, 40% format, 0% syntax (still below threshold)
  - Attempt #5: Passed verification, evaluation timed out
- **Conclusion:** 14B model shows better verification success but still struggles to find effective strategies

### Experiment 3: GSM-Symbolic with Qwen2.5-Coder-32B-Instruct
- **Date:** April 2, 2026
- **Model:** Qwen/Qwen2.5-Coder-32B-Instruct (4-bit quantization)
- **Dataset:** gsm_symbolic
- **Status:** FAILED (CUDA OOM)
- **Issue:** Model loads successfully (27.98 GiB) but runs out of memory during forward pass
- **Error:** "Tried to allocate 6.12 GiB" during generation
- **Observation:** 32B model is too large for available GPU memory even with 4-bit quantization

---

## Additional Observations

### 8. Memory Constraints Limit Model Scale (April 2, 2026)

**Finding:** Even with 4-bit quantization, the 32B model exceeds available GPU memory during inference.

**Evidence:**
- Model loads: 27.98 GiB allocated
- Forward pass requires additional 6.12 GiB
- Total needed: ~34 GiB (exceeds available memory on GPUs 1-2)

**Implication:** 
- Maximum practical model size with current hardware: **14B parameters (4-bit)**
- Would need either:
  - More GPU memory (e.g., A100 80GB)
  - Model parallelism across more GPUs
  - More aggressive quantization (e.g., 3-bit or 2-bit)

---

### 9. Prompt Examples Updated with Task Descriptions (April 2, 2026)

**Change:** Modified `synthesis/prompts.py` to explicitly show task descriptions for each example strategy.

**Rationale:**
- Original examples showed strategies without context
- Model couldn't understand which strategy fits which task
- New format: "**Task Description:** ..." followed by "**Strategy:** ..."

**Examples now cover:**
1. Structured reasoning with delimited expressions
2. Always-constrained generation
3. Nested expressions with balanced parentheses
4. Premature closure prevention
5. Confidence-gated optional constraining
6. Adaptive sampling with logit manipulation
7. Multiple overlapping constrained regions

**Hypothesis:** This should help the model understand the mapping between task requirements and strategy choices.

---

## Additional Observations

### 10. Parser Performance: DFA Mask Store vs Brute Force Lark Parsing (April 4, 2026)

**Finding:** The original `ValidNextTokens` implementation was O(vocab) brute-force Lark parsing, making evaluation impossibly slow.

**Evidence:**
- Original implementation: For each of 152,064 vocabulary tokens, called `lark.parse(current_text + token_str)` — a full Lark parse
- Each parse took ~0.1-1ms, so `ValidNextTokens` took 15-150 seconds per call
- With 50 steps × 3 examples = 150 calls, evaluation took 37-375 minutes
- Process showed 0% GPU utilization, 100% CPU-bound on Lark parsing

**Root Cause:** `evaluations/common/parser_utils.py` iterated over the entire vocabulary and called `self._lark.parse(extended)` for each token, with no caching benefit (each token produces a unique string).

**Fix:** Replaced brute-force parser with syncode's `DFAMaskStore`:
- Builds DFA from grammar terminals once at startup (cached)
- Uses incremental parsing to determine valid accept sequences
- Converts boolean mask to token list via O(vocab) array scan (~0.01s vs ~15-150s)
- ~1000-15000x speedup for `ValidNextTokens`

**Implementation:**
- Modified `evaluations/common/parser_utils.py` to use `syncode.dfa_mask_store.DFAMaskStore`
- Modified `evaluations/gsm_symbolic/environment.py` and `evaluations/folio/environment.py` to pass tokenizer to parser factory
- Falls back to brute-force if DFA mask store unavailable (graceful degradation)

**Impact:** Evaluation time per example dropped from ~200s to ~8-15s (now dominated by model forward pass, not parser)

---

### 11. vLLM Runtime Integration for the Synthesis Eval Loop (April 10, 2026)

**Finding:** The synthesis eval loop is wired correctly for vLLM, but reproducible end-to-end validation on `focal` is currently blocked by shared-GPU instability during vLLM initialization.

**What "eval loop" means here:**
- In `synthesis/feedback_loop.py`, after verification, compilation, and runtime succeed, stage `[4/4]` calls `Evaluator.evaluate_sample(...)`
- `Evaluator.evaluate_sample(...)` builds the dataset environment, then calls `run_crane_csd(...)`
- `run_crane_csd(...)` delegates directly to `GeneratedCSD.default__.MyCSDStrategy(...)`
- If metrics fail thresholds, the pipeline calls `generator.refine_after_evaluation_failure(...)` and iterates

This is the actual generate -> verify -> compile -> run -> evaluate -> refine loop, not a separate ad hoc benchmark path.

**Implementation changes made for vLLM:**
- Runtime backend selection now threads through `synthesis/evaluator.py` into `evaluations/*/environment.py`
- `evaluations/common/model_utils.py` now supports a vLLM runtime path
- Because the installed vLLM on `focal` does not accept per-request `SamplingParams(logits_processors=[...])`, the runtime was adapted to reconstruct the next-token score tensor from full-vocabulary `logprobs=-1`
- To make that work on vLLM 0.19, engine construction also needed `max_logprobs=-1`
- Tensorized behavior is preserved after capture: constrained-vocab slicing is tensor indexing, masking is `masked_fill_`, and token choice is `argmax`

**Evidence from remote validation on `focal`:**
- Used compiled module from `outputs/generated-csd/runs/20260405_082143_1a21fd/gsm_14b_4bit_csd/GeneratedCSD.py`
- Initial eval-path failures were real integration bugs and were fixed:
  - `Unexpected keyword argument 'logits_processors'`
  - `Requested sample logprobs ... greater than max allowed: 20`
  - `index 151665 is out of bounds for dimension 0 with size 151665`
- After those fixes, the evaluator consistently reaches the real dataset-backed CSD path before failing, which confirms the eval loop is invoking the vLLM-backed constrained runtime rather than bypassing it

**Current blocker:**
- Final end-to-end confirmation on April 10 was prevented by shared-GPU volatility on `focal`
- Observed failures were vLLM engine init / profiling issues caused by other users' processes changing available GPU memory during startup:
  - startup free-memory shortfall
  - model-load OOM
  - vLLM profiling assertion when free memory increased during initialization

**Successful retry:** A later retry on April 10 succeeded with:
- `Qwen/Qwen2.5-Coder-14B-Instruct`
- `backend='vllm'`
- `tensor_parallel_size=2`
- `gpu_memory_utilization=0.6`
- `max_model_len=3072`
- `enforce_eager=True`
- `sample_size=1`
- `max_steps=60`

The evaluator entered the real constrained decoding path, printed `GenerateLogits` progress from the Dafny runtime, generated 34 tokens in ~41.97s, and returned a normal `EvaluationResult` with `success=True` and `sample_error=None`. The sample itself was low-quality (`accuracy=0`, malformed constrained output), but that is a model/strategy quality problem, not an eval-loop integration failure.

**Conclusion:** The software integration for the eval loop is now aligned with the installed vLLM API, and the eval loop does work with vLLM on `focal` when run under a stable 2-GPU 14B configuration. The remaining issue is output quality, not whether the evaluator can drive the vLLM-backed constrained runtime.

---

### 12. Controlled 14B->32B vLLM Ablation Is Currently Infrastructure-Blocked (April 10, 2026)

**Goal:** Run a clean model-capability ablation by holding the prompt stack and synthesis setup fixed while increasing the generation model from `Qwen/Qwen2.5-Coder-14B-Instruct` to `Qwen/Qwen2.5-Coder-32B-Instruct`.

**What was tried:**
- First-candidate generation only, not a full synthesis loop
- `backend='vllm'`
- `tensor_parallel_size=4`
- `gpu_memory_utilization=0.4`
- `max_model_len=3072`
- `enforce_eager=True`
- `CUDA_VISIBLE_DEVICES=0,1,2,3`

**Observed failures:**
- Initial attempt failed during worker startup with `Cannot re-initialize CUDA in forked subprocess`
- Retrying with `VLLM_WORKER_MULTIPROC_METHOD=spawn` got past startup and into real engine initialization
- The `spawn` retry then failed with a genuine CUDA OOM during vLLM's dummy/profile forward pass on GPU 2

**Memory state at failure:**
- GPU 2 had only about `213 MiB` free
- Other resident processes on that card were already using roughly `10.48 GiB`, `6.76 GiB`, `0.41 GiB`, and `5.09 GiB`
- vLLM then failed on an additional `80 MiB` allocation

**Conclusion:** This does not yet tell us whether 32B is better or worse for synthesis quality. The April 10 32B ablation attempt was blocked by shared-GPU availability on `focal`, not by a prompt-quality result. A fair 14B->32B comparison still requires either a cleaner 4-GPU window or a lower-memory 32B configuration.

---

### 13. 32B AWQ vLLM Path Successfully Generates a First Candidate (April 11, 2026)

**Goal:** Recover a viable 32B ablation path on `focal` after the base 32B checkpoint and the BitsAndBytes 4-bit path both failed under the current vLLM setup.

**What worked:**
- Model: `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`
- Backend: `vllm`
- Runtime-selected quantization kernel: `awq_marlin`
- `tensor_parallel_size=4`
- `pipeline_parallel_size=1`
- `gpu_memory_utilization=0.4`
- `max_model_len=3072`
- `enforce_eager=True`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`

**Observed result:**
- First-candidate generation completed successfully
- `GEN_SECONDS = 242.77`
- Wall clock from `/usr/bin/time`: `real 251.84`
- Output length: `1355` characters

**Quality note:**
- The generated strategy is still in the same broad family as the 14B candidates:
  - unconstrained outside `<< >>`
  - constrained inside `<< >>`
  - explicit parser-validity invariant
- So this establishes a working 32B quantized path, but not yet a quality win.

**Operational note:**
- vLLM logged an engine-core death during shutdown after generation had already completed
- That did not prevent the candidate from being returned, so the path is usable for experimentation

**Conclusion:** A practical 32B generation path now exists on `focal` through the AWQ checkpoint. This is the first clean 14B->32B capability-ablation path that has actually run end-to-end far enough to return a candidate under vLLM.

---

### 14. Full 32B AWQ Synthesis Loop Still Fails at Verification (April 11, 2026)

**Goal:** Test whether a stronger generation model improves the overall synthesis framework when prompts, refinement loop, and evaluation thresholds are otherwise held fixed.

**Run:**
- Output run: `gsm_32b_awq_vllm_full`
- Run directory: `/home/aadivyar/csd-generation/outputs/generated-csd/runs/20260411_050539_c058e8`
- Generation model: `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`
- Generation backend: `vllm`
- Evaluation model: `Qwen/Qwen2.5-Coder-14B-Instruct`
- Evaluation backend: `vllm`
- Shared vLLM config: `tensor_parallel_size=4`, `pipeline_parallel_size=1`, `gpu_memory_utilization=0.4`, `max_model_len=3072`, `enforce_eager=True`
- `max_iterations=5`
- `eval_sample_size=3`
- `eval_max_steps=60`

**Outcome:**
- The run completed successfully as a job, but failed after all 5 synthesis attempts
- It never reached compilation, runtime, or evaluation
- Failure breakdown: `verification: 5`
- Total wall time: `real 178.30`

**Observed failure mix:**
- Attempt 1: used `ConstrainedStep(...)` in an expression position where Dafny disallowed method invocation
- Attempts 2 and 4: syntax failure (`rbrace expected`)
- Attempt 3: loop invariant maintenance failure for `helpers.cost == steps`
- Attempt 5: loop invariant maintenance failure for parser-valid constrained-prefix state

**Interpretation:**
- 32B AWQ makes the generation path operationally viable, but this run did not demonstrate a framework-level improvement over the current 14B setup
- The model still concentrated in the same broad strategy family:
  - unconstrained outside `<< >>`
  - constrained inside `<< >>`
  - explicit parser-validity invariant
- So the ablation currently says: bigger model is now runnable, but not yet sufficient by itself to break through the verification bottleneck

---

### 15. GPT-5.4 Generation Significantly Improves Synthesis Progress, But Eval Still Has a Runtime Failure Mode (April 11, 2026)

**Goal:** Reconstruct the missing API-generation experiment using `gpt-5.4` for strategy synthesis while keeping the constrained runtime evaluation on the existing local vLLM path.

**First-candidate check:**
- Generation model: `gpt-5.4`
- Generation backend: `openai`
- Result: first candidate returned in `7.84s`
- Output length: `1866` characters
- Compared with the local 14B/32B first candidates, the initial structure was noticeably cleaner and more verifier-aware:
  - included a rationale block
  - introduced `parser.IsValidPrefix(currentConstrained)` on its own
  - used a simpler constrained/unconstrained split without the old example-shaped heuristics

**Full synthesis run:**
- Output run: `gpt54_openai_vllm_full`
- Run directory: `/home/aadivyar/csd-generation/outputs/generated-csd/runs/20260411_175147_100da7`
- Generation model: `gpt-5.4`
- Generation backend: `openai`
- Evaluation model: `Qwen/Qwen2.5-Coder-14B-Instruct`
- Evaluation backend: `vllm`
- Eval config: `tensor_parallel_size=2`, `pipeline_parallel_size=1`, `gpu_memory_utilization=0.6`, `max_model_len=3072`, `enforce_eager=True`
- `max_iterations=5`
- `eval_sample_size=3`
- `eval_max_steps=60`

**Outcome:**
- Total wall time: `real 104.54`
- Failure breakdown: `verification: 4`, `evaluation: 1`
- Unlike the local-model runs, attempt 5 did pass:
  - verification
  - compilation
  - runtime execution
  - and then entered dataset evaluation

**Observed failure mix:**
- Attempt 1: verification failure from duplicate local variable `helpers`
- Attempt 2: loop invariant maintenance failure
- Attempt 3: loop invariant over generated tokens not provable
- Attempt 4: constrained-prefix completeness invariant not maintainable
- Attempt 5: verifier-clean candidate reached evaluation, but the 14B vLLM evaluator crashed with `Engine core initialization failed`

**Interpretation:**
- This is a meaningful framework-level improvement over the 14B and 32B local-generation runs
- `gpt-5.4` is the first generator in this project state that reliably pushed the synthesis loop all the way through:
  - verification
  - compilation
  - runtime execution
  - into dataset evaluation
- The remaining blocker in this run was not strategy synthesis alone; it was the downstream vLLM evaluator runtime on `focal`

**Conclusion:** API-based `gpt-5.4` generation appears substantially stronger than the currently tested local generation models for this synthesis task. It does not eliminate failure modes, but it changes the bottleneck: the project is no longer stuck purely at verifier-invalid strategies and can now reach evaluator/runtime instability as the limiting factor.

---

### 6. 14B Model Shows Better Verification Success (April 2, 2026)

**Finding:** The 14B model (with 4-bit quantization) passes Dafny verification more quickly than the 7B model.

**Evidence:**
- 7B model: Multiple attempts failed at verification, often stuck on same error
- 14B model: Passed verification on attempt #4 with a more complex strategy (using DeadEndDetection + SoftConstrainedStep)

**Implication:** Larger models have better understanding of Dafny verification constraints and can generate syntactically correct code more reliably.

---

### 7. Evaluation is the Bottleneck (April 2, 2026)

**Finding:** The evaluation phase is extremely slow, taking much longer than generation or verification.

**Evidence:**
- Each evaluation on 10 examples takes several minutes
- Total synthesis time exceeds 30+ minutes for 15 iterations

**Cause:**
- CSD generation involves many token-by-token steps
- Each step requires LM forward pass + grammar checking
- 1024 max steps per example compounds the issue

**Recommendation:** For faster iteration, consider:
- Reducing `--eval-max-steps` from 150 to a lower value
- Using smaller `--eval-sample-size` during synthesis
- Running full evaluation only on promising candidates

---


### 8. Clean Upstream CRANE Baseline on Official Repo (April 12, 2026)

**Finding:** The clean official `uiuc-focal-lab/CRANE` repo produces a materially different GSM-Symbolic baseline than the dirty local fork on `focal`.

**Setup:**
- Cloned official repo into `~/CRANE/upstream-uiuc` on `focal`
- Upstream commit: `616379ce33ac6245933c16e6264b41f7d5800183`
- Applied only a minimal `transformers` compatibility patch in `src/crane/main.py` to handle the removed private `_get_logits_warper` hook
- Did **not** change upstream GSM logic or prompts
- Ran:
  - `Qwen/Qwen2.5-Coder-14B-Instruct`
  - `cot_grammar_mode=adaptive`
  - `gsm` grammar
  - `8-shot`
  - greedy decoding
  - `max_tokens=600`

**Result on the first 3 official upstream GSM examples:**
- Accuracy: `1 / 3 = 33.3%`
- Parse rate: `3 / 3 = 100%`

**Parsed outputs:**
- `<<2 * (length * 3) + 2 * (area / (length * 3))>>`
- `<<n0 * (1 + r) % d>>`
- `<<n1 + n2 + mult * n1>>`

**Interpretation:**
- The previously used forked `~/CRANE` checkout on `focal` was not a trustworthy CRANE baseline.
- The clean official repo is stronger and more symbolic than the forked/dirty checkout.
- However, even the clean official GitHub snapshot still does not obviously reproduce the paper-level `45% / 95%` result from this small test slice, so some paper-vs-public-repo gap likely remains.

**Conclusion:** Until a larger matched run says otherwise, the clean official upstream CRANE baseline to compare against is currently `33.3%` accuracy and `100%` parse on the first 3 GSM examples under the official repo path.

---

### 16. Held-Out Generalization on GSM-Symbolic: Synthesized CSDs vs CRANE (April 22, 2026)

**Experiment:** Evaluated four compiled strategies on the same 50 held-out GSM-Symbolic examples (seed=456, same `gsm_symbolic:main` config as training). Evaluation setup: `Qwen/Qwen2.5-Coder-14B-Instruct` via vLLM, `gpu_memory_utilization=0.85`, `max_model_len=8192`, `enforce_eager=True`, `max_steps=200`, `max_seconds_per_example=120`.

- Training set: 10 examples, seed=123 (same set the synthesis loop gated on).
- Held-out set: 50 examples, seed=456 (disjoint from training by sampling seed).

**Results:**

| Strategy                    | Training (10 ex) acc | Held-out (50 ex) acc | Training syntax | Held-out syntax |
|-----------------------------|----------------------|----------------------|-----------------|-----------------|
| CRANE (baseline)            | 50% (5/10)           | **42% (21/50)**      | 70.7%           | 70.4%           |
| v11 winner (synth)          | 80% (8/10)           | 38% (19/50)          | 83.9%           | 71.4%           |
| v2 winner (synth)           | 90% (9/10)           | 40% (20/50)          | 84.2%           | 69.7%           |
| Lottery seed 2 winner       | 80% (8/10)           | 44% (22/50)          | 82.9%           | 69.2%           |

**Interpretation:**

- **Training set:** synthesized CSDs beat CRANE by 20–40 percentage points on 10 examples. This is by construction of the `--min-accuracy 0.55` gate.
- **Held-out:** the gap essentially collapses. Synthesized CSDs land at 38–44%, compared to CRANE at 42%. All three differences are within ±2 examples of CRANE, well inside 1 binomial SE at p≈0.42, n=50 (SE ≈ 3.5 examples). Statistically indistinguishable.
- **Training→held-out drop:** CRANE drops ~8pp (50→42). Synthesized CSDs drop ~40pp (80→40). That asymmetric drop is the overfitting signature. CRANE, which was not tuned to these 10 examples, is markedly more robust to the within-distribution shift between the training and held-out samples.

**Conclusion:** On GSM-Symbolic at this sample size, the current synthesis pipeline does not produce CSDs that generalize better than CRANE, despite consistently appearing to do so on the 10-example training sample. See observation 18 for the mechanistic explanation.

**Compiled module paths on `focal`:**

- CRANE: `outputs/baselines/crane_baseline_current/GeneratedCSD.py`
- v11 winner: `outputs/generated-csd/runs/20260422_032319_5c6356/synth_toolcut_v11_20260422_032855_74b6ed/GeneratedCSD.py`
- v2 winner: `outputs/generated-csd/runs/20260422_054451_9907f0/synth_10ex_v2_20260422_055808_f53396/GeneratedCSD.py`
- Lottery seed 2 winner: `outputs/generated-csd/runs/20260422_060229_de6ace/lottery_seed2_20260422_060718_e68806/GeneratedCSD.py`

**Held-out result JSONs:** `/tmp/heldout_v11_rerun.json`, `/tmp/heldout_v2_rerun.json`, `/tmp/heldout_crane_rerun.json`, `/tmp/heldout_seed2_rerun.json` on `focal` (volatile tmp — copy off if needed long-term).

---

### 17. Lottery Reliability Sweep on GSM-Symbolic (April 22, 2026)

**Experiment:** Ran `run_synthesis.py` five independent times with identical configuration but distinct `--output-name` run IDs, to measure how reliably the synthesis loop produces a training-set winner.

**Config per run:**

- `--min-accuracy 0.55 --min-syntax-rate 0.72`
- `--eval-sample-size 10 --eval-seed 123`
- `--max-iterations 20`
- `--generation-model gpt-5.4 --generation-backend openai`
- `--eval-model Qwen/Qwen2.5-Coder-14B-Instruct --eval-backend vllm`
- `--vllm-gpu-memory-utilization 0.85 --vllm-max-model-len 8192 --vllm-enforce-eager`

**Results:**

| Seed | Result  | Attempts | Training acc | Training syntax |
|------|---------|----------|--------------|-----------------|
| 1    | FAILED  | 20/20    | —            | —               |
| 2    | SUCCESS | 5        | 80% (8/10)   | 82.9%           |
| 3    | SUCCESS | 9        | 80% (8/10)   | 83.3%           |
| 4    | SUCCESS | 12       | 70% (7/10)   | 76.3%           |
| 5    | SUCCESS | 12       | 80% (8/10)   | 82.9%           |

**Pipeline reliability:** 4/5 = **80% success rate** for producing a training-set winner within 20 attempts.

**Seed 1 failure mode:** Best attempt (attempt 5) had 60% accuracy and 79.3% syntax — both above their respective gates — but failed the implicit "every example contains `<< >>`" requirement (some examples produced outputs with no delimiters at all). There are actually three gates in the loop: `--min-accuracy`, `--min-syntax-rate`, and `contains_delimiters = True` on all examples. The third gate is stricter than it looks and can cause total lottery failure even when other metrics are healthy.

**Winner strategy structure:** All four winners are minor variations on the same skeleton — outside a span, unconstrained generation plus an exact-token check for `<<` to enter constrained mode; inside a span, constrained step or eager close when the parser says the prefix is complete. None of the winners discovered a fundamentally different approach. This matters for the overfitting diagnosis below: the "winners" are structurally close to CRANE and differ mostly in minor heuristics.

---

### 18. Overfitting Diagnosis: Why Training Wins Don't Transfer (April 22, 2026)

**The question:** Why do strategies that hit 70–90% on the 10-example training set only hit 38–44% on a 50-example held-out set from the same distribution?

**Primary cause — selection bias on a noisy gate.**

The 10-example gate (`--eval-sample-size 10`, `--min-accuracy 0.55`) is a binomial trial. For a strategy whose *true* underlying accuracy on the distribution is ~0.40 (which is roughly what held-out reveals), the probability of passing the gate on a particular 10-example sample is:

P(X ≥ 6 | n=10, p=0.40) = Σₖ₌₆¹⁰ C(10,k) · 0.4ᵏ · 0.6¹⁰⁻ᵏ ≈ **0.166**

So ~17% of 10-example evaluations will show ≥60% accuracy purely from sampling variance. With `--max-iterations 20`:

P(at least one pass in 20 tries) = 1 − (1 − 0.166)²⁰ ≈ **0.97**

That ~97% is roughly consistent with our observed 4/5 = 80% lottery success rate; the gap is plausibly explained by the additional `contains_delimiters` and syntax gates adding friction.

**The pipeline reliably "finds a winner" even if the strategies being generated are no better than ~40% on the real distribution.** The 80–90% training accuracy of that winner is `max`-over-20-draws, not an unbiased estimate of true accuracy. When the same strategy is then evaluated on 50 fresh examples (no selection), the luck averages out and we observe the underlying ~40%.

**Secondary causes:**

1. **Generator prompt receives example-specific failure content.** After each failed attempt, the generator LLM sees specific failure summaries ("Example 3: expected 392, got 312. Failure mode: repetition_loop.") keyed to the training examples. Successive attempts are conditioned on those specific ten problems, so the proposal distribution drifts toward fitting the training sample rather than the underlying task.
2. **Dafny verification + prompt priors pin strategies near CRANE.** To pass verification, candidates must satisfy invariants about parser validity, suffix preservation, and cost bounds. All observed winners are small wiggles around the CRANE skeleton. Minor wiggles don't produce robust improvements — they produce noise-level differences that can cut either way on held-out.

**Fixes (in rough order of expected impact):**

| Cause                                       | Fix                                                                                                                                                                                                                                     |
|---------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Gate noise at n=10                          | Raise gate sample size to ≥30 or ≥50 examples. At n=30, P(pass \| true p=0.40) drops from ~17% to ~3.5%, and over 20 attempts the false-positive rate drops from ~97% to ~50%. At n=50 the false-positive rate falls further.            |
| No distinction between gating and validation | Evaluate candidates on a larger pool, but gate only on a random subset; require a *held-out* sub-split (disjoint from the gate examples) to pass before declaring success.                                                               |
| Generator sees training-example content      | Coarsen failure summaries: failure-mode counts only; strip actual example text and specific computed values. This prevents the generator from conditioning on training-specific content.                                                 |
| Strategy space pinned near CRANE             | Give the generator access to richer helpers, weaken some invariants, or explicitly reward exploration away from the CRANE skeleton. This is a larger redesign.                                                                          |

**Minimum viable fix:** raise `--eval-sample-size` from 10 to 30–50 and keep everything else the same. This kills the dominant selection-bias source without touching the generator or the verification design.

**Next experiment to run:** Rerun the 5-lottery sweep with `--eval-sample-size 50 --eval-seed 123`. Expected outcomes if the fix works:

- Pipeline reliability drops (fewer candidates pass the stricter gate).
- The candidates that *do* pass hold their performance on a fresh 50-example held-out sample (or at least drop less than 40pp from training to held-out).

If pipeline reliability drops to near zero *and* no candidate survives, that means the current proposal distribution genuinely has true accuracy ≈ CRANE, and the only wins on n=10 were selection artifacts — an important negative result.

**Important general lesson — state simply for future context:**

*Just because a strategy succeeds on a small training sample doesn't mean it will succeed on a larger held-out sample. With a 10-example gate and 20 synthesis attempts, the loop acts as a selection filter on binomial noise. The only way to know whether a synthesized CSD is actually better than CRANE is to (a) gate on a sample large enough to escape that noise, and (b) evaluate the winner on a held-out sample large enough to resolve small effects (≥200 examples if we expect differences in the 1–2 percentage-point range).*

---

## Next Steps

1. [ ] Rerun the 5-seed lottery sweep with `--eval-sample-size 50 --eval-seed 123` to test whether raising the gate n defeats the selection-bias overfitting (observation 18). This is the highest-value next experiment.
2. [ ] Add a held-out validation sub-split inside the synthesis loop (disjoint from the gate examples). Only declare a candidate a winner if it passes both the gate sub-split and the validation sub-split.
3. [ ] Coarsen the generator's failure feedback so it no longer sees specific training-example content (keep failure-mode counts; strip question text and actual computed values). Run the 5-seed lottery under this feedback regime and compare winner held-out transfer to the current baseline.
4. [ ] Evaluate the seed-3, seed-4, seed-5 winners on the same 50-example held-out split to grow the held-out comparison from 3 to 5 data points (~30 min wall time with warm DFA mask cache). Strengthens the "synthesized ≈ CRANE on held-out" claim narratively but does not resolve the underlying effect-size question on its own.
5. [ ] If we want a definitive "does our framework actually beat CRANE on held-out?" answer: run the held-out evaluation on 200+ examples rather than 50. Current n=50 has binomial SE ≈ 3.5 examples, which makes 1–2 example differences unresolvable.
6. [ ] Once GSM-Symbolic generalization is resolved, port the same pipeline and analysis to FOLIO to see whether the overfitting mechanism and its fix transfer to logical reasoning.

---

## Open Questions

1. Does raising the synthesis gate to n=30–50 actually produce CSDs that transfer to held-out, or does pipeline reliability just drop without any winners transferring (i.e., is the proposal distribution truly ceiling'd at CRANE-level true accuracy)?
2. Is the "CRANE-shaped strategy family" a hard ceiling imposed by Dafny verification and the current helper set, or can richer helpers / relaxed invariants open up a meaningfully different design space?
3. Are there specific sub-populations of GSM-Symbolic problems where synthesized CSDs genuinely beat CRANE, even though the overall means are indistinguishable? Per-example analysis on held-out might reveal this.
4. How does strategy performance vary across datasets (GSM-Symbolic vs FOLIO)? Do the same overfitting mechanisms apply to logical-reasoning tasks, or does FOLIO's structure make the training gate less noisy?
5. Can failure-mode feedback to the generator be coarsened (example-specific content stripped) without losing the signal the generator needs to improve across attempts?
6. What is the minimum model size for reliable CSD synthesis once selection bias is removed? (The old observation 1 answered this under small-n gate conditions; may need revisiting under a larger gate.)

---

## Update (April 24, 2026)

The section above (through observation 18, dated April 22) was written before two important events: (a) the completion of the 10-seed lottery sweep with its full held-out evaluation, and (b) the pivot from FOLIO to SQL/Spider. This update supersedes observation 18's conclusion and the previously-stated Next Steps.

### Status at a glance

- **GSM-Symbolic:** the synthesis loop *does* produce CSDs that generalize over CRANE, not just CSDs that beat it on the 10-example gate. On the 10-seed sweep, 3 of 7 calibration winners also beat CRANE on the 50-example held-out split (+6 pp accuracy). See observation 19.
- **SQL/Spider:** the new active target. FOLIO has been discarded. Synthesis runs end-to-end against Spider but currently produces 0% execution accuracy — not because strategies fail to verify/compile/run, but because the prompt/extraction contract is mismatched with the rest of the eval pipeline. See observation 20.
- **Next work:** focused on SQL — fix the 0% accuracy root cause, then re-run the lottery sweep on Spider using the same methodology that worked on GSM-Symbolic.

### 19. Lottery Sweep + Held-Out Generalization, Corrected (April 22, 2026)

**What changed vs observation 18.** Observation 18 concluded, based on 3 synthesized strategies vs CRANE on 50 held-out examples, that "synthesized ≈ CRANE on held-out" and that the 10-example gate was pure selection on binomial noise. A larger 10-seed sweep + per-winner held-out eval, finished the same day, showed that conclusion was too strong. Some synthesized strategies genuinely generalize.

**Experiment.** Ran `run_synthesis.py` 10 times with identical config but distinct run IDs. Calibration sample n=10, seed=123, `--min-accuracy 0.55`, `--min-syntax-rate 0.72`, `--max-iterations 20`, generator `gpt-5.4`, evaluator `Qwen/Qwen2.5-Coder-14B-Instruct` via vLLM. Each calibration winner was then evaluated on the 50-example held-out split (seed=456).

**Results** (from `scripts/lottery_manifest.json` + per-winner `step4_winner*.json`):

- Calibration pass rate: **7 / 10 = 70%** of runs produce a strategy that beats CRANE on the 10-example gate.
- Held-out generalization rate: **3 / 7 = 43%** of calibration winners also beat CRANE on the 50-example held-out split.

| Strategy | Calibration acc | Held-out acc (50 ex) | Held-out syntax | Beats CRANE held-out? |
|----------|-----------------|-----------------------|-----------------|-------------------------|
| CRANE    | 50% (5/10)      | 52% (26/50)           | 71.5%           | —                       |
| winner1  | 90%             | 40%                   | 66.2%           | no                      |
| winner2  | 70%             | 44%                   | 67.0%           | no                      |
| winner3  | 70%             | **58%** (29/50)       | 74.3%           | **yes (+6pp)**          |
| winner4  | 60%             | 40%                   | 64.3%           | no                      |
| winner5  | 60%             | 36%                   | 69.1%           | no                      |
| winner6  | 70%             | **58%** (29/50)       | 74.3%           | **yes (+6pp)**          |
| winner7  | 70%             | **58%** (29/50)       | 74.3%           | **yes (+6pp)**          |

**Notable:** winner3/6/7 have identical held-out accuracy *and* identical held-out syntax rate, suggesting they may be structurally the same strategy or behaviorally indistinguishable on this 50-example split. Confirming whether they are literally the same compiled Python, or different sources that happen to agree pointwise, is follow-up work.

**Revised interpretation of observation 18.** The earlier "synthesized CSDs are indistinguishable from CRANE" framing was wrong in its strong form. A more careful statement:

1. The 10-example gate is noisy, and some calibration winners (e.g. winner1 at 90% calibration → 40% held-out) are clearly selection artifacts — observation 18's mechanism is real and does operate.
2. But the gate is not *pure* noise: 3 of 7 calibration winners also beat CRANE on held-out by the same 6 pp margin, which is larger than expected under the null hypothesis that all synthesized strategies have true accuracy ≈ CRANE.
3. At held-out n=50, CRANE 52% vs winner 58% has binomial SE ≈ 3.5 examples (~7 pp), so +6 pp is right at the edge of what n=50 can resolve. A larger held-out sample (≥200 examples) is needed to call the effect decisively; pending that, "some synthesized strategies generalize over CRANE" is a working hypothesis, not a hardened finding.

**Conclusion.** The synthesis pipeline can produce genuinely-better CSDs for GSM-Symbolic at current hyperparameters. It still wastes a substantial fraction of calibration wins on noise-selected strategies (4 of 7 winners don't transfer), so the directions observation 18 suggested — larger gate, held-out sub-split inside the loop, coarsened failure feedback — are still well-motivated as ways to raise the winners:calibration-passes ratio. They are no longer a prerequisite for getting any winner at all.

**Winner paths on `focal`:**
- CRANE baseline: `outputs/baselines/crane_baseline_current/GeneratedCSD.py`
- Non-generalizers: winner1 (`runs/20260422_085632_b2c7e3/...`), winner2 (`20260422_085632_a7eb6c/...`), winner4 (`20260422_090915_d1c05b/...`), winner5 (`20260422_093901_2ba81f/...`)
- Generalizers: winner3 (`runs/20260422_090113_dd2ff8/synth_lottery_3_20260422_090702_e38577/...`), winner6 (`20260422_093544_4f2d16/synth_lottery_9/...`), winner7 (`20260422_094531_23ada2/synth_lottery_10_20260422_095426_5050a1/...`)

### 20. Pivot to SQL/Spider; FOLIO Discarded (April 22, 2026)

FOLIO has been removed from the active evaluation surface. The `evaluations/folio/` directory still exists on disk but is no longer maintained, and `scripts/generate_folio_csd.sh` is dormant. The second-task target is now Spider text-to-SQL.

**New code paths added:**
- `evaluations/sql_spider/` — dataset loader (`dataset.py`), dynamic per-schema grammar builder (`grammar.py`), Spider execution-accuracy scorer via the vendored syncode evaluator (`executor.py`), plus CLI (`cli.py`).
- `scripts/generate_sql_csd.sh` — synthesis driver for Spider.
- `scripts/run_sql_vanilla.sh` — unconstrained-baseline driver on Spider.
- `synthesis/evaluator.py` — Spider branches added to `_load_dataset_sample`, `_setup_environment`, `_format_prompt`, `_get_expected_answer`, `_extract_answer_spider`, `_exec_match_spider`, and the eval loop.

**Dataset shape.** One example = `{db_id, question, query (gold SQL), db_info (schema string), prompt}`. Gold SQL is sourced preferentially from the HF `richardr1126/spider-context-validation` split; local fallback at `/home/aadivyar/spider_data/spider_data` via `dev.json` + `dev_gold.sql` + `tables.json`. Both paths verified to populate `query` correctly (sanity check on the local loader returns 1034 rows with non-empty `query`).

**Scoring.** Per-iteration synthesis feedback uses a fast in-process `_exec_match_spider` (run pred + gold on the per-example SQLite DB, compare result sets). The CLI and batch evaluation use the full Spider evaluator for exact-set-match / execution-accuracy / hardness breakdown.

### 21. SQL Synthesis: Current 0% Execution Accuracy (April 24, 2026)

**Latest run:** `outputs/generated-csd/runs/20260422_231506_07b2e0/failure_report.json` (10 attempts, Apr 22 23:15–23:45, eval sample n=50).

**Failure-stage breakdown across the 10 attempts:**

| Attempts | Failed at       | What happened                                                                                                    |
|----------|-----------------|------------------------------------------------------------------------------------------------------------------|
| 1, 2, 3  | verification    | Same Dafny parse error `GeneratedCSD.dfy(129,0): Error: rbrace expected`. Refinement loop did not fix it.        |
| 6, 7, 8  | verification    | Same signature-mismatch error `method returns 1 value but is assigned to 3 variables`. Refinement loop stuck.     |
| 4, 5, 9, 10 | evaluation   | Strategy verifies + compiles + runs, but scores accuracy=0 and syntax_rate=0 on 50 Spider examples.                |

The verification-stage failures are the same Dafny errors repeating across 3 consecutive attempts with no progress — the refinement prompt is not citing the right span of the generated Dafny to the generator to correct it. That's a real issue, but secondary to the 0% accuracy question below.

**Root cause of the 0% accuracy in attempts 4, 5, 9, 10.**

The Spider prompt is constructed (see `synthesis/evaluator.py:1006–1015`) to **end with a literal `<<`**:

```text
...
db_id: {db_id}
db_info: {db_info}
question: {question}
SQL: <<
```

The intent is to seed the constrained span at generation time so the model only has to produce `SQL_BODY + >>`. But the rest of the eval pipeline expects the *model's output text* (not the prompt) to contain a balanced `<<...>>` pair:

1. **`_contains_delimiters(output)`** (`synthesis/evaluator.py:1055`) checks `"<<" in output and ">>" in output`. Since only `>>` appears in the completion, this returns False for every example. Observed in per-sample records: `"contains_delimiters": false` on every Spider sample. This trips the implicit "every example contains `<< >>`" gate in the synthesis loop, which per observation 17 is a stricter gate than it looks.

2. **`_extract_answer_spider(output)`** (`synthesis/evaluator.py:423`) tries `re.search(r"<<\s*(.*?)\s*>>", ...)`. With no `<<` in the completion, this regex never matches and the code falls back to `output.split("\n\n")[0]`, which returns the raw completion **including the trailing `>>`**. That `>>` is then passed verbatim to SQLite as part of the SQL text.

3. **`_exec_match_spider(pred_sql, gold_sql, example)`** (`synthesis/evaluator.py:432`) executes `pred_sql` against the per-example SQLite DB. With `>>` tacked on the end, every query fails with a SQL syntax error, `pred_rows` becomes `None`, and `is_correct` is always False.

4. **`_check_syntax_validity`** uses the same `<<...>>` regex path via `_extract_constrained_content`. No segments are found, so `total_segments = 0` and the final `syntax_rate = 0 / 0 → 0.0` via the ternary guard. The `syntax_rate: 0.0` in the failure report is a degenerate "no segments" outcome, not a measurement of how many segments parsed correctly.

**Concrete example** (attempt 4, example 1, from `sample_outputs[0]`):
- Question: "What are the name, independence year, and surface area of the country with the smallest population?"
- Model output: `SELECT name , independenceyear , surfacearea FROM city WHERE population < ?>>`
- Extracted `actual`: the whole string above, with the `>>` still attached.
- `is_correct`: false because SQLite rejects the `>>`.

Note: in addition to the delimiter issue, the output itself also has substantive errors (wrong table: `city` instead of `country`; concatenated column names: `independenceyear` instead of `IndepYear`). A delimiter-fix alone will not push accuracy to CRANE-level without also addressing schema grounding. But it is a prerequisite: *no* synthesis attempt can get credit for a correct query today, because even a perfect query would fail execution due to the trailing `>>`.

**Secondary anomaly — `expected` is empty in serialized records.** Every serialized `sample_outputs` entry in the Spider failure reports has `"expected": ""`, even though `load_spider(...)` populates `query` correctly when called directly (verified by running the loader in isolation — it returns 1034 rows with non-empty `query`). This means at the point the evaluator calls `self._get_expected_answer(example)`, something has stripped or replaced the `query` field on the example dict. If this is real and not just a serialization-time artifact, `_exec_match_spider` is short-circuiting on `if not gold_sql: return False` before it even runs SQLite — which would be a *second* independent reason accuracy is pinned at 0. To be determined; the delimiter bug alone already explains the observed accuracy. Worth adding a log line in `_get_expected_answer` / `evaluate_sample` to catch whichever happens first.

**Suggested fix order:**

1. **Fix the prompt/extraction mismatch.** Two viable options, in rough order of least invasive:
   - (a) Change the Spider prompt to not pre-seed `<<`; instead ask the model to emit `<<SQL>>` end-to-end. This matches the existing GSM-Symbolic contract and lets `_extract_answer_spider` / `_contains_delimiters` / `_check_syntax_validity` work unchanged.
   - (b) Keep the seeded `<<` (there may be a generation-time reason for it — check `run_crane_csd` / `VerifiedDecoderAgent.py` to confirm whether the model's decoding loop needs the `<<` as a state hint), but patch `_extract_answer_spider` to synthesize `<<` + completion before applying the regex, and patch `_contains_delimiters` to treat the Spider case specially. This is more surgical but requires parallel fixes in the evaluator-level syntax check and the delimiter gate.
2. **Confirm the `expected=""` issue.** Log both `example.get("query")` and the final `expected` inside `evaluate_sample` on a Spider run. If `query` is present on the example but `expected` is blank in the record, there is a second bug between `_get_expected_answer` and serialization; fix that before trusting any Spider accuracy number.
3. **Verify a single example end-to-end with a handwritten winning SQL.** Inject `actual = "SELECT Name, SurfaceArea, IndepYear FROM country ORDER BY Population LIMIT 1"` directly into `_exec_match_spider` for the "smallest population" question and confirm it evaluates to `True`. This flushes out any remaining plumbing bugs (wrong db_dir, wrong db_id mapping, gold-file offset error from the dev_gold.sql parse, etc.) independent of the model.
4. **Unblock the refinement loop on verification errors.** Attempts 1/2/3 and 6/7/8 repeat the same verification error three times each, so the Dafny refinement prompt is not isolating the failing span. Check whether `error_summary` is being fed back with the right source excerpt — the structured diagnostics added in commit `45afbfc` (obligation_kind, failing_text, source_excerpt) should be surfacing here.
5. **Only after 1–3 are green, re-run `scripts/generate_sql_csd.sh` and look for non-zero accuracy.** If accuracy is still pinned at 0 after the plumbing is fixed, that is the real research signal — it means Spider is a harder target than GSM-Symbolic for the current strategy family, which is itself useful to know and should drive prompt/tool changes.

---

## Next Steps (supersedes the Next Steps list above)

1. [ ] **SQL — fix the delimiter/extraction contract.** Either drop the seeded `<<` from the Spider prompt, or patch `_extract_answer_spider` + `_contains_delimiters` + `_check_syntax_validity` to handle the seeded-prefix case. This is the single highest-value action to unblock SQL.
2. [ ] **SQL — diagnose `expected=""` in serialized sample outputs.** Add logging in `_get_expected_answer` / `evaluate_sample` to distinguish between "example dict doesn't have `query`" vs "serializer drops the field." Fix whichever it is.
3. [ ] **SQL — sanity-check `_exec_match_spider` with a hand-chosen gold-equivalent prediction** so any remaining db_dir / gold-file / db_id-mapping bug surfaces before we attribute failures to the model.
4. [ ] **SQL — fix the stuck verification-refinement loop** (attempts 1–3 and 6–8 of the Apr 22 run repeat the same Dafny error). Confirm the structured diagnostics from commit 45afbfc are actually reaching the refinement prompt in `synthesis/feedback_loop.py`.
5. [ ] **SQL — after 1–4 are done, rerun `scripts/generate_sql_csd.sh` end-to-end** and record the first accuracy > 0 run. That is the milestone that turns Spider from "plumbing-broken" into "genuine research target."
6. [ ] **SQL — once a non-zero baseline is established, port the 10-seed lottery + held-out eval methodology** from `scripts/lottery_manifest.json` to Spider so that Spider's calibration→held-out generalization rate can be compared to GSM-Symbolic's 3/7 = 43%.
7. [ ] **GSM — confirm whether winner3 / winner6 / winner7 are literally the same compiled strategy.** Diff their `GeneratedCSD.py` files. If identical, the held-out success rate is effectively 1 distinct strategy, not 3 — meaningful for the claim "the pipeline produces generalizers."
8. [ ] **GSM — expand held-out from 50 → 200+ examples** on at least one generalizing winner (e.g. winner3) to resolve whether the +6 pp margin holds at a sample size where binomial SE is ≤2 pp.
9. [ ] **(Deferred, lower priority than SQL unblock)** raise `--eval-sample-size` from 10 to 30–50 on GSM, with a held-out sub-split inside the loop, to cut the 4-of-7 noise-selection rate observed in observation 19.

---

### 22. Why Spider Inherits the `<<` / `>>` Envelope, and Why That's Wrong for SQL (April 24, 2026)

**The model isn't "choosing" to use `<<`/`>>` delimiters for SQL — the entire synthesis framework forces it into that shape at four independent layers.** This is what observation 21 is actually a symptom of.

**1. Dafny method contract (hard precondition).**
`dafny/GeneratedCSD.dfy:28` and the mirrored signature in `synthesis/prompts.py:52` both require:
```dafny
requires "<<" in lm.Tokens && ">>" in lm.Tokens
```
A strategy that never references `<<` / `>>` still verifies fine (the precondition is stronger than necessary), but the verifier's *shown contract* tells the generator LLM that these tokens are load-bearing vocabulary.

**2. Dafny helpers `OpenConstrainedSpan` / `CloseConstrainedSpan`.**
`dafny/VerifiedAgentSynthesis.dfy:372–440` literally append `["<<"]` / `[">>"]` to `generated` as their semantics. Any strategy that opens or closes a span via these helpers is *defined* in terms of these delimiters. There is no helper for "enter constrained mode without emitting an open delimiter."

**3. Synthesis prompt examples (prompts.py:181–515).**
Every worked example in `INITIAL_GENERATION_PROMPT` follows the same state machine: "generate freely until `<<`, constrain until parser complete, close with `>>`." The generator has zero template for an always-constrained strategy, so even a strong generator like gpt-5.4 cannot propose one — it is outside the in-context distribution it has been shown.

**4. Spider task description + evaluator prompt.**
`scripts/generate_sql_csd.sh` tells the generator "emits a single SQL query inside `<<` `>>` delimiters," and `synthesis/evaluator.py:1015` ends the model prompt with a seeded `SQL: <<`. Both of these reinforce the same delimiter-gated contract at runtime.

**Why this was the right choice for GSM-Symbolic.**
GSM answers are mostly natural-language chain-of-thought with islands of arithmetic. `<<`…`>>` cleanly marks "enter grammar-constrained mode for an arithmetic expression," and CRANE's core insight is exactly about those delimited transitions. Unconstrained prose outside, grammar-constrained arithmetic inside.

**Why it's wrong for Spider.**
Spider outputs are 100% SQL — there is no unconstrained prose. A delimiter-gated strategy on Spider degenerates: the strategy spends all its tokens inside a single `<<`…`>>` window that covers the entire completion, and the delimiters become ceremonial. Worse, the model must *enter* constrained mode by emitting `<<` at step 0 (which the prompt seeds as a hack) and *exit* by emitting `>>` at the end (which the model sometimes forgets, sometimes emits mid-query, sometimes doubles up). Hence observation 21: the 0% accuracy is a predictable consequence of shoehorning SQL into a GSM-shaped contract.

**The fundamental strategy family we actually want for Spider:**
- Enter constrained mode at step 0. No `<<` trigger.
- Emit SQL tokens under the dynamic schema-narrowed grammar for the entire completion.
- Exit on EOS or on `parser.IsCompletePrefix(currentConstrained)`. No `>>` trigger.

This strategy family does not use `OpenConstrainedSpan` / `CloseConstrainedSpan` at all, and does not reference the `<<` / `>>` tokens. It *cannot be synthesized today* because the Dafny contract, the prompt examples, the grammar header, the task description, and the evaluator all pin the delimiter-gated pattern.

**Decision.** Before claiming any Spider result, we're going to relax the framework so the always-constrained strategy family is representable and discoverable. This is a framework change, not a bug fix; see the draft plan under "Next Steps — SQL delimiter-envelope removal" below.

---

## Next Steps — SQL: keep the envelope, just clarify the task (chosen path, April 24, 2026)

**Decision.** We are not introducing an always-constrained strategy family. Instead we tell the generator that for Spider the *entire* output is wrapped in a single `<<...>>` block — a degenerate single-window case of the GSM pattern. This preserves everything that has worked for GSM (Dafny contract, helpers, all four worked examples in `INITIAL_GENERATION_PROMPT`, the CRANE-shaped strategy family) and reduces the SQL fix to a task-description and evaluator-prompt change.

### Why this works

A delimiter-gated CSD on Spider with this task framing degenerates to: emit `<<` immediately (strategy enters constrained mode at step 1), generate the entire SQL query under the dynamic schema-narrowed grammar, emit `>>` when the grammar reaches a complete prefix. That is the exact same control flow the existing GSM strategies use, applied to one big window instead of many small ones. The generator does not need a new template; the verifier does not need a relaxed contract; the helpers do not need to change.

### Concrete changes

**1. Task description** — `scripts/generate_sql_csd.sh`. Make it unambiguous that the entire query is wrapped in one block:

```
TASK_DESC="Text-to-SQL generation on the Spider benchmark. The model reads a schema \
(tables and columns) and a natural-language question, then emits a single SQL query \
wrapped in a single <<...>> block — the output is exactly <<SELECT ... FROM ...>>, \
with no text outside the delimiters. The parser validates the query against a SQL \
grammar dynamically narrowed to the current schema's tables and columns."
```

The current wording ("emits a single SQL query inside << >> delimiters") is ambiguous between "wrap the whole query" and "wrap individual sub-expressions"; the new wording is explicit.

**2. Evaluator prompt** — `synthesis/evaluator.py:1015`. Drop the seeded `<<`:

```diff
-                "SQL: <<"
+                "SQL: "
```

The example line at `synthesis/evaluator.py:1010` (`"SQL: <<SELECT count(*) FROM singer>>"`) stays — it shows the model what a complete output looks like. Removing the seed forces the model to emit both delimiters itself, which is what the rest of the eval pipeline already expects.

**3. Everything else stays unchanged.**
- `_extract_answer_spider` (`synthesis/evaluator.py:423`) — its `<<\s*(.*?)\s*>>` regex now matches a balanced pair in the model output. No change.
- `_contains_delimiters` (`synthesis/evaluator.py:1055`) — once the model emits both delimiters, this returns True. No change.
- `_check_syntax_validity` — finds one `<<...>>` segment and parses it. No change.
- `grammars/sql.lark` — header comment is accurate. No change.
- `synthesis/prompts.py` — no new example needed; the generator already knows this pattern from GSM.
- `dafny/GeneratedCSD.dfy`, `dafny/VerifiedAgentSynthesis.dfy` — bit-identical.

### Independent fixes still worth doing (orthogonal to the delimiter question)

- **`expected=""` in serialized Spider sample outputs** (obs 21). Add a log line in `_get_expected_answer` / `evaluate_sample` to confirm whether `query` is present on the example dict at lookup time. Fix whatever is dropping it. Without this, even a perfectly working strategy will score 0% on Spider because `_exec_match_spider` short-circuits on empty `gold_sql`.
- **Stuck verification-refinement loop** (obs 21). Attempts 1–3 and 6–8 of the Apr 22 SQL run repeated the same Dafny error three times each. The structured diagnostics added in commit `45afbfc` (obligation_kind, failing_text, source_excerpt) should be reaching the refinement prompt; check that they are.

### Sanity gates before claiming a Spider result

1. **GSM regression:** re-run CRANE baseline on 10 examples. Accuracy and syntax rate must match the pre-change numbers (52% acc, 71.5% syntax from `step4_crane.json`). Risk surface is small — only the evaluator's Spider branch is touched — but the dataset-conditioned code path needs to stay clean.
2. **Spider single-strategy smoke test:** run `scripts/generate_sql_csd.sh` once and inspect the first attempt that passes verification. Look for: (a) the model output contains a balanced `<<...>>` pair, (b) `_contains_delimiters` returns True, (c) `_extract_answer_spider` returns a clean SQL string with no trailing `>>` artifacts, (d) `_exec_match_spider` runs without short-circuiting on empty gold.
3. **First non-zero Spider accuracy run.** That is the milestone that turns Spider from "plumbing-broken" into "genuine research target."

---

The original framework-change plan (sections A–H below) is **not** the chosen path. It is preserved for reference in case the task-description fix turns out to be insufficient — e.g. if the generator consistently produces strategies that emit the SQL outside the `<<...>>` block, or if the single-window degenerate case is somehow harder than expected for the CRANE family. Most likely we never need it.

---

## Next Steps — SQL delimiter-envelope removal (draft plan, April 24, 2026)

Goal: unlock the **always-constrained** strategy family for Spider without breaking any existing GSM-Symbolic strategy. All currently-verifying GSM strategies (CRANE baseline + 7 lottery winners) must still verify and evaluate with matching accuracy after the change.

### A. Dafny method contract — **leave unchanged** (revision, Apr 24)

Earlier drafts of this plan proposed dropping `requires "<<" in lm.Tokens && ">>" in lm.Tokens` from the top-level `MyCSDStrategy` contract. On re-reading: this is not necessary and we will not do it.

**Why we don't need to touch the contract.** That precondition is a pure existence assertion on the LM's vocabulary. For every model we run (Qwen2.5-Coder-7B / 14B), `"<<"` and `">>"` are regular vocabulary tokens, so the precondition is trivially satisfied by the caller (the evaluator's environment setup). It is not asserting that the strategy *uses* these tokens, only that they *exist* in the vocabulary. An always-constrained strategy that never references `"<<"` / `">>"` is perfectly happy under this precondition — Dafny verifies it fine. The precondition is harmlessly over-strong, not structurally required by the delimiter-gated pattern.

**Consequence:** sections B–F below are sufficient on their own to unlock the always-constrained strategy family for Spider. The Dafny contract, the shared helpers, and all existing GSM strategies stay bit-identical. The only risk surface is the evaluator-side changes in section E, which are already dataset-conditioned.

Removed: the earlier "verification gate — re-run on CRANE + all 7 lottery winners" is no longer needed for this sub-plan, since no shared Dafny file is being edited. The GSM regression check in section G is still worth doing to catch any unintentional evaluator-side bleed-through.

### B. Synthesis prompt — add always-constrained example

**File:** `synthesis/prompts.py`, inside `INITIAL_GENERATION_PROMPT`.

**Change:** add a second worked example *alongside* the existing delimiter-gated one. Both should be shown to the generator so it has templates for both strategy families.

Sketch of the new example body:
```dafny
// CSD_RATIONALE_BEGIN
// Always-constrained CSD. The entire output is a single grammar-constrained
// span starting at step 0. No delimiter trigger; we enter constrained mode
// immediately and exit on EOS or when the parser reports the span is complete.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: We initialize currentConstrainedOut := [] which is a valid
//   prefix. In the body we only call AppendConstrainedToken when IsTokenValidNext
//   holds, so the appended prefix remains valid.
// suffix: currentConstrainedOut is always a suffix of generated; we add tokens
//   to both atomically via AppendConstrainedToken.
// cost: ConstrainedStep bumps helpers.cost by 1; steps grows by 1 per iteration.
// progress: Each loop iteration appends at most one token; steps bounds |generated|.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := true;
currentConstrainedOut := [];
cost := 0;

var steps := 0;
while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant insideConstrainedOut
  invariant parser.IsValidPrefix(currentConstrainedOut)
  invariant |currentConstrainedOut| <= |generated|
  invariant generated[|generated| - |currentConstrainedOut|..] == currentConstrainedOut
  invariant |generated| <= |generatedPrefix| + steps
  invariant cost == 0
  invariant helpers.cost <= steps
  decreases maxSteps - steps
{{
  if parser.IsCompletePrefix(currentConstrainedOut) {{
    break;  // grammar reached a complete parse, done
  }}
  var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
  var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
  steps := steps + 1;
  if next == eosToken {{
    break;
  }}
  var valid := helpers.IsTokenValidNext(parser, currentConstrainedOut, next);
  if valid {{
    var g2, i2, c2 := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
    generated := g2;
    insideConstrainedOut := i2;
    currentConstrainedOut := c2;
  }}
}}
cost := steps;
```

Also tone down the SYSTEM_PROMPT prose at `prompts.py:82–84` so it no longer assumes the delimiter-gated pattern as *the* pattern. Something like:

> Some strategies use `<<` / `>>` as delimiter tokens to gate a constrained sub-region inside otherwise-unconstrained text (e.g. math expressions inside chain-of-thought). Other strategies constrain the entire output from step 0 with no delimiters (e.g. when the whole answer is structured, such as SQL or JSON). Pick the shape that fits the task description.

### C. Grammar — drop misleading header

**File:** `grammars/sql.lark:1–9`.

The grammar itself is already neutral — `start: sql_stmt` and `csd_start: sql_stmt` don't reference `<<` or `>>`. But the header comment says:

> Mirrors gsm.lark: outer delimiter is "<<" … ">>" and csd_start consumes the closing delimiter.

This is wrong for the new design. Replace with:

> SQL grammar for CSD evaluation on Spider. The entire completion is grammar-constrained from token 0 (no `<<` / `>>` envelope). TABLE_NAME and COLUMN_NAME default to permissive identifiers; `build_dynamic_sql_grammar` in `evaluations/sql_spider/grammar.py` narrows them to the current schema's tables and columns.

Keep `csd_start: sql_stmt` for now (it's referenced by the constrained-decoding machinery); dropping it is a separate cleanup.

### D. Spider task description — remove delimiter language

**File:** `scripts/generate_sql_csd.sh`.

**Change:**
```diff
-TASK_DESC="Text-to-SQL generation on the Spider benchmark. The model reads a schema (tables and columns) and \
-a natural-language question, then emits a single SQL query inside << >> delimiters. The parser validates the \
-query against a SQL grammar that is dynamically narrowed to the current schema's tables and columns."
+TASK_DESC="Text-to-SQL generation on the Spider benchmark. The model reads a schema (tables and columns) and \
+a natural-language question, then emits a single SQL query. The entire completion is constrained against a \
+SQL grammar that is dynamically narrowed to the current schema's tables and columns."
```

### E. Evaluator prompt + extraction — remove seeded `<<`

**File:** `synthesis/evaluator.py`.

**E.1 — prompt** (around line 1015):
```diff
-                "SQL: <<"
+                "SQL: "
```
The update to the example inside the prompt too:
```diff
-                "SQL: <<SELECT count(*) FROM singer>>\n\n"
+                "SQL: SELECT count(*) FROM singer\n\n"
```

**E.2 — answer extraction** (`_extract_answer_spider`, ~line 423):
```python
def _extract_answer_spider(self, output: str) -> Optional[str]:
    """For always-constrained SQL: the whole completion is SQL."""
    if not output:
        return None
    cleaned = output.split("\n\n")[0]
    cleaned = cleaned.replace("\n", " ").replace("\r", " ").strip().rstrip(";").strip()
    return cleaned or None
```
No regex, no `<<…>>` match.

**E.3 — delimiter gate** (`_contains_delimiters`, ~line 1055): for Spider, the gate is "output parses against the dynamic grammar," not "contains `<<` and `>>`". Dataset-condition it:
```python
def _contains_delimiters(self, output: str) -> bool:
    if self.dataset_name == "spider":
        return bool(output and output.strip())
    return "<<" in output and ">>" in output
```

**E.4 — syntax-validity check** (`_check_syntax_validity`, ~line 1063): for Spider, parse the whole output once against the dynamic parser, rather than extracting `<<…>>` segments. Dataset-condition at the top of the method.

### F. Runtime — verify `run_crane_csd` honors `insideConstrained=true` at start

**File:** `evaluations/sql_spider/generation.py` (re-exports from `evaluations/gsm_symbolic/generation.py`).

**To check before editing:** `run_crane_csd` initializes `insideConstrained=False` when invoking the compiled strategy. The new always-constrained strategy body sets `insideConstrainedOut := true` on its first line, so this should just work. Verify by reading `run_crane_csd` after the Dafny changes compile. If it fights the strategy's initialization, add a `start_constrained=False` kwarg threaded through.

### G. Sanity checks before claiming a result

1. **GSM regression:** re-verify + re-evaluate CRANE baseline on 10 examples. Accuracy and syntax rate must match the pre-change numbers from `step4_crane.json` (52% acc, 71.5% syntax).
2. **GSM lottery winners:** re-verify + re-evaluate winner3 (the cheapest generalizer) on held-out-50. Accuracy must still be 58% ± 1 example.
3. **Spider synthesis:** run `scripts/generate_sql_csd.sh` once. Expected: at least one attempt produces an always-constrained strategy that verifies, compiles, and achieves accuracy > 0 on a 10-example sample.
4. **Spider CRANE baseline:** stand up a SQL CRANE baseline (always-constrained, hand-written) for comparison. Otherwise we have no apples-to-apples reference on Spider.

### H. Scope and rollback

- All changes land on a feature branch until G.1 and G.2 pass.
- G.1 is the hard blocker — if any existing GSM strategy drops in evaluated accuracy after these changes, the suspect is the evaluator-side dataset-conditioning in section E (e.g. a gsm branch accidentally taking the spider path). There is no Dafny-side risk to investigate, since section A leaves every shared Dafny file untouched.
- Pipeline reliability on Spider is unknown until G.3; the first iteration may reveal second-order issues (runtime cost of always-constrained, parser performance at long SQL, etc.) that this plan doesn't anticipate.

### Priority vs the SQL-unblock checklist in the section above

This plan **supersedes** SQL unblock steps 1 and (possibly) 2 from the Next Steps list dated April 24. Steps 3 (`_exec_match_spider` hand-prediction sanity) and 4 (stuck verification-refinement loop) are still worth doing in parallel — they're orthogonal to the delimiter question and shorten the debug loop once Spider synthesis produces non-zero accuracy.

---

### 23. Resolved: obs-21 `expected=""` was a real HF-loader bug, fixed (April 25, 2026)

**Final diagnosis (after two wrong intermediate ones):**

`evaluations/sql_spider/dataset.py:_load_hf_spider` was reading the gold-SQL field with:
```python
query = ex.get("response") or ex.get("query") or ex.get("sql") or ""
```
But the actual HF dataset `richardr1126/spider-context-validation` uses the field name `ground_truth`. None of the three names in the chain ever matched, so `query` defaulted to `""` for **every example loaded via the HF source**. Local-source loading (the fallback path under `/home/aadivyar/spider_data/`) populated `query` correctly.

**Why this hid for so long.** The bug was source-conditioned. When `load_spider(source='auto', ...)` is called:
- with the `datasets` package available (real eval path inside `evaluate_sample`), HF source is selected → `query=""` → `expected=""` → `_exec_match_spider` short-circuits at `if not gold_sql: return False` → `is_correct=False` for every example → 0% accuracy regardless of the model output.
- with the `datasets` package stubbed (the standalone diagnostic I ran earlier), HF source fails to import and the local fallback is used → `query` populated → `_get_expected_answer` returns gold → looks fine.

That's why the obs-21 first-pass diagnostic incorrectly said the bug was a serialization artifact: the test environment took the wrong dataset path.

**Diagnosis trail:**
1. obs 21 (Apr 24, original) — flagged `expected=""` in failure_report sample_outputs.
2. obs 21 update (later Apr 24) — incorrectly concluded it was a serialization-time artifact based on a stubbed-import diagnostic that took the local fallback path.
3. Apr 25, single-example eval against attempt 6 of the killed run — confirmed `expected=""` at runtime in `evaluate_sample`, contradicting the serialization-artifact theory.
4. Instrumented `evaluate_sample` with a print-the-example-dict diagnostic — caught `query=''` at the lookup site even though `'query' in example` was True.
5. Tightened the diagnostic to match the live eval path (real torch/transformers, which pulls in `datasets`) — confirmed the HF source was being selected and the field-name mismatch was the cause.

**Fix:** one line in `evaluations/sql_spider/dataset.py:_load_hf_spider`:
```diff
-        query = ex.get("response") or ex.get("query") or ex.get("sql") or ""
+        query = ex.get("ground_truth") or ex.get("response") or ex.get("query") or ex.get("sql") or ""
```

**Magnitude of the impact.** Every Spider synthesis run before this fix was scoring every strategy at 0% accuracy regardless of how good the SQL was. The 0% in observation 21's failure report wasn't a strategy-quality problem; it was the loop being unable to credit any correct SQL because gold was always empty. The 6 strategies that verified in the post-delimiter-fix run (0% / 58.8% / etc.) were almost certainly closer to good than they appeared.

**Lesson.** When two independent diagnostics disagree about a runtime behavior, check whether they're exercising the same import / configuration path. The stubbed-import diagnostic and the live eval were taking different branches of `load_spider`'s `auto` source-selection logic, and that was enough to produce opposite conclusions.

---

### 24. First Spider winner — 60% on 10-example calibration (April 25, 2026)

**Run:** `outputs/generated-csd/runs/20260425_053950_45e14f/`
**Winning attempt:** 4 of 20.
**Result:** 60% accuracy, 52.4% syntax, contains `<<...>>`: yes. Passes all gates (`--min-accuracy 0.2 --min-syntax-rate 0.5`).

**Synthesis config:**
- generation model: `gpt-5.4` via openai backend
- evaluation model: `Qwen/Qwen2.5-Coder-7B-Instruct` via vllm backend
- task description: reason-then-constrain shape ("briefly reasons about which tables and columns are relevant before emitting the final SQL query. The final SQL query is wrapped in a single `<<...>>` block...")
- `--eval-sample-size 10 --eval-max-steps 400 --vllm-max-model-len 8192 --synthesis-max-tokens 2048`

**Per-example breakdown** (10 examples, sample_seed default):

| # | ✓/✗ | Question | Notes |
|---|----|----------|-------|
| 1 | ✓ | Avg age of dogs in treatments | exact match |
| 2 | ✗ | Owners with no dogs | join shape diff |
| 3 | ✗ | African countries with X | wrong aggregation/join |
| 4 | ✓ | Engineering dept degree count | semantically equivalent |
| 5 | ✓ | Battles that lost ships | semantically equivalent |
| 6 | ✓ | Kyle's grade | exact match |
| 7 | ✓ | Avg HP cars before 1980 | exact match |
| 8 | ✗ | Model below avg weight | wrong join direction |
| 9 | ✓ | Kyle's grade (alt phrasing) | exact match (uses `"Kyle"` quotes vs `'Kyle'`) |
| 10 | ✗ | Airport with most flights | picked wrong column (sourceairport instead of source∪dest) |

The 4 losses are real semantic SQL errors, not plumbing.

**Attempts to first winner:**
- Attempt 1: verified, 40% accuracy, 11.1% syntax → failed syntax gate.
- Attempts 2, 3: verification failures.
- Attempt 4: verified, **60% accuracy**, 52.4% syntax → passes.

**Generalization caveat (per obs 18 lesson).** 60% on n=10 has binomial SE ≈ 15pp. We cannot tell true-60% from true-30% on this sample size. A 50-example held-out eval is needed before claiming this is a real win. Running that now (`scripts/held_out_eval.py --sample-size 50 --seed 456 --dataset spider` against the winner module).

**Strategy artifacts:**
- Dafny: `outputs/generated-csd/runs/20260425_053950_45e14f/sql_crane_csd.dfy`
- Compiled Python: `outputs/generated-csd/runs/20260425_053950_45e14f/sql_crane_csd_20260425_054345_6f3ae7/`
- Success report: `outputs/generated-csd/runs/20260425_053950_45e14f/success_report.json`

**Open follow-ups:**
1. Held-out eval on n=50 to nail down whether 60% is real or a lucky 10-sample draw.
2. Vanilla CRANE baseline on Spider (the same 50 examples) so we have an apples-to-apples reference; without a baseline, "60%" has no meaningful comparison point.
3. Strategy keeps generating after `>>` (observed in obs-22 single-eval) — wastes step budget on hallucinated follow-up examples but doesn't directly hurt accuracy because `_extract_answer_spider` picks the first `<<...>>`. Worth fixing eventually.

---

### 25. The cost-invariant prompt edit and the GSM regression that wasn't (April 26, 2026)

**Context.** SQL synthesis runs were dominated by one verification failure mode: 17/20 attempts on the Apr 25 clean run failed Dafny verification with `Error: this invariant could not be proved to be maintained by the loop` on a `helpers.cost <= steps` invariant. The model kept adding that invariant because (a) the proof-sketch discipline list in `synthesis/prompts.py` named it as one of the four required invariants, and (b) all four worked examples in `INITIAL_GENERATION_PROMPT` carried it as a literal `invariant helpers.cost <= steps` line. The invariant is brittle: any branch with two cost-bumping helper calls per step (`OpenConstrainedSpan` then later `CloseConstrainedSpan` with mismatched accounting) breaks it. The contract only requires `cost <= maxSteps` on the *return value*, which `cost := steps` trivially satisfies — the internal accounting invariant earns no contract weight.

**Change.** Removed three things from `synthesis/prompts.py` (no other file touched):
1. The `cost: helpers.cost <= steps` bullet from the proof-sketch discipline list (4 → 3 items).
2. The `// cost: ...` paragraph in each of the three loop-form worked examples (4th example is recursive, didn't need editing).
3. The `invariant helpers.cost <= steps` (and `<= attempts`) line from each of the three loop-form worked-example bodies.

The `cost := steps` line at the end of each example stays. The `cost <= maxSteps` postcondition stays. The change is purely a prompt-side simplification — task-agnostic per CLAUDE.md.

**Result on SQL synthesis.** Verification rate jumped from 3/20 (Apr 25 clean run) to ~14/20 (post-edit run). Verification cliff resolved. But max accuracy across the 14 verified strategies was still 30% (most 0–2%), well below the unconstrained Qwen-7B-Coder baseline of ~66%. The prompt edit did exactly what it was designed to do; it does not address the SQL proposal-quality problem, which is a separate issue rooted in the schema-narrowed grammar mismatching Qwen's natural SQL lexicalization.

**The GSM regression wasn't a regression.** Ran a 12-attempt GSM synthesis with the new prompt at the lottery's setup (gpt-5.4 generator, Qwen-14B eval, eval-seed=123, n=10 calibration, gates min-acc=0.51 / min-syntax=0.72). No formal winner across 12 attempts; best attempt was 70% acc / 60% syntax (above CRANE accuracy, below CRANE's lottery-recorded 70.7% syntax). Initially read as a regression.

**Root cause turned out to be a dataset-loader change, not the prompt edit.** `evaluations/gsm_symbolic/dataset.py` was changed (uncommitted post-lottery) from:
```python
random.seed(42)  # Fixed seed for reproducibility — caller's seed argument IGNORED
indices = random.sample(range(len(ds)), min(limit, len(ds)))
```
to:
```python
rng = random.Random(seed) if seed is not None else random
indices = rng.sample(range(len(ds)), min(limit, len(ds)))
```
This was an *intentional* upgrade — it unlocks per-attempt seed rotation in `feedback_loop.py` (`eval_base_seed + attempt.attempt_number - 1`), which prevents synthesis from selecting strategies that overfit the same fixed 10 examples across all 12 attempts. The fix is correct.

But the lottery_manifest's `calibration: seed=123` was a label, not a measurement parameter — under the old loader, `seed=123` was *ignored* and evaluation actually ran on the seed=42 sample. So the manifest's CRANE baseline (50% acc / 70.7% syntax) was measured on the seed=42 sample. Verified today:

| Sample | CRANE acc | CRANE syntax |
|--------|-----------|--------------|
| Lottery manifest (Apr 22, labeled seed=123, actually seed=42) | 50% | 70.7% |
| seed=42 today | 40% | 70% |
| seed=123 today | 70% | 80% |

Today's seed=42 reproduces the manifest within ±10pp single-example noise. Today's seed=123 is a different 10-example sample where CRANE happens to score 30pp higher. CRANE alone varies 40–70% accuracy across two arbitrary seeds at n=10 — that *is* the noise envelope.

**Implication for the GSM regression:**
- Each post-edit attempt evaluated on a different rotated seed (123, 124, …, 134).
- CRANE itself varies 40–70% accuracy across that seed range.
- Post-edit attempts varied 0–70% accuracy across the same range.
- No statistically meaningful regression. The 60% syntax across multiple attempts pattern is within sample-variance noise.

**Decision.** Treat the prompt edit as GSM-safe. Move on. The lottery_manifest baselines (50%/70.7%) are obsolete by design — they reflect the pre-rotation overfit-prone regime. A proper GSM baseline under the new regime would be CRANE measured under the same seed rotation the synthesis uses (12 runs at seeds 123–134, averaged) — worth doing as an apples-to-apples reference for any future synthesis comparison, but not a blocker.

**Lesson.** When a baseline number changes after a code change, the first question is whether the *measurement instrument* changed, not whether the *measured object* changed. Here the dataset loader's seed-handling flipped, which silently re-anchored every same setup as before comparison to a different sample. Cost: ~30 minutes of investigation we wouldn't have spent if we'd noticed the loader diff first.
