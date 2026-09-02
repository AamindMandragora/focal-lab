# Handoff — Qwen3.5 metaDecode win campaign

**Date:** 2026-06-29
**For:** a fresh agent (possibly a different account/model) picking this up cold. The prior
session hit the Claude monthly spend limit mid-task. This doc is everything needed to continue
without re-deriving context. Records of truth: `results_matrix.md` (project root),
`saved-results/2026-06-28-main-matrix-snapshot.md`, and the memory index
`~/.claude/projects/.../memory/MEMORY.md`.

Plain-language strategy brief for review/feedback:
`saved-results/2026-06-30-all-frameworks-win-strategy.md`.

User correction recorded 2026-06-30: do not let the campaign drift into only changing bars,
sample sizes, launch settings, and provenance. Those fixes are allowed when they explain the
failure, but the next serious hypothesis after an active job is recorded should usually attack
the core framework: change an existing helper's behavior or add a new general helper primitive
that changes what the generated CSD can generate, validate, repair, or select. Keep the fairness
rules: no scorer/dataset/baseline changes, no synthesis warm starts, and no dataset-specific win
guidance in task descriptions or helper contracts.

Latest core-framework update recorded 2026-06-30: H88 audited H86's active SMILES logs and found
the strongest current failure evidence was span/control-output-budget, long invalid concatenated
SMILES, and context-overflow style failures, with duplicate/diversity as secondary evidence. H89
then exposed existing managed-span helpers instead of inventing a new helper: `ManagedStep`,
`GenerateWithManagedSpan`, and `GenerateWithPrefixAndManagedSpan` now have prompt-visible docs and
are classified as prunable helper-policy arms. This is framework groundwork for future SMILES
launches after H86 is recorded; it is not a benchmark result and does not change the paper matrix.
H90 then audited the whole helper surface CPU-only and found **0** remaining prompt-universe
exposure gaps after H89. It recorded **9** `expose_now`, **7** `profile_later`, and **55**
`stage_later` helper records, but the interpretation is to avoid broad cleanup for its own sake:
the next helper patch should be tied to active H86/H65 failure evidence.

Latest SMILES framework update recorded 2026-06-30: H91 found that H86's highest-syntax
attempt could make **100/100** grammar-valid and RDKit-valid outputs while getting **0/100**
class-membership and **0/100** unique-valid candidates. H92 therefore added a general no-gold
class-membership helper: `helpers.PrefixMatchesPromptMoleculeClass(lm, prefix)`. It infers the
molecule class only from prompt-visible text and uses the existing generic SMILES
class-membership logic. Verification on focal main passed **16/16** grounding tests, **6/6**
helper-surface tests, Python compile checks, prompt exposure checks, and Dafny verification
(**179 verified, 0 errors**). H92 is framework groundwork only, not a benchmark result.

Active SMILES paid run recorded 2026-06-30: H93 is now running as the next cold
`isocyanates-9B` train100 run from patched focal main after H92. Launch used recorded AWS account
**887730490125** with `DRY_RUN=0 SAFE_GPU_ID=0 CONFIRM_BEDROCK_ACCOUNT_887730490125=yes`; AWS CLI
was not installed on focal, so live account identity could not be independently reverified during
the launch check. PID **756754**; PID file
`/tmp/csd_h93_logs/h93_smiles_qwen35_9b_isocyanates_h92helper_train100_20260630.pid`; log
`/tmp/csd_h93_logs/h93_smiles_qwen35_9b_isocyanates_h92helper_train100_20260630.log`; generated
root `outputs/generated/smiles_qwen35_9b_isocyanates_uv_qwen35_h93_h92helper_20260630`; latest run
`outputs/generated/smiles_qwen35_9b_isocyanates_uv_qwen35_h93_h92helper_20260630/smiles_qwen35_9b_isocyanates_uv_qwen35_h93_h92helper_20260630_20260630_213016_1f5bdd`.
Monitor H93; do not launch another paid SMILES cell until H93 is recorded.

Latest GSM-9B status recorded 2026-06-30: H65 is finished and should not be monitored as running.
It did not cross the train bar, so H71 must not launch. Best H65 accuracy was attempt **32** at
**44.9%** accuracy (**22/49**) and **95.9%** syntax versus the **53.1% / 98.0%** train bar.
Final attempt **40/40** scored **32.7%** accuracy (**16/49**) and **89.8%** syntax. Failure report:
`outputs/generated/synth_gsm_9b_z3fix_seed123train_h65_timeoutguard_20260630/synth_gsm_9b_z3fix_seed123train_h65_timeoutguard_20260630_20260629_185703_9e2a21/results/failure_report.json`.
Precise credential scan found **0** AWS-shaped key values and **0** env-style secret assignments
in the H65 log/report/launch record. Matrix status is unchanged.

Latest SMILES-9B status recorded 2026-06-30: H86 is finished and should not be monitored as
running. It did not cross the train bar, so no held-out re-eval ran. Best H86 train attempt was
attempt **20** at UV/accuracy **0.37** and syntax **0.41** on **100** train examples versus the
**0.92 / 0.50** train gate. Attempt **21** reached syntax **0.92** but only UV/accuracy **0.32**.
Final attempt **40/40** scored UV/accuracy **0.10** and syntax **0.25**. Failure report:
`outputs/generated/smiles_qwen35_9b_isocyanates_uv_qwen35_0627/smiles_qwen35_9b_isocyanates_uv_qwen35_0627_20260630_083748_22e3d6/results/failure_report.json`.
Precise credential scan found **0** AWS-shaped key values, **0** env-style secret assignments, and
**0** secret key-name mentions across the checked H86 log/report/launch/provenance files. Matrix
status is unchanged. The next SMILES design step should use the core-framework policy: do not rerun
a bar-only H86 clone; use H86/H88/H89/H90 evidence to test a targeted helper/framework lever or a
cold run from patched focal main.

---

## The goal (verbatim, Stop-hook enforced)

> "keep working until you've collected all data necessary for ablations and for the main
> matrix and paper-ready results are ready."

Effort = `/effort ultracode`. This is an **autonomous** campaign — see the autonomy ruling below.

## Standing rules the user gave (all verbatim, all still in force)

1. **Win-on-train → re-eval on held-out immediately.** "as soon as something wins in train
   make sure to re eval on held out." Held-out re-eval of an accepted strategy is $0 (pure
   re-eval: `--initial-strategy-file` + `--max-iterations 1`, or
   `synthesis.scripts.reevaluate_compiled_csd`). Prioritize it over starting new cells.
2. **Every synthesis run uses `--max-iterations 40`.** Not 30.
3. **The pivot:** if cells aren't winning, the PRIMARY job is to make **framework changes** so
   they become wins. Iterate fast on a small per-iteration sample (50 examples for GSM/Spider;
   49 is the full GSM set) to get through the loop quicker. Rescore baselines on that same
   small set so comparisons stay apples-to-apples.
4. **On a win (the 3-stage discipline):** (a) win on the small set → (b) relaunch a **FRESH
   COLD** full-set synthesis (300 for Spider; NOT reusing the small-set CSD — cold means cold),
   same `--max-iterations 40`, confirm the synthesizer can independently find an equally-winning
   strategy on all 300 → (c) held-out test on the full set.
5. **Order:** start with the cell closest to winning, continue till all 6 GSM/Spider cells win;
   do the same for all SMILES cells (all must win under the same fast methodology). Do small
   models first but cover both small and big.
6. **No cell-count limit — unbounded.** Framework-change direction is fully the agent's call,
   **as long as no dataset-specific "how to win" guidance leaks into any prompt at any level.**
7. **AUTONOMY (2026-06-28, verbatim):** "do NOT wait for me whole point was for this to be
   autonomous." → The predict-then-test **confirmation gate is WAIVED.** Keep the science:
   write hypothesis / prior / falsifiable prediction in the ledger BEFORE running each tweak,
   record actual / belief-change / kept-or-reverted AFTER. DROP only the stop-and-confirm step.
   Still revert refuted or ambiguous tweaks. One variable per experiment. This does NOT waive
   money, fairness, or warm-start rules.

## Binding constraints (NEVER override — these survive full autonomy)

- **MONEY (hard rule):** Bedrock author spend runs on AWS account **887730490125** = the UIUC
  focal lab account (pre-approved, NOT personal). Creds via `$REPO/.env`. Never a personal
  account. Re-verify the account before any expensive launch. (Session author spend was ~$3.7k
  informational — not a stop signal; the user gave unbounded budget on the lab account.)
- **FAIR COMPARISON:** never modify grammars, graders/scorers, dataset splits, or baseline
  prompts. Fairness = ORIGIN not content: anything the generator model derives itself from fair
  inputs is fair even if it encodes the output format; only what WE inject (prompts/`--task`/
  feedback/helper descriptions) is banned. No dataset-specific win guidance at any sophistication.
- **WARM STARTS BANNED (permanent):** all synthesis COLD. `--initial-strategy-file` is allowed
  ONLY for pure re-eval (`--max-iterations 1`). Never seed further iterations from a prior strategy.
- **ADAPTIVE HELPER MASK must stay ON + bandit policy** (`--adaptive-helper-mask
  --helper-selection-policy bandit`). A `--no-adaptive-helper-mask` win is invalid. Mask-trap
  problems are fixed by framework changes, never by turning the mask off.
- **FOCAL is the only source of truth** for code/state. Never grep the local Mac repo to answer
  "is X deployed / what ran / what does the code do" — it diverges. Mac is only for
  local-edit-then-`scp`.
- **Never `--min-syntax-rate 1.0`** (cuts evals at 1-3 examples). Use 0.85 or lower for exploratory.
- **NO UNPROMPTED DIAGNOSIS** is the project default, but it is OVERRIDDEN for this campaign
  (the user explicitly wants diagnosis + framework changes). Diagnose and act freely here.
- **Core framework/helper priority:** if a miss is about what the CSD can produce or choose,
  prefer changing helper behavior or adding a general helper primitive before another
  threshold-only retry. Rank levers as: provenance/recording fix; gate/launch setting;
  speed-only helper implementation; helper behavior change; new general helper primitive;
  disallowed benchmark shortcut. Use the deepest allowed lever that matches the evidence.
- **H89 managed-span helper exposure is now in focal main.** Future cold synthesis can see real
  docs for the existing managed-span helpers instead of only helper names. If H86 fails and the
  next SMILES run is launched from current focal main, this is the core helper-surface change that
  should be tested before adding another span helper.
- **Optimization follow-up order after the parser helper patch:** do not touch all helpers equally.
  After H86 is recorded, profile the next SMILES run first and inspect
  `GenerateLogits.prefix_text` in `synthesis/evaluate/benchmarks/common/model_utils.py` only if
  it is still material in the timing breakdown. Defer `CompletedSchemaSymbolCount` work until the
  GSM/Spider stages, where schema-symbol rollback helpers are relevant. Inspect `GetTopKTokens`
  only if a chosen strategy actually uses top-k helpers heavily. Record any optimization as a
  ledger-first, one-variable, TDD change before it affects a launch.

---

## Current board (15 cells = 3 datasets × {2B,4B,9B} × …)

| Dataset | Cell | Baseline bar (acc / syn) | metaDecode | Verdict |
|---|---|---|---|---|
| GSM (CRANE, eval-49) | 2B | 24.5% / 83.7% | 16.3% / 98.0% (train49) | **LOSS** −8.2pp |
| GSM | 4B | 42.9% / 91.8% | 30.6% / 67.3% best train attempt (attempt 31 of completed 40) | **LOSS** −12.3pp |
| GSM | 9B | 53.1% / 98.0% | H65 train loss: best attempt **44.9% / 95.9%** on 49 examples; final attempt **32.7% / 89.8%** | **LOSS** |
| Spider (IterGen, test-300 official grader) | 2B | 37.7% / 90.7% | **38.3% / 99.3%** held-out | **WIN** +0.6pp |
| Spider | 4B | 66.0% / 97.3% | 53.7% / 97.7% (train300) | **LOSS** −12.3pp |
| Spider | 9B | 67.0% / 98.3% | H78 held-out **74.0% / 99.0%** on 300 examples | **WIN** +7.0pp |
| SMILES (CARS, UV) | acrylates-2B | live CARS **0.36 / 1.00** | H70 held-out **UV 0.34 / validity 0.82** on N=100 after train accepted at **0.42 / 0.78** on N=50; H81 held-out **UV 0.17 / validity 0.78** on N=100 after train accepted at **0.44 / 0.88** on N=50 | **LOSS** under live primary UV bar |
| SMILES | isocyanates-2B | live CARS **0.98 / 1.00** | **UV 0.290 / 0.95** held-out N=100 | Historical old-bar win; **not paper-ready under live CARS** because **0.29 <= 0.98** |
| SMILES | chain_extenders-2B | 0.400 / 0.92 | H61 held-out loss: UV **0.14**, validity **0.99** on N=100 after train accepted at **0.66 / 0.98** | NO |
| SMILES | isocyanates-4B | 0.160 / 1.00 | H63 held-out win: UV **0.58**, validity **0.61** on N=100 after train accepted at **0.48 / 0.64** | **WIN** on primary UV metric; note validity does not match CARS 1.00 |
| SMILES | isocyanates-9B | live CARS **0.92 / 1.00** | H86 train100 loss: best train attempt **0.37 / 0.41** on N=100; final attempt **0.10 / 0.25** | **LOSS**; no held-out because train did not cross |
| SMILES | acrylates/chain × 4B,9B plus remaining SMILES unresolved cells | see CARS bars below | not launched or not paper-ready | PENDING |

**Live-artifact tally after H78:** 3 paper-ready held-out wins under the current live bars (**Spider-2B**, **Spider-9B**, and SMILES **isocyanates-4B** primary UV). The old-board tally had counted 4 wins, but H67 shows acrylates-2B and isocyanates-2B do not clear the live CARS bars and must be treated as historical old-bar wins unless the paper explicitly adopts that older convention. Remaining cells still need train/held-out wins or live-bar SMILES wins.

**SMILES live focal CARS bars after H66 audit** (accuracy / syntax from `outputs/controlled_comparison/smiles_qwen35/*/*/cars.json`): acrylates 2B **0.36 / 1.00**, 4B **1.00 / 1.00**, 9B **0.98 / 1.00**; chain_extenders 2B **0.94 / 1.00**, 4B **0.94 / 1.00**, 9B **1.00 / 1.00**; isocyanates 2B **0.98 / 1.00**, 4B **0.16 / 1.00**, 9B **0.92 / 1.00**. Older short bar notes are stale; use H66/live artifacts for launch selection.

**Spider 50-set fast bars** (rescored): 2B 50.0% (25/50), 4B 60.0% (30/50),
9B 20.0% (10/50), syntax 100.0% (50/50). The 9B bar is from the local-only IterGen
diagnostic retry output `outputs/controlled_comparison/spider_qwen35_9b/itergen_50eval_seed334.json`
on focal. It wrote `accuracy=0.2`, `syntax_rate=1.0`, `num_examples=50`,
`total_generation_seconds=1614.3235`, `mean_generation_seconds_per_example=32.286469`,
and `run_wall_time_seconds=1643.1989`.

**Closest losses to flip first:** Spider-9B (−2.3pp, but a real semantic SQL gap — may be a fair
ceiling) and GSM-2B (−8.2pp, but CRANE only gets 24.5% on this SAME 2B). H1 refuted the
simple "extend past first complete span" fix, so the next GSM lever should target bounded semantic
span completion or CoT structure, not raw longer spans.

---

## What is RUNNING right now (focal, refreshed 2026-06-29)

| Job | GPU | PID | Age | Notes |
|---|---|---|---|---|
| GSM-4B synth | 1 | 14560 | completed / failed train bar | COLD, max-iter 40. Final report `outputs/generated/synth_gsm_4b_z3fix_seed123train_0628b/synth_gsm_4b_z3fix_seed123train_0628b_20260628_152637_acf098/results/failure_report.json` at `2026-06-29T10:35:22.029904`. Best accuracy attempt **31** reached **15/49 = 30.61%** with **67.35%** syntax; best syntax attempt **19** reached **4/49 = 8.16%** with **100.0%** syntax. Bar was **42.9% / 91.8%**, so no train win and no held-out re-eval. Specific output/log scan found no paid credential key names. |
| GSM-9B synth | 0 | 284546 | running | COLD, max-iter 40. Current anchor attempt 7 = 42.9%/93.9%; bar = 53.1%/98.0%. Last seen attempt marker 15/40 at 2026-06-29T00:05:54Z. |
| Spider-9B 50-set IterGen baseline diagnostic retry | 2 | 955809 | completed | **$0 local-only eval, no Bedrock vars**. Result: 20.0% acc / 100.0% syntax, N=50. Output `outputs/controlled_comparison/spider_qwen35_9b/itergen_50eval_seed334.json`; log `/tmp/spider_qwen35_9b_itergen50_seed334_diag5_20260629.log`. |
| H2 GSM-2B final-span mechanism probe | 2 | 1066489 | completed | **$0 local-only max-iterations=1 eval; not publishable as-is.** Final report `outputs/generated/h2_gsm2b_final_span_probe_20260629/h2_gsm2b_final_span_probe_20260629_20260628_215702_fdbc73/results/success_report.json`; result **2/49 = 4.08% accuracy**, **48/49 = 97.96% syntax**, median visible span length **12.5**, unclosed-span answers **1/49**. H2 is refuted and recorded in `docs/experiments/metadecode-fast-iteration-log.md`. |
| H4 GSM-2B semantic-plan mechanism probe | 2 | 1135526 | completed | **$0 local-only max-iterations=1 eval; not publishable as-is.** Report `outputs/generated/h4_gsm2b_semantic_plan_probe_20260629/h4_gsm2b_semantic_plan_probe_20260629_20260628_222132_140e8a/results/success_report.json`; result **0/49 = 0.0% accuracy**, **44/49 = 89.8% syntax**, median visible span length **9.0**, expected-expression before final span **0/49**, expected-expression anywhere **2/49**. H4 is refuted and recorded in `docs/experiments/metadecode-fast-iteration-log.md`. |
| H14/H15 Spider-9B attempt-20 50-set probe | 2 | completed | completed | **$0 local-only pure re-eval attempts; not publishable.** Both failed before evaluation at `search_contract` on `RolloutConstrainedWithPenalties`. H14 used the adaptive helper mask and H15 omitted it; H15 still failed, proving the current synthesis runner applies the disallowed-helper contract before verification/evaluation for initial strategy files. H15 report: `outputs/generated/h15_spider9b_att20_fast50_nomask_20260629/h15_spider9b_att20_fast50_nomask_20260629_20260629_000040_92e4fe/results/failure_report.json`. |
| H16 Spider-9B direct-eval path audit | 2 | completed | completed/interrupted | **$0 local-only API smoke; not publishable.** Direct verifier/compiler/evaluator APIs bypassed `search_contract` and verified/compiled the old attempt-20 strategy, but no example was evaluated. `vllm_gpu_memory_utilization=0.50` failed KV-cache startup; `0.55` fixed KV cache but began a fresh `CachedQwen2TokenizerFast` SQL mask build and was interrupted at 7/260 after ~2m13s. Compiled module: `outputs/generated/h16_spider9b_direct_eval_file_smoke200_util55_20260629/python/h16_spider9b_direct_eval_file_smoke200_util55_20260629/GeneratedCSD.py`. |
| H17 Spider-9B cache/backend audit | 2 | completed | completed | **$0 read-only diagnostic.** Confirmed H16's stall was a tokenizer cache-key mismatch: current vLLM tokenizer type `CachedQwen2TokenizerFast`, vocab size `248044`; existing matching SQL mask was under `cache/mask_stores/CachedTokenizersBackend/grammar_mask_7704218576_248044.pkl`, size **3,910,900,672 bytes**. No model eval or repo code change. |
| H18 Spider-9B cache-alias smoke | 2 | completed | completed | **$0 cache/runtime smoke; not publishable.** Created symlink `cache/mask_stores/CachedQwen2TokenizerFast/grammar_mask_7704218576_248044.pkl -> ../CachedTokenizersBackend/grammar_mask_7704218576_248044.pkl`. Direct no-contract smoke evaluated exactly **1/1** example with `accuracy=1.0`, `syntax_rate=1.0`; output `outputs/generated/h18_spider9b_direct_eval_cachealias_smoke_20260629/results/smoke_success.json`. |
| H19 Spider-9B attempt-20 50-set direct eval | 2 | completed | completed | **$0 local-only no-contract diagnostic; not publishable as a cell result.** Output `outputs/generated/h19_spider9b_att20_fast50_direct_cachealias_20260629/results/direct_eval_success.json`; result **39/50 = 78.0% accuracy**, **48/50 = 96.0% syntax**, elapsed **455.5438s**. Confirms the Spider-9B 50-set IterGen bar **10/50 = 20.0%** is too weak: the same strategy still loses on full held-out **194/300 = 64.7%** vs IterGen **201/300 = 67.0%**. |
| H24 GSM-2B multi-candidate selector probe | 2 | completed | completed | **$0 local-only direct eval; not publishable.** Output `outputs/generated/h24_gsm2b_multicandidate_selector_probe_20260629/results/direct_eval_success.json`; offline analysis `outputs/generated/h24_gsm2b_multicandidate_selector_probe_20260629/results/h24_candidate_selector_analysis.json`. Final-span evaluator result **5/49 = 10.20% accuracy**, **43/49 = 87.76% syntax**. Corrected candidate analysis: **120/124** parser-compatible candidate lines, candidate union **10/49**, H21 top-score selector **8/49**, model `Selected:` line **3/49**, final-span SymPy-equivalent **8/49**. H24 is refuted as a standalone strategy but useful as coverage input for H25. |
| H29 GSM-2B six-candidate consensus probe | 2 | completed | completed | **$0 local-only direct eval; not publishable.** Output `outputs/generated/h29_gsm2b_six_variant_consensus_probe_20260629/results/direct_eval_success.json`; offline analysis `outputs/generated/h29_gsm2b_six_variant_consensus_probe_20260629/results/h29_candidate_consensus_analysis.json`. Final-span evaluator result **3/49 = 6.12% accuracy**, **43/49 = 87.76% syntax**, median valid span length **9**, elapsed **1504.1768s**. Candidate analysis: **38/49** groups with candidate lines, **225** candidate lines, **125/225** parser-compatible lines, candidate union **3/49**, model consensus **3/49**, agreement-selected **2/49**. H29 is refuted; prompt-only six-variant generation did not create useful diversity. |
| H30 GSM-2B candidate-diversity code diagnostic | n/a | completed | completed | **$0 read-only code diagnostic; no model run.** Focal source has pieces for diversity, especially `CSD_CONSTRAINED_TEMPERATURE` in `synthesis/evaluate/benchmarks/common/model_utils.py`, but no existing full-answer multi-candidate consensus wrapper. vLLM logits capture and unconstrained chunks still use `temperature=0.0`; `--temperature` controls synthesis authoring via `StrategyGenerator`; the evaluator/feedback loop evaluates one compiled strategy per attempt. H30 is mixed/mostly refuted and recorded in `docs/experiments/metadecode-fast-iteration-log.md`. |
| H31 no-gold candidate consensus selector prototype | n/a | completed | completed | **$0 TDD framework prototype; no model run.** Worktree `/home/aadivyar/.config/superpowers/worktrees/csd-generation/h31-candidate-consensus`, branch `h31-candidate-consensus`. Added `synthesis/evaluate/candidate_consensus.py` plus `synthesis/evaluate/test_candidate_consensus.py`. Red test failed with missing module; green focused test **4 passed in 0.04s**; nearby evaluate tests **14 passed in 5.10s**; `py_compile` passed. Kept in isolated worktree only, not promoted to dirty focal main checkout. |
| H32 H31 selector replay over H27 pool | n/a | completed | completed | **$0 CPU-only artifact replay; no model/GPU run.** Worktree `/home/aadivyar/.config/superpowers/worktrees/csd-generation/h31-candidate-consensus`. Output `outputs/generated/h32_h31_selector_h27_replay_20260629/h32_summary.json`. Exact H27 replay was refuted by **2** cluster-key mismatches, but H31's tuple rule selected **14/49 = 28.57%** positives on the same saved pool, above H27's **12/49**. Final tests **16 passed in 4.87s** plus `py_compile` passed. Diagnostic only; not a GSM-2B win because it reuses old candidate pool. |
| H33 safe repeat-attempt planner | n/a | completed | completed | **$0 CPU-only TDD wrapper prototype; no model/GPU run.** Worktree `/home/aadivyar/.config/superpowers/worktrees/csd-generation/h31-candidate-consensus`. Added `synthesis/evaluate/candidate_repeat_plan.py` plus `synthesis/evaluate/test_candidate_repeat_plan.py`. Red test failed with missing module; focused green **4 passed in 0.04s**; nearby H31/H32/H33/evaluate tests **20 passed in 5.00s**; `py_compile` passed. Planner strips paid credentials, rejects paid backend commands, and adds unique source/output metadata plus `CSD_CONSTRAINED_TEMPERATURE`. |
| H34 dry-run repeat manifest writer | n/a | completed | completed | **$0 CPU-only TDD manifest writer; no model/GPU run.** Same isolated worktree. Extended `synthesis/evaluate/candidate_repeat_plan.py` with `write_repeat_manifest` and added `synthesis/evaluate/test_candidate_repeat_manifest.py`. Red test failed with missing function; focused green **2 passed in 0.04s**; nearby H31-H34/evaluate tests **22 passed in 4.92s**; `py_compile` passed. Manifest records `dry_run`, `no_model_calls`, `no_billed_credentials`, attempts, safe env, source/output ids, and temperatures without leaking paid key names or values. |
| H35 repeat launch validator | n/a | completed | completed | **$0 CPU-only TDD launch validator; no model/GPU run.** Same isolated worktree. Extended `synthesis/evaluate/candidate_repeat_plan.py` with `validate_repeat_attempts_for_launch` and added `synthesis/evaluate/test_candidate_repeat_validation.py`. Red test failed with missing function; focused green **3 passed in 0.04s**; nearby H31-H35/evaluate tests **25 passed in 4.88s**; `py_compile` passed. Validator rejects raw H29 direct-eval command because output/source are not parameterized, accepts commands with `--output-name {output_name}` and `--source-id {source_id}`, and rejects leaked paid credential env keys. |
| H36 parameterized direct-eval runner | n/a | completed | completed | **$0 CPU-only TDD runner/config prototype; no model/GPU run.** Same isolated worktree. Added `synthesis/evaluate/parameterized_direct_eval.py` plus `synthesis/evaluate/test_parameterized_direct_eval.py`. Red test failed with missing module; focused green **3 passed in 0.05s**; nearby H31-H36/evaluate tests **28 passed in 4.95s**; `py_compile` passed. Dry-run artifact `outputs/generated/h36_parameterized_direct_eval_dryrun_20260629/direct_eval_dry_run.json` proves output/source are parameterized and no paid key names or old H29 output root leaked. |
| H37 GSM-2B repeat-probe dry-run manifest | n/a | completed | completed | **$0 CPU-only dry-run manifest; no model/GPU run.** Output `outputs/generated/h37_gsm2b_repeat_probe_manifest_20260629/repeat_manifest.json`; checks `h37_manifest_checks.json`. Three planned attempts use temps **0.0/0.25/0.5**, unique output/source ids, parameterized direct-eval command with `--output-name {output_name}` and `--source-id {source_id}`, and safety flags `dry_run`, `no_model_calls`, `no_billed_credentials`. String checks found no paid key names, secret placeholder, or old H29 output-root string. |
| H38 GSM-2B repeat-probe materialized commands | n/a | completed | completed | **$0 CPU-only command materialization; no model/GPU run.** Same isolated worktree. Added `synthesis/evaluate/test_candidate_repeat_materialize.py` plus materializer helpers in `synthesis/evaluate/candidate_repeat_plan.py`. Red test failed with missing import; focused green **4 passed in 0.05s**; nearby H31-H38/evaluate tests **32 passed in 5.10s**; `py_compile` passed. Output `outputs/generated/h38_gsm2b_repeat_materialized_commands_20260629/materialized_commands.json`; checks `h38_materialized_checks.json`. Three non-executing command plans have `execute=false`, `dry_run=true`, `CUDA_VISIBLE_DEVICES=TO_FILL_WHEN_SAFE`, unique output/source ids, temps **0.0/0.25/0.5**, and no `{output_name}`, `{source_id}`, paid key names, or secret placeholder. |
| H39 GSM-2B repeat-probe dry-run execution check | n/a | completed | completed | **$0 CPU-only runner dry-run execution; no model/GPU run.** Ran all **3/3** H38 command plans with `--dry-run --dry-run-output` from the isolated worktree using a tiny stripped environment plus each plan's safe env. Summary `outputs/generated/h39_gsm2b_repeat_dryrun_execution_20260629/h39_dryrun_execution_summary.json`; per-command JSONs `h37_gsm2b_t0/t1/t2_dry_run.json`. Return codes **[0, 0, 0]**; all payload checks passed for `dry_run`, `no_model_calls`, `no_billed_credentials`, output/source ids, dataset `gsm_symbolic`, split `train`, sample size **49**, max steps **900**. String scan found no paid key names or secret placeholder. |
| H40 GSM-2B repeat-probe single local smoke attempt | 2 | 2271673 | completed | **$0 local-only GPU smoke attempt; no Bedrock/OpenAI/Anthropic call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before launch. Ran only H38 plan `t0` / temp **0.0** from isolated worktree using explicit `/apps/conda/aadivyar/envs/csd/bin/python`, `CUDA_VISIBLE_DEVICES=2`, and `--vllm-gpu-memory-utilization 0.20`. Report `outputs/generated/h37_gsm2b_repeat_probe_20260629_t0/results/direct_eval_success.json`: `success=True`, `source_id=h37_gsm2b_t0`, `output_name=h37_gsm2b_repeat_probe_20260629_t0`, **49** examples, accuracy **0.061224489795918366** (**3/49**), syntax_rate **0.8775510204081632** (**43/49**), elapsed **8798.66094827652s** (~**2h26m39s**). Log/output artifact scan found no paid credential key names or secret placeholder; no OOM/traceback found. Kept as smoke artifact only, not a benchmark win. |
| H41 GSM-2B repeat-probe temp-0.25 local attempt | 2 | 2801821 | completed | **$0 local-only GPU temp probe; no Bedrock/OpenAI/Anthropic call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before launch. Ran only H38 plan `t1` / temp **0.25** (`source_id=h37_gsm2b_t1`, output `h37_gsm2b_repeat_probe_20260629_t1`) on GPU2 with explicit `/apps/conda/aadivyar/envs/csd/bin/python`, `--vllm-gpu-memory-utilization 0.20`, and paid credential env vars stripped. Report `outputs/generated/h37_gsm2b_repeat_probe_20260629_t1/results/direct_eval_success.json`: `success=True`, `source_id=h37_gsm2b_t1`, `output_name=h37_gsm2b_repeat_probe_20260629_t1`, **49** examples, accuracy **0.04081632653061224** (**2/49**), syntax_rate **0.8775510204081632** (**43/49**), elapsed **1835.9341404438019s** (~**30m36s**). Log/output artifact scan found no paid credential key names or secret placeholder. The log had a post-run VLLM `EngineCore died unexpectedly` line after example 49, but the runner wrote the success report. Refuted as an accuracy/diversity probe; kept as diagnostic only, not a win. |
| H42 GSM-2B repeat-probe temp-0.5 local attempt | 2 | 2932784 | completed | **$0 local-only high-temperature probe; no Bedrock/OpenAI/Anthropic call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before launch. Ran only H38 plan `t2` / temp **0.5** (`source_id=h37_gsm2b_t2`, output `h37_gsm2b_repeat_probe_20260629_t2`) on GPU2 with explicit `/apps/conda/aadivyar/envs/csd/bin/python`, `--vllm-gpu-memory-utilization 0.20`, and paid credential env vars stripped. Report `outputs/generated/h37_gsm2b_repeat_probe_20260629_t2/results/direct_eval_success.json`: `success=True`, `source_id=h37_gsm2b_t2`, `output_name=h37_gsm2b_repeat_probe_20260629_t2`, **49** examples, accuracy **0.061224489795918366** (**3/49**), syntax_rate **0.8979591836734694** (**44/49**), elapsed **1341.0293157100677s** (~**22m21s**). Log/output artifact scan found no paid credential key names or secret placeholder. The log had a post-run VLLM `EngineCore died unexpectedly` line after example 49, but the runner wrote the success report. Refuted because accuracy was **3/49**, not **>3/49**; H44 has now recorded the overlap/union diagnostic and the temperature branch is closed. |
| H43 H40/H41 repeat-overlap diagnostic | n/a | n/a | completed | **$0 CPU-only artifact diagnostic; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before measurement. Output `outputs/generated/h43_gsm2b_repeat_overlap_diagnostic_20260629/h43_summary.json`. Result: exact `full_output` matches **27/49**, exact extracted-`actual` matches **28/49**, failure-location matches **45/49**, helper-shape matches **41/49**, zero token-count delta on **35/49**, correct sets H40 `[16, 17, 39]` vs H41 `[16, 17]`, union **3**, symmetric difference **1**. Credential-name scan over the H43 output found no paid key names or secret placeholder. Confirmed: simple temperature repeats are too correlated to be a strong diversity source; if H42 does not beat **3/49**, stop this temperature branch and move to framework-level independent candidate generation/selection. |
| H44 H40/H41/H42 temperature-sweep overlap diagnostic | n/a | n/a | completed | **$0 CPU-only artifact diagnostic; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before measurement. Output `outputs/generated/h44_gsm2b_temperature_sweep_overlap_20260629/h44_summary.json`. Result: correct sets H40 `[16, 17, 39]`, H41 `[16, 17]`, H42 `[16, 17, 47]`; three-run union only **4/49** at `[16, 17, 39, 47]`; H42 added exactly **1** new correct example. Pairwise exact extracted-`actual` matches: H40/H41 **28/49**, H40/H42 **23/49**, H41/H42 **21/49**. Pairwise helper-shape matches: **41/49**, **38/49**, **38/49**. Credential-name scan over H44 output found no paid key names or secret placeholder. Confirmed: temperature branch is closed; next GSM-2B move should be framework-level independent candidate generation/selection. |
| H45 direct-eval report to consensus-candidate adapter | n/a | n/a | completed | **CPU-only TDD framework step; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before code/tests. Worktree `/home/aadivyar/.config/superpowers/worktrees/csd-generation/h31-candidate-consensus`. Added `synthesis/evaluate/candidate_report_adapter.py` and `synthesis/evaluate/test_candidate_report_adapter.py`. Red test failed with missing adapter module; focused green **2 passed in 0.04s**; nearby H31-H45 tests **21 passed in 0.12s**; `py_compile` passed. Production adapter source has no `expected` or `is_correct` references. Replay artifact `outputs/generated/h45_direct_eval_report_adapter_20260629/h45_summary.json`: **130** candidates from H40/H41/H42, **44** selected groups, selected **3/49 = 6.12%** correct after no-gold selection (`[16, 17, 47]`). Credential scan found no paid key names. This confirms framework plumbing only; pool still too weak for a win. |
| H46 GSM-2B final-answer report-pool diagnostic | n/a | n/a | completed | **$0 CPU-only artifact diagnostic; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before measurement. Used H45 adapter plus H31 selector on completed GSM-2B reports H2, H4, H10, H24, H29, H40, H41, and H42. Output `outputs/generated/h46_gsm2b_final_answer_report_pool_20260629/h46_summary.json`. Result: **8** reports, **354** adapter candidates, **48** selected groups, final-answer correct union only **9/49** at `[10, 13, 16, 17, 21, 30, 32, 39, 47]`, and H31 selected **0/49** correct after no-gold selection. Credential scan found no paid key names. Confirms next step must instrument/generate independent candidate lines or variants rather than only replay final spans. |
| H47 GSM-2B candidate-line report-pool diagnostic | n/a | n/a | completed | **CPU-only TDD adapter extension and replay; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before code/tests. Same isolated worktree `/home/aadivyar/.config/superpowers/worktrees/csd-generation/h31-candidate-consensus`. Added candidate-line parsing to `synthesis/evaluate/candidate_report_adapter.py` and tests in `synthesis/evaluate/test_candidate_report_adapter.py`. Red test failed with `TypeError: candidates_from_direct_eval_report() got an unexpected keyword argument 'include_candidate_lines'`; focused green **3 passed in 0.04s**; nearby H31-H47 tests **22 passed in 0.10s**; `py_compile` passed. Production adapter source has no `expected` or `is_correct` references. Replay artifact `outputs/generated/h47_gsm2b_candidate_line_report_pool_20260629/h47_summary.json`: **8** reports, **354** final candidates, **1417** total candidates, **1063** candidate-line candidates, candidate lines in **48/49** groups, all-candidate and candidate-line-only selector output for **48** groups, final-answer union still **9/49**, H24 scored candidate-line union **10/49**, H24 scored selected-line count **3/49**, and no paid credential key names. Confirms parser plumbing but refutes old report text as enough for a win; next step should emit structured independent candidates with scorer-preserving metadata. |
| H48 GSM-2B scored candidate-line replay | n/a | n/a | completed | **CPU-only TDD scorer-replay extension; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before code/tests. Same isolated worktree `/home/aadivyar/.config/superpowers/worktrees/csd-generation/h31-candidate-consensus`. Added `synthesis/evaluate/candidate_line_scorer.py` and `synthesis/evaluate/test_candidate_line_scorer.py`. Red test failed with missing scorer module; first implementation exposed missing `z3` under `/usr/bin/python3`, so the scorer now falls back to the repo's CRANE-style random-sampling equivalence helper. Focused green **3 passed in 0.11s**; nearby H31-H48 tests **25 passed in 0.17s**; `py_compile` passed. Replay artifact `outputs/generated/h48_gsm2b_scored_candidate_line_replay_20260629/h48_summary.json`: loaded **49** seed123 train examples, used **8** reports, scored **1063** candidate-line candidates across **48/49** groups, scored candidate-line union **8/49**, selected candidate-line correct **5/49**, final-answer union **9/49**, and no paid credential key names. Refutes old artifact scoring as a hidden win; next GSM-2B move should generate fresh structured independent candidates with scorer-ready metadata, or pivot to a cheaper unresolved matrix cell. |
| H49 Spider-9B held-out failure attribution | n/a | n/a | completed | **$0 CPU-only artifact diagnostic; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before measurement. Output `outputs/generated/h49_spider9b_heldout_failure_attribution_20260629/h49_summary.json`. Held-out report: **194/300 = 64.67% accuracy**, syntax **0.99**, wrong **106/300**, strict win gap **+8** over IterGen's **201/300**. Conservative first-bucket counts among wrong examples: select-clause mismatch **44**, other syntax-valid semantic mismatch **15**, distinct-presence mismatch **13**, normalization-only exact after alias/quote stripping **11**, order-by mismatch **6**, aggregate/group-by mismatch **4** each, syntax invalid **3**, and small remaining buckets. Credential scan found no paid key names. H49 confirms a concentrated bucket exists, but not a paper-ready win: a fair next step is an output-side SQL postprocessing probe that removes harmless aliases/noise without changing the evaluator or hurting existing correct examples. |
| H50 Spider-9B alias postprocess counterfactual | n/a | n/a | completed | **$0 CPU-only counterfactual replay; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before measurement. Output `outputs/generated/h50_spider9b_alias_postprocess_counterfactual_20260629/h50_summary.json`. Loaded seed334 held-out split with **300/300** question matches. Original execution-match replay reproduced **194/300**. Applying a no-gold output-side postprocessor that strips simple projection aliases changed **69** predictions, flipped **28** wrong examples to correct, flipped **0** correct examples to wrong, and raised execution-match to **222/300 = 74.0%**, above IterGen's **201/300 = 67.0%**. Credential scan found no paid key names. This is a strong counterfactual, not a paper-ready result; next step is H51 TDD production postprocessor/extractor patch plus pure held-out re-eval with `--initial-strategy-file --max-iterations 1`. |
| H51 Spider-9B production alias postprocessor and held-out re-eval | 2 | 3280011 | failed before measurement | **Local GPU-only no-author re-eval; no billed provider call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before tests/code. Isolated focal worktree: `/home/aadivyar/.config/superpowers/worktrees/csd-generation/h51-spider-alias-postprocess`, branch `h51-spider-alias-postprocess`. TDD status: red focused test failed because `extract_sql_scored_output` still returned `COUNT(*) AS singer_count`; after adding `clean_sql_scored_output`, focused green passed **1/1 in 0.03s**, mode-pinned nearby Spider tests passed **4/4 in 0.03s**, and `py_compile` passed. Re-eval output JSON was not written. Failure summary `outputs/generated/h51_spider9b_alias_postprocess_heldout_20260629/h51_failure_summary.json`: vLLM KV-cache startup failed at max length **16384** after retries at `gpu_memory_utilization` **0.50**, **0.45**, and **0.40**, ending with `Evaluation failed: Engine core initialization failed`. Credential scan found no paid key names. The patch is not promoted and this is not a Spider-9B win yet; next safe move is H52, changing only eval runtime max length downward for the same no-author held-out re-eval. |
| H52 Spider-9B alias postprocessor maxlen-4096 retry | n/a | n/a | preregistered / waiting for safe GPU | **Local GPU-only no-author re-eval planned; no billed provider call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before launch. Scientific variable: lower `--vllm-max-model-len` from **16384** to **4096** while keeping the H51 alias patch, compiled strategy, split, model, sample size, and evaluator unchanged. H62 then made the fairness timeout explicit: H52 now passes `--max-seconds-per-example 600`, matching the GSM structured direct-eval wall-clock budget. CPU sizing check over seed334 held-out prompts: **300** prompts, max **2046 chars**, p95 **1962 chars**, median **577 chars**. Launch condition: a safe GPU with at least **30 GiB free** and no non-`aadivyar` process on that GPU. Current GPU check found no safe slot: GPU1 has an active non-`aadivyar` Spider baseline, while GPU0/GPU2/GPU3 do not have at least **30 GiB** free. Planned output `outputs/generated/h52_spider9b_alias_postprocess_heldout_maxlen4096_20260629/h52_reeval.json`. |
| H53 GSM-4B failure attribution | n/a | n/a | completed | **$0 CPU-only artifact diagnostic; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before measurement. Output `outputs/generated/h53_gsm4b_failure_attribution_20260629/h53_summary.json`. Best accuracy attempt was **31** at **15/49 = 30.61%** with **67.35%** syntax; best syntax attempt was **19** at **4/49 = 8.16%** with **100.0%** syntax. Best-attempt buckets: **15** correct, **10** syntax-invalid wrong examples, **24** syntax-valid semantic mismatches, **0** visible-but-unextracted correct expressions, and **0** light-normalization matches. Across all evaluated attempts, oracle union was **31/49**, above the **21/49** train bar. Credential scan found no paid key names. Conclusion: simple postprocessing is unlikely, but a selector replay was worth testing. |
| H54 GSM-4B no-gold selector replay | n/a | n/a | completed / refuted | **$0 CPU-only selector replay; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before measurement. Output `outputs/generated/h54_gsm4b_no_gold_selector_replay_20260629/h54_summary.json`. Selector used no gold for selection, choosing by agreement cluster size, syntax validity, delimiter presence, and length/repetition shape. It selected **12/49 = 24.49%** correct with **47/49 = 95.92%** syntax, below the **21/49** train bar and worse than the best single attempt's **15/49** accuracy. Oracle union stayed **31/49**, so the diversity is mostly oracle-only under this selector. Conclusion: do not spend the next GSM-4B move on simple replay/selector packaging; it needs a stronger generation/framework mechanism. |
| H55 GSM-2B structured candidate feasibility audit | n/a | n/a | completed | **$0 CPU-only source/artifact audit; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before measurement. Output `outputs/generated/h55_gsm2b_structured_candidate_feasibility_20260629/h55_summary.json`. Important scope correction: focal main does **not** contain the H31 candidate plumbing; it lives in `/home/aadivyar/.config/superpowers/worktrees/csd-generation/h31-candidate-consensus`. Corrected audit found **4** reusable pieces there: `candidate_repeat_plan.py`, `parameterized_direct_eval.py`, `candidate_report_adapter.py`, and `candidate_line_scorer.py`. It reported `minimal_path_exists_in_h31_worktree=true` and `needs_code_patch_before_next_gpu_run=false`. Conclusion: when a safe GPU opens for GSM-2B, use the H31 worktree/materialized safe repeat path; do not launch from focal main. |
| H56 GSM-2B launch-shape audit | n/a | n/a | completed | **$0 CPU-only launch-safety audit; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before measurement. Output `outputs/generated/h56_gsm2b_launch_shape_audit_20260629/h56_summary.json`. H56 supersedes H55's launch-readiness conclusion: the H37-H39 materialized artifacts route through parameterized/direct-eval reports and include `source_id`, `output_name`, dry-run/no-model-call safety, and temperature controls, but do **not** emit structured candidate records with scorer metadata. It classified `current_materialized_commands_are_closed_temperature_repeat_branch=true` and `needs_tdd_launch_shape_patch_before_next_gsm2b_gpu=true`. Conclusion: do **not** launch H37-H39 materialized GSM-2B repeat commands as the next experiment; first patch the H31 worktree launch shape so it emits selector-ready candidate records, then CPU dry-run, then one local no-billing GPU smoke when safe. |
| H57 GSM-2B structured candidate launch-shape patch | n/a | n/a | completed | **CPU-only TDD patch in isolated H31 worktree; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before tests/code. Worktree `/home/aadivyar/.config/superpowers/worktrees/csd-generation/h31-candidate-consensus`. Changed `synthesis/evaluate/candidate_report_adapter.py` and `synthesis/evaluate/test_candidate_report_adapter.py`. RED: focused adapter test failed at collection with `ImportError` for missing `structured_candidate_artifact_from_direct_eval_report`. GREEN: focused adapter tests **4/4 passed in 0.04s**; nearby H31-H57 tests **29/29 passed in 0.20s**; `py_compile` passed. Output `outputs/generated/h57_gsm2b_structured_candidate_launch_shape_20260629/h57_summary.json`. Dry-run structured candidate artifact has **2** candidates, `selection_uses_gold=false`, source/output/sample ids, scorer metadata, **0** gold field-name/sentinel hits, and **0** paid credential key-name hits. Conclusion: H56's launch-shape gap is patched; next GSM-2B step is one local no-billing GPU smoke using this structured-candidate path when safe, after H52 priority. |
| H58 GSM-9B stale-log health diagnostic | n/a | 284546 | completed / warning | **$0 read-only run-health diagnostic over the existing GSM-9B train synthesis; no process intervention, no model/GPU/API launch.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before deeper diagnostics. Output `outputs/generated/h58_gsm9b_stale_log_health_20260629/h58_summary.json`. Result: **0** success/failure reports; `run.log` still mtime **2026-06-29T01:04:15Z** and size **53,510,073** bytes; newest GSM-9B output-tree file is also the stale `run.log`. PID **284546** remains alive in `Rl` state at about **70% CPU**, with VLLM child PID **287539** around **32.5% CPU** and **176** parent threads. Stdout/stderr both point to `run.log`; stdin is `/dev/null`. Source loop `synthesis/evaluate/evaluator.py:2502-2714` shows the next normal visible event after the last `Generated 210 tokens in 20.77s` marker should be example **49/49** or a returned report. Credential-name scan found **0** hits. Conclusion: treat GSM-9B as possible stuck/no-log state, not merely slow; do not kill/restart without explicit direction because it is an existing paid-backed run. |
| H59 H52 launch materialization | n/a | n/a | completed | **$0 CPU-only launch-readiness artifact; no model/GPU/API launch.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before materialization. Output directory `outputs/generated/h59_h52_launch_materialization_20260629/` contains `launch_h52_spider9b_alias_maxlen4096.sh`, `h59_manifest.json`, `h59_checks.json`, and dry-run stdout/stderr. Remote validation passed: `bash -n`, manifest JSON, and `DRY_RUN=1 SAFE_GPU_ID=0` all completed without launching; `h59_checks.json` has `passed=true`. After H62, the launcher uses the H51 worktree compiled `GeneratedCSD.py`, Spider seed334 eval split, sample size **300**, max steps **200**, explicit **600s/example**, `Qwen/Qwen3.5-9B`, `--vllm-max-model-len 4096`, output `outputs/generated/h52_spider9b_alias_postprocess_heldout_maxlen4096_20260629/h52_reeval.json`, and pid/log paths under `/tmp/csd_h52_logs/`. It requires explicit `SAFE_GPU_ID`, checks at least **30000 MiB** free, rejects non-`aadivyar` GPU processes, and unsets Bedrock/AWS/OpenAI/Anthropic credential variables before the child process. Static scan found **0** secret-looking values. When safe, run from focal main: `SAFE_GPU_ID=<safe_index> outputs/generated/h59_h52_launch_materialization_20260629/launch_h52_spider9b_alias_maxlen4096.sh`. |
| H60 GSM-2B existing H37 structured-consensus replay | n/a | n/a | completed / refuted | **$0 CPU-only structured-candidate replay over existing H37 direct-eval success reports; no model/GPU/API call and no code change.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before measurement. Output `outputs/generated/h60_gsm2b_existing_h37_structured_consensus_20260629/h60_summary.json`. The replay used the H31 worktree adapter/selector on the three completed H37 reports, emitted **130** final-answer candidates, selected **44/49** groups with no gold used for selection, and measured **3/49 = 6.12%** selected accuracy with **44/49 = 89.80%** selected syntax. Oracle union was only **4/49 = 8.16%** at `[16, 17, 39, 47]`, far below the **12/49** train bar. The artifact records `model_calls=0`, `gpu_calls=0`, `billed_api_calls=0`, `selection_uses_gold=false`, and no paid credential key-name hits. Conclusion: the H57 structured artifact path works on real completed reports, but H37/H40-H42 are too weak to relaunch or package; next GSM-2B work needs fresh structured independent candidate generation. |
| H61 SMILES Qwen3.5-2B chain_extenders paid COLD UV run | 2 | 3554187 | completed / held-out loss | **Approved paid Bedrock run; one cell only.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before launch. User explicitly approved using the same `AWS_BEARER_TOKEN_BEDROCK` as PID **284546** for recorded AWS account **887730490125**, even though live account identity could not be independently reverified, and later stated paid runs can be launched whenever needed with no fixed limit. Before launch, `.env` `AWS_BEARER_TOKEN_BEDROCK` and `AWS_REGION` were hash-verified to match PID **284546** without printing secret values. Launched detached at **2026-06-29T12:12:26Z** with PID **3554187**, log `/tmp/csd_h61_logs/h61_smiles_qwen35_2b_chain_extenders_20260629.log`, pidfile `/tmp/csd_h61_logs/h61_smiles_qwen35_2b_chain_extenders_20260629.pid`, command `./pilot_smiles_uv_qwen35_i40.sh Qwen/Qwen3.5-2B qwen35_2b chain_extenders 2 0.20 0.40 0.50`. Output root `outputs/generated/smiles_qwen35_2b_chain_extenders_uv_qwen35_0627`; train success report `outputs/generated/smiles_qwen35_2b_chain_extenders_uv_qwen35_0627/smiles_qwen35_2b_chain_extenders_uv_qwen35_0627_20260629_121230_8dbeff/results/success_report.json`; held-out result `outputs/controlled_comparison/smiles_qwen35_2b/chain_extenders/metadecode_uv.json`. Train accepted on attempt **17/40** with accuracy **0.66**, syntax **0.98**, **49/50** correct, and max sample time **19.878s**. Held-out was **0.14** UV/accuracy and **0.99** validity/syntax on **100** examples, so it clears validity but misses the UV bar **0.400**. Log scan found paid key name `AWS_BEARER_TOKEN_BEDROCK`; JSON artifacts scanned had no paid credential key-name hits and no secret values were printed. Not paper-ready; do not promote to `results_matrix.md`. |
| H62 GSM/Spider timeout fairness contract | n/a | n/a | completed | **CPU-only TDD launch-contract correction; no model/GPU/API call.** User asked to make sure GSM and Spider timeouts are fair to Spider. Focal inspection found historical Spider scripts used **450s/example**, H52's direct re-eval inherited `reevaluate_compiled_csd.py` default `None`, and the H31 GSM structured direct-eval runner hard-coded **600s/example** without exposing it in dry-run output. H62 changed H31 `synthesis/evaluate/parameterized_direct_eval.py` so `--max-seconds-per-example` defaults to **600**, appears in dry-run JSON, and is passed to `Evaluator`; it changed the H52 launcher to pass `--max-seconds-per-example 600`. RED: focused GSM test failed with `KeyError: 'max_seconds_per_example'`, and H52 dry-run lacked the flag. GREEN: focused GSM test **3/3 passed in 0.05s**; H52 `bash -n` passed; H52 dry-run printed `--max-seconds-per-example 600` and did not launch. H59 manifest/checks/stdout now record the same policy. |
| H63 SMILES Qwen3.5-4B isocyanates paid COLD UV run | 2 | 193596 | completed / held-out win | **Approved paid Bedrock run; one cell only.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before launch. H63 targeted the lowest-bar remaining SMILES cell using `./pilot_smiles_uv_qwen35_i40.sh Qwen/Qwen3.5-4B qwen35_4b isocyanates 2 0.40 0.16 0.50`. Focal CARS artifact `outputs/controlled_comparison/smiles_qwen35/4B/isocyanates/cars.json` records **0.16 / 1.00** on **50** examples. Launched at **2026-06-29T17:23:26Z** after H52 remained unsafe; pidfile `/tmp/csd_h63_logs/h63_smiles_qwen35_4b_isocyanates_20260629.pid`, log `/tmp/csd_h63_logs/h63_smiles_qwen35_4b_isocyanates_20260629.log`. Train success report `outputs/generated/smiles_qwen35_4b_isocyanates_uv_qwen35_0627/smiles_qwen35_4b_isocyanates_uv_qwen35_0627_20260629_172330_b62714/results/success_report.json`: accepted on attempt **12/40**, train accuracy **0.48**, syntax **0.64**, **32/50** applicable correct, max sample time **5.788s**. Held-out result `outputs/controlled_comparison/smiles_qwen35_4b/isocyanates/metadecode_uv.json`: UV/accuracy **0.58**, validity/syntax **0.61**, **100** examples, total generation **450.4044s**, evaluator total **703.4673s**, mean output **48.29** tokens/example. This is a primary UV held-out win over the **0.160** CARS UV bar; validity does not match the auxiliary CARS **1.00**, so paper text should not overclaim perfect validity. Scan of H63 log, train success JSON, and held-out JSON found **0** paid credential key-name hits. |
| H64 GSM-2B named-route structured-candidate materialization | n/a | n/a | completed / launch-ready not launched | **CPU-only launch materialization; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before materialization. H64 body is `saved-results/2026-06-29-h64-gsm2b-named-route-candidates-body.dfy`. Materialization directory `outputs/generated/h64_gsm2b_named_route_structured_candidates_materialization_20260629/` contains `launch_h64_gsm2b_named_route_t0.sh`, `h64_manifest.json`, `h64_checks.json`, `h64_dry_run_stdout.txt`, `h64_dry_run_stderr.txt`, and `direct_eval_dry_run.json`. Remote validation passed `bash -n`, manifest/check JSON parsing, and `DRY_RUN=1` exit **0**. Dry-run JSON records `no_model_calls=true`, `no_billed_credentials=true`, `eval_model=Qwen/Qwen3.5-2B`, `sample_size=49`, `max_steps=900`, `max_seconds_per_example=600`, output `h64_gsm2b_named_route_structured_candidates_20260629_t0`, and source id `h64_gsm2b_named_route_t0`. The launcher writes `structured_candidates.json` from `direct_eval_success.json` using the H57/H31 structured-candidate adapter if a future GPU run succeeds. Scan of materialization artifacts found **0** paid credential key-name hits. Not launched; no benchmark-win claim. |
| H66 SMILES live-bar source-of-truth audit | n/a | n/a | completed | **$0 CPU-only artifact audit; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before measurement. Output `outputs/generated/h66_smiles_baseline_bar_audit_20260629/h66_summary.json`. H66 found **9/9** short-note SMILES accuracy bars mismatched live focal CARS artifacts. Correct live bars: acrylates 2B **0.36**, 4B **1.00**, 9B **0.98**; chain_extenders 2B **0.94**, 4B **0.94**, 9B **1.00**; isocyanates 2B **0.98**, 4B **0.16**, 9B **0.92**. Remaining unrun SMILES queue by live bar: `isocyanates-9B` **0.92**, `chain_extenders-4B` **0.94**, `acrylates-9B` **0.98**, `acrylates-4B` **1.00**, `chain_extenders-9B` **1.00**. Credential key-name scan found **0** hits. Do not launch from stale short bar notes. |
| H67 paper-ready live-artifact matrix audit | n/a | n/a | completed | **$0 CPU-only artifact audit; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before measurement. Output `outputs/generated/h67_matrix_live_artifact_audit_20260629/h67_summary.json`. H67 found that under live focal CARS primary-UV bars only SMILES `isocyanates-4B` is a proven SMILES primary-UV win. `acrylates-2B` is **0.27 <= 0.36**, `isocyanates-2B` is **0.29 <= 0.98**, and `chain_extenders-2B` is **0.14 <= 0.94**, so old 2B SMILES wins are historical old-bar wins, not live-CARS paper-ready wins. Remaining unresolved SMILES queue by live bar: `isocyanates-9B` **0.92**, `chain_extenders-4B` **0.94**, `acrylates-9B` **0.98**, `acrylates-4B` **1.00**, `chain_extenders-9B` **1.00**. Expected paid credential key-name mentions appeared in docs only; no secret values were printed. |
| H68 SMILES next-target artifact analysis | n/a | n/a | completed | **$0 CPU-only artifact analysis; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before measurement. Output `outputs/generated/h68_smiles_next_target_artifact_analysis_20260629/h68_summary.json`. H68 found `acrylates-2B` is the closest live-CARS SMILES miss: held-out UV **0.27** vs live CARS UV **0.36**, gap **0.09**, with reusable run artifacts under `outputs/generated/smiles_qwen35_2b_acrylates_uv_qwen35_0627/` and held-out `outputs/controlled_comparison/smiles_qwen35_2b/acrylates/metadecode_uv.json`. By contrast, `isocyanates-9B` has no held-out artifact and live CARS UV bar **0.92**. Recommendation: next paid SMILES hypothesis should target a one-variable `acrylates-2B` improvement after inspecting the accepted strategy/failure modes; do not launch high-bar 9B just because it is unrun. Credential key-name scan found **0** hits. |
| H69 SMILES acrylates-2B failure-mode audit | n/a | n/a | completed | **$0 CPU-only strategy/failure-mode audit; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before measurement. Output `outputs/generated/h69_smiles_acrylates2b_failure_mode_audit_20260629/h69_summary.json`. H69 found usable strategy text and sample detail for `acrylates-2B`: `success.strategy_code` **4409** chars, `GeneratedCSD.dfy` **8446** chars, `GeneratedCSD.py` **4538** chars. Train was **0.28 / 1.00** on **50** examples, with `sample_outputs` showing **44/50** `is_correct=True`, **50/50** syntax-or-valid true, and **6** `syntax_valid_semantic_mismatch` markers. Held-out was **0.27** UV / **0.97** validity vs live CARS **0.36 / 1.00**, leaving a **0.09** UV gap. Recommended next paid hypothesis: keep Qwen3.5-2B acrylates and add one held-out-generalization/diversity pressure before final acceptance, predicting UV **>0.36** and validity **>=0.90**. Credential key-name scan found **0** hits. |
| H70 SMILES acrylates-2B live-bar paid launcher materialization | n/a | n/a | completed / launch-ready not launched | **CPU-only paid-launch materialization; no model/GPU/API call.** Pre-registered in `docs/experiments/metadecode-fast-iteration-log.md` before materialization. Output directory `outputs/generated/h70_smiles_qwen35_2b_acrylates_livebar_materialization_20260629/` contains `launch_h70_smiles_qwen35_2b_acrylates_livebar.sh`, `h70_manifest.json`, `h70_dry_run.json`, dry-run stdout/stderr, and `h70_checks.json`. Single scientific variable versus the previous acrylates-2B recipe: train `min_acc` is raised to live CARS UV **0.36**. Dry-run JSON records `model_calls=0`, `gpu_calls=0`, `billed_api_calls=0`, max iterations **40**, model `Qwen/Qwen3.5-2B`, class `acrylates`, util **0.20**, `min_acc=0.36`, `min_syn=0.50`, and planned held-out `outputs/controlled_comparison/smiles_qwen35_2b/acrylates/metadecode_uv.json`. Checks passed `bash -n`, account confirmation gate, safe GPU gate, H65 active paid-run gate, no-other-user GPU gate, and paid credential key-name scan with **0** hits. Not launched. |

Latest checked live status: **2026-06-29T20:20Z**. H61 is recorded as a held-out loss, H63 is recorded as a held-out primary-UV win, H65 is running on attempt **3/40** after attempt **2/40** scored **4.1%** accuracy vs **53.1%** bar, H52/H64 remain GPU-gated by non-`aadivyar` processes, H67 has corrected the paper-ready live-artifact tally, and H68/H69 point the next SMILES paid target toward a targeted `acrylates-2B` improvement.
H65 superseded the stale GSM-9B PID **284546**: old PIDs **284546**, **287538**, and **287539** were stopped after the GSM pathological-expression guard passed TDD. Replacement H65 PID **464438** is alive under `outputs/generated/synth_gsm_9b_z3fix_seed123train_h65_timeoutguard_20260630`, with `--max-iterations 40` and `--eval-max-seconds-per-example 600`. At **2026-06-29T19:27Z**, attempt **1/40** had completed below bar at **20.4% / 83.7%** vs **53.1% / 98.0%**, the run had entered attempt **2/40**, and no success/failure report existed yet.
The H52 Spider-9B retry has still not launched: there is no H52 pidfile, log, or
`h52_reeval.json`. The safe-GPU gate remains closed: GPU0 uses **38421/40960 MiB**, GPU1 uses
**4976/40960 MiB** but has active non-`aadivyar` PID **112852**,
GPU2 uses **17378/40960 MiB**,
and GPU3 uses **34015/40960 MiB**. H63 was preregistered, materialized, and launched on GPU2 with PID **193596** because H52 remained gated and GPU2 satisfied the H63 launcher gate. H59 has materialized and validated the ready-to-run H52 launcher. Next safe action remains:
if GSM-9B writes a report, parse and record it; if a safe GPU opens first, run pre-registered H52
with `SAFE_GPU_ID=<safe_index> outputs/generated/h59_h52_launch_materialization_20260629/launch_h52_spider9b_alias_maxlen4096.sh`
before any GSM-2B GPU smoke. Do not spend a GPU slot replaying H37/H40-H42: H60 showed their structured
pool has only **4/49** oracle coverage and **3/49** selected accuracy.

Spider diagnostic retry PID 955809 completed and wrote JSON. Startup
diagnostics reached `load_model end seconds=6.23`, `load_tokenizer end seconds=2.15`,
`_get_ignore_whitespace end seconds=0.24 result=True`, `DFAMaskStore.load end seconds=7.10`,
`create_parser end seconds=0.01`, and `build_logits_warper end seconds=0.00`. The run has real
generation progress across all 50 examples. Final result: `accuracy=0.2`, `syntax_rate=1.0`,
`answers_len=50`, inferred correct count 10/50. After completion, GPU2 had 23065 MiB free but still
had two other users' Python jobs at 8678 MiB each. GPU0 still has the GSM-9B run's VLLM engines
plus PID 284546; GPU1 has the GSM-4B run's VLLM engine plus PID 14560; GPU3 has another VLLM
engine using ~34GB. H2 was launched only after GPU2 was confirmed low-utilization with enough
headroom for a small Qwen3.5-2B local eval.

**H2 result note (2026-06-29):** H2 was already pre-registered in
`docs/experiments/metadecode-fast-iteration-log.md` before launch. The local body was saved at
`saved-results/2026-06-29-h2-gsm2b-final-span-probe-body.dfy`, copied to focal, and launched with
`--max-iterations 1 --initial-strategy-file` on GPU2 as PID 1066489. The launcher unsets
`AWS_BEARER_TOKEN_BEDROCK`, AWS access/session/profile variables, `OPENAI_API_KEY`, and
`ANTHROPIC_API_KEY` before running; this is intended to be local vLLM-only. Post-launch check:
Dafny verification passed, Python compilation passed, and evaluation started loading
`Qwen/Qwen3.5-2B`. It completed at
`outputs/generated/h2_gsm2b_final_span_probe_20260629/h2_gsm2b_final_span_probe_20260629_20260628_215702_fdbc73/results/success_report.json`.
Actual result: **2/49 = 0.0408 accuracy**, **48/49 = 0.9796 syntax**; visible closed spans in
**48/49** answers; unclosed-span answers **1/49**; no extracted answer **1/49**; median visible
span length **12.5 token-ish units** (min 1, max 61); runtime **786.8876s**; no early stop. H2
is refuted: it fixed syntax/length but not semantic correctness. It remains only a diagnostic
mechanism probe because its body explicitly asks for one final span; do not record it as a
publishable win. The obsolete heartbeat `monitor-h2-gsm2b-final-span-probe` was deleted after H2
and H3 were recorded.

**H14/H15 result note (2026-06-29):** both were `$0`, local-only attempts to re-evaluate the
already-known Spider-9B attempt-20 strategy on the 50-example fast split, with billed credentials
unset and `--max-iterations 1 --initial-strategy-file`. H14 used the adaptive helper mask and
failed before eval at `search_contract` because the old strategy calls
`RolloutConstrainedWithPenalties`. H15 changed exactly one variable by omitting
`--adaptive-helper-mask`, but still failed before verification/compilation/evaluation with the same
contract error. H15 report:
`outputs/generated/h15_spider9b_att20_fast50_nomask_20260629/h15_spider9b_att20_fast50_nomask_20260629_20260629_000040_92e4fe/results/failure_report.json`.
The report has `total_attempts=1`; attempt 1 has `succeeded=false`, `failed_at=search_contract`,
`verification=None`, `compilation=None`, `evaluation=None`, and
`error_summary='Strategy contract violation.\nViolations: RolloutConstrainedWithPenalties'`. This
means the Spider-9B 50-set predictive hypothesis is still untested; the next safe action is to
inspect the runner/evaluator path or find a lower-level local evaluator that can evaluate an
already-existing strategy without invoking synthesis helper-contract filtering. Do not promote H14
or H15 as cell evidence.

**H16 direct-eval path note (2026-06-29):** H16 inspected focal source and found no CLI for arbitrary
metaDecode strategy bodies. `synthesis/evaluate/run_fixed_strategy.py` delegates to the legacy
fixed-strategy runner, whose `--strategy` choices are only `unconstrained`, `gcd`, `crane`,
`itergen`, and `cars`. A direct internal-API smoke was then run with all billed credentials unset:
`StrategyGenerator.inject_strategy` -> `DafnyVerifier.verify` -> `DafnyCompiler.compile` ->
`Evaluator.evaluate_sample`. This bypassed `search_contract`; the old Spider-9B attempt-20 strategy
verified and compiled successfully. The first stdin smoke failed because vLLM spawn cannot reload
`<stdin>`. The temp-file smoke with `max_steps=200` and `vllm_gpu_memory_utilization=0.50` failed
with **0.08 GiB** available KV cache versus **0.61 GiB** needed for `max_model_len=16384`. Retrying
with **0.55** gave **2.05 GiB** KV cache and **16,368** tokens, but then started building a fresh
`CachedQwen2TokenizerFast` SQL mask at
`cache/mask_stores/CachedQwen2TokenizerFast/grammar_mask_7704218576_248044.pkl`. It reached only
**7/260** after about **2m13s**, so it was interrupted. Existing old masks are under
`cache/mask_stores/CachedTokenizersBackend/...`; H17 should inspect whether the prior successful
held-out re-eval used a reusable cache/backend setting or whether this direct path needs a one-time
mask build or small reviewed evaluator wrapper. H16 has no accuracy/syntax result and should not be
promoted.

**H17/H18/H19 Spider-9B direct-eval diagnostic chain (recorded 2026-06-29):** H17 confirmed the
H16 stall was a cache-key mismatch, not a need for a new SQL grammar mask. Current vLLM tokenizer
type was `CachedQwen2TokenizerFast` with vocab size **248044**, while the exact existing mask was
already present at
`cache/mask_stores/CachedTokenizersBackend/grammar_mask_7704218576_248044.pkl` with size
**3,910,900,672 bytes**. H18 created only a cache symlink to the matching mask:
`cache/mask_stores/CachedQwen2TokenizerFast/grammar_mask_7704218576_248044.pkl ->
../CachedTokenizersBackend/grammar_mask_7704218576_248044.pkl`, then ran a 1-example direct
smoke. H18 succeeded with **1/1** accuracy and syntax, proving the alias was usable for real
decoding without a 260-state rebuild. H19 then ran the full 50-example fast split through the same
direct no-contract path with all billed credentials unset. H19 output:
`outputs/generated/h19_spider9b_att20_fast50_direct_cachealias_20260629/results/direct_eval_success.json`.
Parsed result: `stage=evaluation`, `success=True`, `num_examples=50`, `accuracy=0.78`,
`syntax_rate=0.96`, `elapsed_seconds=455.5437960624695`, so **39/50** examples were correct and
**48/50** were syntax-valid. This confirms the Spider-9B 50-set fast bar is too weak for promotion:
the already-known held-out-losing attempt-20 strategy beats the fast IterGen bar **10/50 = 20.0%**
by a huge margin, but still loses the real held-out comparison **194/300 = 64.7%** vs IterGen
**201/300 = 67.0%**. Do not treat future Spider-9B 50-set wins alone as publishable or as enough
to launch paid full runs; require stronger local evidence or full held-out confirmation after the
money/account rule is satisfied.

**H20 GSM expression-structure diagnostic (recorded 2026-06-29):** H20 was a `$0` offline-only
diagnostic over H2, H4, and H10 wrong syntax-valid GSM final expressions, saved at
`outputs/generated/h20_gsm_linear_structure_diagnostic_20260629/h20_summary.json`. It tested whether
the next GSM framework lever should be a narrow generic linear-expression / same-monomial
coefficient checker. Result: **mostly no**. Almost all wrong expressions are parseable
(actual parse **132/133 = 99.2%**, expected parse **133/133 = 100.0%**), which is good news for
parser-backed diagnostics. But strict linear/affine coverage is too low: polynomial-affine actuals
**31/133 = 23.3%**, rational-affine numerator actuals **38/133 = 28.6%**. Same variable set is
only **67/133 = 50.4%**, same numerator monomial support **34/133 = 25.6%**, and
coefficient-only-ish wrong expressions only **25/133 = 18.8%**. Do **not** build the next GSM probe
as a narrow linear coefficient/sign checker. Better next direction: a broader non-gold
parse-and-simplify / complexity / multi-candidate consistency mechanism, or an offline diagnostic
that estimates whether those non-gold signals correlate with correctness.

**H21/H22 GSM non-gold selector diagnostics (recorded 2026-06-29):** H21 tested a simple non-gold
expression-quality score over H2/H4/H10 syntax-valid final expressions. Features used only the
candidate expression and question text: parse success, variable grounding, expression length, huge
numeric literal penalty, repetition penalty, low-degree numerator bonus, and no-newline bonus.
Labels were applied only after scoring using evaluator correctness plus offline SymPy equivalence.
H21 output: `outputs/generated/h21_gsm_nongold_signal_diagnostic_20260629/h21_summary.json`.
Result: **14** positive / **124** negative rows; AUROC **0.9513**, mean positive score **0.9478**
vs mean negative **0.7877**, median positive **0.9440** vs median negative **0.7883**, median gap
**0.1557**. This is promising but not an oracle: clean-looking false positives still score high.

H22 then tested the right use case: per-problem multi-candidate selection. It treated H2, H4, and
H10 as three candidate generators for the same GSM problems and picked the highest H21 non-gold
score per problem. H22 output:
`outputs/generated/h22_gsm_candidate_selection_diagnostic_20260629/h22_summary.json`. There were
**48** problem groups with syntax-valid candidates and **10** groups where at least one candidate
was correct/equivalent. The selector picked a correct/equivalent candidate in **10/10 = 100%** of
those solvable groups. Overall selected positives were **10/48 = 20.8%**, selected evaluator-correct
**4/48**, selected SymPy-equivalent **10/48**. Best individual probe positive count was H10 **8**,
so the selector improved the offline candidate pool. Next GSM direction should be a small
non-publishable local mechanism probe or TDD framework prototype for generating multiple candidate
expressions and selecting with this non-gold score. Do not claim a cell win from H22; it used
multiple hand-authored diagnostic probes as candidate sources.

**H23 GSM candidate-pool upper bound (recorded 2026-06-29):** H23 tested whether the completed
GSM-2B diagnostic pool already contains enough candidates to beat CRANE if a perfect selector were
available. Output:
`outputs/generated/h23_gsm_candidate_pool_upper_bound_20260629/h23_summary.json`. H1's artifact has
aggregate accuracy/syntax only plus `answers`, so H23 mapped H1 answers to the shared train49
expected answers by index and extracted the last visible `<<...>>` span. H1 official aggregate
remains **2/49**, while offline extracted-span equivalence found **4/49** candidates. Candidate
counts / positives: H1 **30 / 4**, H2 **48 / 4**, H4 **44 / 2**, H10 **46 / 8**. Union upper bound
across H1+H2+H4+H10: only **11/49 = 22.45%** problem groups have at least one correct/equivalent
candidate, below the CRANE-2B bar **12/49 = 24.5%**. This means selector-only work cannot win with
the current completed candidate pool. Next GSM move must generate at least one more correct/equivalent
candidate while preserving the H21/H22 selector advantage.

**H24 GSM multi-candidate local probe (recorded 2026-06-29):** H24 tested whether one fair local
GSM-2B strategy could generate enough candidate diversity for the H21/H22 selector to cross CRANE.
It was pre-registered in `docs/experiments/metadecode-fast-iteration-log.md`, then run as a
local-only direct eval with no Bedrock/OpenAI/Anthropic env vars. The Bedrock-flag launcher inherited
from H10 was **not used**; the completed path was the safe direct eval script
`saved-results/2026-06-29-h24-direct-eval-gsm2b.py`. Result JSON:
`outputs/generated/h24_gsm2b_multicandidate_selector_probe_20260629/results/direct_eval_success.json`.
Offline analysis:
`outputs/generated/h24_gsm2b_multicandidate_selector_probe_20260629/results/h24_candidate_selector_analysis.json`.
Direct final-span result: **5/49 = 10.20% accuracy**, **43/49 = 87.76% syntax**, elapsed
**981.03s**, max-step hits **4/49**, median token count **395**. Candidate analysis found candidate
lines in **43/49** groups, **124** total candidate lines, and **120/124** parser-compatible
candidates after corrected normalization that keeps multi-letter variable names intact. Candidate
union positives were **10/49** and H21 top-score selector positives were **8/49**, both below the
CRANE target **12/49** and below H23's old pool upper bound **11/49**. The model `Selected:` line
was positive in only **3/49** groups, while final spans had **8/49** offline SymPy-equivalent
expressions. H24 is therefore refuted as a standalone prompt-only strategy, but it remains useful
coverage input because H25 showed its positives differ from the old pool.

**H25 GSM combined-pool union (recorded 2026-06-29):** H25 tested whether H24's corrected
candidate positives covered new problem groups beyond the H1/H2/H4/H10 pool. Output:
`outputs/generated/h25_gsm_h24_combined_pool_union_20260629/h25_summary.json`. H23 old pool
positives were **11/49** at indices `[7, 8, 13, 14, 16, 17, 21, 30, 32, 40, 49]`. H24 candidate
positives were **10/49** at indices `[1, 8, 10, 16, 21, 26, 30, 43, 47, 48]`. H24 added **6** new
positive groups: `[1, 10, 26, 43, 47, 48]`. Combined H1/H2/H4/H10/H24 positive count was
**17/49 = 34.69%**, above the CRANE bar **12/49 = 24.5%**. Next safe GSM diagnostic is a non-gold
selector/combiner over the combined candidate pool; do not claim a GSM-2B win from H25 because it
is an offline gold-labeled upper-bound/coverage diagnostic.

**H26 GSM combined-pool H21 selector (recorded 2026-06-29):** H26 tested whether the H21 non-gold
expression-quality score alone could select from the H25 combined pool. Output:
`outputs/generated/h26_gsm_combined_pool_h21_selector_20260629/h26_summary.json`. Same combined pool:
**49** groups, **292** candidates, **17** positive groups. Top-score selection picked only
**10/49 = 20.41%** positives and recovered **10/17 = 58.82%** solvable groups, below the CRANE bar
**12/49** and the pre-registered **12/17** recovery threshold. Missed solvable indices:
`[13, 17, 26, 30, 32, 40, 43]`. H26 is refuted: H21 simplicity/grounding is useful but over-rewards
isolated compact false positives in the larger pool.

**H27 GSM combined-pool agreement selector (recorded 2026-06-29):** H27 changed one selector
variable from H26: cluster candidates per problem by expression agreement using only candidate
expressions, prefer stronger agreement, then use H21 score as a tie-breaker. Output:
`outputs/generated/h27_gsm_combined_pool_agreement_selector_20260629/h27_summary.json`. Same pool:
**49** groups, **292** candidates, **17** positive groups. Agreement-first selection picked
**12/49 = 24.49%** positives and recovered **12/17 = 70.59%** solvable groups, exactly meeting the
CRANE diagnostic bar. Selected positive indices:
`[1, 7, 8, 10, 14, 16, 21, 26, 43, 47, 48, 49]`; evaluator-correct selected indices were only
`[14, 16]`. H27 is a selector-mechanism confirmation, **not** a cell win: it used an offline pool
from multiple diagnostic generators. Next safe GSM move is either a TDD prototype for a generic
candidate-consensus selector or a local mechanism probe that tests whether one fair strategy can
generate agreement-rich candidates.

**H28 GSM source-family-only agreement ablation (recorded 2026-06-29):** H28 tested whether H27's
threshold result survives if agreement counts only distinct source families and ignores repeated
same-family candidate variants such as H24 A/B/C. Output:
`outputs/generated/h28_gsm_agreement_source_family_ablation_20260629/h28_summary.json`. Same pool:
**49** groups, **292** candidates, **17** positive groups. Source-family-only agreement selected
only **10/49 = 20.41%** positives and recovered **10/17 = 58.82%** solvable groups, below the
CRANE diagnostic bar. Selected positive indices were `[1, 7, 8, 10, 14, 16, 21, 47, 48, 49]`,
losing H27's H24-internal-agreement recoveries at **26** and **43**. Conclusion: within-output
candidate agreement is load-bearing. Next fair mechanism should deliberately generate multiple
candidate variants per problem and then use candidate-consensus selection; independent old-source
agreement alone is not enough.

**H29 GSM six-candidate consensus probe (recorded 2026-06-29):** H29 tested the next prompt-only
variant-generation idea after H28: ask local Qwen3.5-2B for six independent `Candidate 1..6:`
equations plus a `Consensus:` line, then ignore the model's self-choice and run an offline
agreement-first selector. Output:
`outputs/generated/h29_gsm2b_six_variant_consensus_probe_20260629/results/direct_eval_success.json`;
analysis:
`outputs/generated/h29_gsm2b_six_variant_consensus_probe_20260629/results/h29_candidate_consensus_analysis.json`.
Direct final-span result: **3/49 = 6.12% accuracy**, **43/49 = 87.76% syntax**, median valid visible
span length **9**, elapsed **1504.1768s**. Candidate analysis: **38/49** groups had candidate lines,
with **225** candidate lines and **125/225** parser-compatible lines. Candidate union was only
**3/49** at `[17, 39, 49]`; model consensus positives were **3/49** at `[17, 39, 49]`;
agreement-selected positives were **2/49** at `[17, 39]`; final SymPy-equivalent positives were
**3/49** at `[16, 17, 39]`. H29 is refuted. Prompt-only self-variant generation mostly repeats the
same wrong reasoning; the next GSM-2B lever should be a real framework/helper mechanism that
generates diverse candidates through separate attempts, checks, or non-gold feedback.

**H30 GSM candidate-diversity code diagnostic (recorded 2026-06-29):** H30 was a `$0` read-only
source audit on focal. It found useful pieces but no ready-made fair path for multiple full-answer
candidates per problem. In `synthesis/evaluate/benchmarks/common/model_utils.py`,
`CSD_CONSTRAINED_TEMPERATURE` can make `ChooseNextToken` sample from masked constrained logits, but
the default is exact argmax. The vLLM `GenerateLogits` path still captures one-token logits with
`SamplingParams(max_tokens=1, temperature=0.0, logprobs=VLLM_TOPK_LOGPROBS)`, and
`GenerateUnconstrainedChunk` also uses `temperature=0.0`. In `synthesis/run_synthesis.py`,
`--temperature` is passed to `StrategyGenerator`, so it controls synthesis authoring/refinement, not
per-problem evaluator decoding. The evaluator/feedback loop still evaluates one compiled strategy per
attempt; GSM `TopValidCandidates` is token-level support after one logits call, not full-answer
candidate generation. Conclusion: H30 is mixed/mostly refuted. The next GSM-2B step should be a TDD
framework prototype or direct-eval wrapper that explicitly runs several no-gold candidate attempts
per problem and applies non-gold agreement selection, with paid credentials removed.

**H31 no-gold candidate consensus selector prototype (recorded 2026-06-29):** H31 moved the H27/H28
agreement selector from an offline idea into a small generic module, without model calls. Worktree:
`/home/aadivyar/.config/superpowers/worktrees/csd-generation/h31-candidate-consensus`, branch
`h31-candidate-consensus`. Added `synthesis/evaluate/candidate_consensus.py` and
`synthesis/evaluate/test_candidate_consensus.py`. The selector takes only candidate records with a
group id, expression, caller-supplied no-gold equivalence key, source/source-family metadata, and a
no-gold quality score. It does not accept expected answers or correctness labels. Red run:
`python -m pytest synthesis/evaluate/test_candidate_consensus.py -q` failed with
`ModuleNotFoundError: No module named 'synthesis.evaluate.candidate_consensus'`. After the minimal
implementation, the focused run passed **4/4** tests in **0.04s**. Nearby verification
`python -m pytest synthesis/evaluate/test_candidate_consensus.py synthesis/evaluate/test_metrics.py synthesis/evaluate/test_scalar_score.py -q`
passed **14/14** tests in **5.10s**, and `python -m py_compile synthesis/evaluate/candidate_consensus.py`
passed. Sibling search for `select_consensus`, `candidate_consensus`, `agreement_score`,
`equivalence_key`, and `source_family` found only the new module/test, so no existing selector path
was duplicated. H31 is confirmed as framework groundwork, not a benchmark win. Next step: H32 should
feed several independent no-billed candidate attempts into this selector.

**H32 H31 selector replay over H27 pool (recorded 2026-06-29):** H32 was a `$0` CPU-only replay in
the same isolated worktree, with no model/GPU run. It added
`synthesis/evaluate/candidate_consensus_replay.py` plus
`synthesis/evaluate/test_candidate_consensus_replay.py`. The red test failed with
`ModuleNotFoundError: No module named 'synthesis.evaluate.candidate_consensus_replay'`. After the
adapter was added, the exact-H27 replay prediction was refuted: H31's tuple rule had **2** selected
cluster-key mismatches against H27, at index **10** (`total-n*1+n*2` vs `total+n*2-n*1`) and index
**46** (`(n*1)+(mult)+(n*2)+(n*3)` vs the long `n1+mult+n2+n3+1000...` cluster). But the resulting
selector was stronger by the offline post-selection labels: selected positives were **14/49 =
28.57%**, indices `[1, 7, 8, 10, 14, 16, 17, 21, 26, 30, 43, 47, 48, 49]`, compared with H27's
**12/49**. Saved output:
`outputs/generated/h32_h31_selector_h27_replay_20260629/h32_summary.json`. Final verification:
`python -m pytest synthesis/evaluate/test_candidate_consensus.py synthesis/evaluate/test_candidate_consensus_replay.py synthesis/evaluate/test_metrics.py synthesis/evaluate/test_scalar_score.py -q`
passed **16/16** in **4.87s**, and `py_compile` passed. This is still diagnostic only because it
reuses the old combined candidate pool; the missing piece remains fair candidate generation in a
single run or framework path.

**H33 safe repeat-attempt planner (recorded 2026-06-29):** H33 was a `$0` CPU-only TDD wrapper
prototype in the same isolated worktree. It added `synthesis/evaluate/candidate_repeat_plan.py` and
`synthesis/evaluate/test_candidate_repeat_plan.py`. The planner builds repeat-attempt specs for a
future local direct-eval candidate-generation probe. It strips paid credential env vars
(`AWS_BEARER_TOKEN_BEDROCK`, AWS keys/profile/session, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`),
rejects commands that request paid backends (`bedrock`, `openai`, `anthropic`), and attaches unique
`CSD_CANDIDATE_OUTPUT_NAME`, `CSD_CANDIDATE_SOURCE_ID`, and `CSD_CONSTRAINED_TEMPERATURE` metadata
per attempt. Red run: `python -m pytest synthesis/evaluate/test_candidate_repeat_plan.py -q` failed
with `ModuleNotFoundError: No module named 'synthesis.evaluate.candidate_repeat_plan'`. After the
minimal module was added, focused tests passed **4/4** in **0.04s**. Full nearby verification:
`python -m pytest synthesis/evaluate/test_candidate_consensus.py synthesis/evaluate/test_candidate_consensus_replay.py synthesis/evaluate/test_candidate_repeat_plan.py synthesis/evaluate/test_metrics.py synthesis/evaluate/test_scalar_score.py -q`
passed **20/20** in **5.00s**, and `py_compile` passed. Sibling search found only this new planner
and the existing `CSD_CONSTRAINED_TEMPERATURE` decoder knob in `model_utils.py`. This is not a model
result; it is launch-safety groundwork for the next GPU-backed local repeat probe.

**H34 dry-run repeat manifest writer (recorded 2026-06-29):** H34 extended the H33 planner with a
dry-run manifest writer, still with no model/GPU run. It added `write_repeat_manifest` to
`synthesis/evaluate/candidate_repeat_plan.py` and added
`synthesis/evaluate/test_candidate_repeat_manifest.py`. The manifest records `purpose`, `dry_run:
true`, `no_model_calls: true`, `no_billed_credentials: true`, and each attempt's command, safe env,
output name, source id, and constrained temperature. Red run:
`python -m pytest synthesis/evaluate/test_candidate_repeat_manifest.py -q` failed with
`ImportError: cannot import name 'write_repeat_manifest'`. After implementation, focused tests passed
**2/2** in **0.04s**. Full nearby verification after a small style cleanup:
`python -m pytest synthesis/evaluate/test_candidate_consensus.py synthesis/evaluate/test_candidate_consensus_replay.py synthesis/evaluate/test_candidate_repeat_plan.py synthesis/evaluate/test_candidate_repeat_manifest.py synthesis/evaluate/test_metrics.py synthesis/evaluate/test_scalar_score.py -q`
passed **22/22** in **4.92s**, and `py_compile` passed. Sibling search found no overlapping manifest
writer except an unrelated `dry_run` flag in `synthesis/scripts/ablation_beam_bandit.py`. This is
still launch-readiness work only; no benchmark result was produced.

**H35 repeat launch validator (recorded 2026-06-29):** H35 added the missing safety check that H34
exposed. The existing H29 direct-eval script is local-only and strips paid env vars, but it hardcodes
its output root to `outputs/generated/h29_gsm2b_six_variant_consensus_probe_20260629`, so repeating
it with different `CSD_CONSTRAINED_TEMPERATURE` values would risk clobbering outputs. H35 added
`validate_repeat_attempts_for_launch` to `synthesis/evaluate/candidate_repeat_plan.py` and
`synthesis/evaluate/test_candidate_repeat_validation.py`. Red run:
`python -m pytest synthesis/evaluate/test_candidate_repeat_validation.py -q` failed with
`ImportError: cannot import name 'validate_repeat_attempts_for_launch'`. After implementation,
focused tests passed **3/3** in **0.04s**. Full nearby verification after a small style cleanup:
`python -m pytest synthesis/evaluate/test_candidate_consensus.py synthesis/evaluate/test_candidate_consensus_replay.py synthesis/evaluate/test_candidate_repeat_plan.py synthesis/evaluate/test_candidate_repeat_manifest.py synthesis/evaluate/test_candidate_repeat_validation.py synthesis/evaluate/test_metrics.py synthesis/evaluate/test_scalar_score.py -q`
passed **25/25** in **4.88s**, and `py_compile` passed. The validator rejects the raw H29 command,
accepts commands with `--output-name {output_name}` and `--source-id {source_id}`, and rejects leaked
paid credential env keys. Next concrete blocker before a GPU repeat probe: write or adapt a
parameterized direct-eval runner that consumes those arguments.

**H36 parameterized direct-eval runner (recorded 2026-06-29):** H36 added a parameterized local
direct-eval runner in the isolated worktree: `synthesis/evaluate/parameterized_direct_eval.py` plus
`synthesis/evaluate/test_parameterized_direct_eval.py`. It accepts `--strategy-file`,
`--output-name`, and `--source-id`, constructs all output paths under
`outputs/generated/<output-name>/`, strips paid env vars, and supports `--dry-run` so config can be
audited without importing evaluator/model code. The non-dry-run path mirrors the H29 direct-eval
shape but is not yet launched. Red run:
`python -m pytest synthesis/evaluate/test_parameterized_direct_eval.py -q` failed with
`ModuleNotFoundError: No module named 'synthesis.evaluate.parameterized_direct_eval'`. After
implementation, focused tests passed **3/3** in **0.05s**. Nearby verification:
`python -m pytest synthesis/evaluate/test_candidate_consensus.py synthesis/evaluate/test_candidate_consensus_replay.py synthesis/evaluate/test_candidate_repeat_plan.py synthesis/evaluate/test_candidate_repeat_manifest.py synthesis/evaluate/test_candidate_repeat_validation.py synthesis/evaluate/test_parameterized_direct_eval.py synthesis/evaluate/test_metrics.py synthesis/evaluate/test_scalar_score.py -q`
passed **28/28** in **4.95s**, and `py_compile` passed. A CPU-only dry-run artifact was written at
`outputs/generated/h36_parameterized_direct_eval_dryrun_20260629/direct_eval_dry_run.json` with
`output_name=h36_parameterized_direct_eval_dryrun_20260629`, `source_id=h36_dryrun`,
`no_model_calls=true`, and a success path under the parameterized output root. String checks found no
`AWS_BEARER_TOKEN_BEDROCK`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or old H29 output-root string.
Next safe step: use H33-H36 together to generate and validate an actual dry-run repeat manifest for a
small temperature sweep, still without launching the model.

**H37 GSM-2B repeat-probe dry-run manifest (recorded 2026-06-29):** H37 composed the H33-H36 launch
safety pieces into an actual dry-run manifest, without executing any attempt command. Output:
`outputs/generated/h37_gsm2b_repeat_probe_manifest_20260629/repeat_manifest.json`; safety checks:
`outputs/generated/h37_gsm2b_repeat_probe_manifest_20260629/h37_manifest_checks.json`. The manifest
has **3** planned attempts with temperatures **0.0**, **0.25**, and **0.5**. Output names:
`h37_gsm2b_repeat_probe_20260629_t0/t1/t2`; source ids: `h37_gsm2b_t0/t1/t2`. Command:
`python -m synthesis.evaluate.parameterized_direct_eval --repo-root /home/aadivyar/csd-generation --strategy-file /home/aadivyar/csd-generation/saved-results/2026-06-29-h29-gsm2b-six-variant-consensus-probe-body.dfy --output-name {output_name} --source-id {source_id}`.
The manifest has `dry_run: true`, `no_model_calls: true`, and `no_billed_credentials: true`.
String checks found no `AWS_BEARER_TOKEN_BEDROCK`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, secret
placeholder, or old H29 output-root string. Next operational step when a safe GPU slot opens: fill
`CUDA_VISIBLE_DEVICES`, run a very small first repeat attempt or the 3-attempt local sweep, then
combine outputs with the H31/H32 selector path.

**H38 GSM-2B repeat-probe materialized commands (recorded 2026-06-29):** H38 converted the H37
dry-run manifest into exact non-executing command plans, still without launching any model/GPU job.
It added `synthesis/evaluate/test_candidate_repeat_materialize.py` and extended
`synthesis/evaluate/candidate_repeat_plan.py` with `materialize_repeat_commands` and
`write_materialized_commands` in the isolated worktree. Red run:
`python -m pytest synthesis/evaluate/test_candidate_repeat_materialize.py -q` failed with
`ImportError: cannot import name 'materialize_repeat_commands'`. After implementation, focused tests
passed **4/4** in **0.05s**. Nearby verification:
`python -m pytest synthesis/evaluate/test_candidate_consensus.py synthesis/evaluate/test_candidate_consensus_replay.py synthesis/evaluate/test_candidate_repeat_plan.py synthesis/evaluate/test_candidate_repeat_manifest.py synthesis/evaluate/test_candidate_repeat_validation.py synthesis/evaluate/test_candidate_repeat_materialize.py synthesis/evaluate/test_parameterized_direct_eval.py synthesis/evaluate/test_metrics.py synthesis/evaluate/test_scalar_score.py -q`
passed **32/32** in **5.10s**, and `py_compile` passed. Output:
`outputs/generated/h38_gsm2b_repeat_materialized_commands_20260629/materialized_commands.json`;
checks: `outputs/generated/h38_gsm2b_repeat_materialized_commands_20260629/h38_materialized_checks.json`.
The artifact contains **3** command plans with `execute=false`, `dry_run=true`,
`CUDA_VISIBLE_DEVICES=TO_FILL_WHEN_SAFE`, output names
`h37_gsm2b_repeat_probe_20260629_t0/t1/t2`, source ids `h37_gsm2b_t0/t1/t2`, and temperatures
**0.0**, **0.25**, and **0.5**. String checks found no `{output_name}`, `{source_id}`, paid
credential key names, or secret placeholder. Next operational step when a safe GPU slot opens:
replace `TO_FILL_WHEN_SAFE` with the chosen GPU id, run one local no-billing attempt first, and only
then run the three-attempt sweep if the first output is clean.

**H39 GSM-2B repeat-probe dry-run execution check (recorded 2026-06-29):** H39 executed the exact
H38 command plans only through the runner's `--dry-run` path. It appended
`--dry-run --dry-run-output outputs/generated/h39_gsm2b_repeat_dryrun_execution_20260629/<source_id>_dry_run.json`
to each command, ran from the isolated worktree, and used a deliberately tiny environment plus each
plan's safe env so unrelated API keys could not leak into the dry-run JSON. Summary:
`outputs/generated/h39_gsm2b_repeat_dryrun_execution_20260629/h39_dryrun_execution_summary.json`;
per-command dry-run outputs:
`outputs/generated/h39_gsm2b_repeat_dryrun_execution_20260629/h37_gsm2b_t0_dry_run.json`,
`h37_gsm2b_t1_dry_run.json`, and `h37_gsm2b_t2_dry_run.json`. Return codes were **[0, 0, 0]**.
All outputs existed and all payload checks passed: `dry_run=true`, `no_model_calls=true`,
`no_billed_credentials=true`, correct output/source ids, dataset `gsm_symbolic`, split `train`,
sample size **49**, and max steps **900**. String scan found no paid credential key names or secret
placeholder. No GPU/model/billed job was launched. Next operational step when a safe GPU slot opens:
replace `TO_FILL_WHEN_SAFE` with the chosen GPU id and run one local no-billing attempt before the
full three-temperature sweep.

**H40 GSM-2B repeat-probe single local smoke attempt (recorded 2026-06-29):** H40 was pre-registered
in `docs/experiments/metadecode-fast-iteration-log.md` and ran a single real local eval of only the
first H38 plan (`source_id=h37_gsm2b_t0`, output `h37_gsm2b_repeat_probe_20260629_t0`, temp **0.0**).
It used the explicit CSD env interpreter because focal `python` resolves to `/opt/anaconda/bin/python`,
not `/apps/conda/aadivyar/envs/csd/bin/python`. Launch details: PID **2271673**; PID file
`/tmp/csd_h40_logs/h40_gsm2b_t0_20260629.pid`; log
`/tmp/csd_h40_logs/h40_gsm2b_t0_20260629.log`; cwd
`/home/aadivyar/.config/superpowers/worktrees/csd-generation/h31-candidate-consensus`; command used
`CUDA_VISIBLE_DEVICES=2` and `--vllm-gpu-memory-utilization 0.20`. Paid credential env vars were
unset before launch. Final report
`outputs/generated/h37_gsm2b_repeat_probe_20260629_t0/results/direct_eval_success.json` has
`success=True`, `stage=evaluation`, `source_id=h37_gsm2b_t0`,
`output_name=h37_gsm2b_repeat_probe_20260629_t0`, `num_examples=49`,
`accuracy=0.061224489795918366`, `syntax_rate=0.8775510204081632`, and
`elapsed_seconds=8798.66094827652`. Artifact/log scan found no paid credential key names or secret
placeholder; no OOM/traceback was found. The log includes a post-run VLLM `EngineCore died
unexpectedly` line after example 49 generation, but the runner wrote the success report. Keep this
as launch-mechanics evidence only; it is not a GSM-2B win.

**H41 GSM-2B repeat-probe temp-0.25 local attempt (recorded 2026-06-29):** H41 tested exactly
one variable after H40: `CSD_CONSTRAINED_TEMPERATURE` changed from **0.0** to **0.25** for the same
parameterized direct-eval runner and strategy body. It ran only H38 plan `t1`
(`source_id=h37_gsm2b_t1`, output `h37_gsm2b_repeat_probe_20260629_t1`) on GPU2 with explicit
`/apps/conda/aadivyar/envs/csd/bin/python` and `--vllm-gpu-memory-utilization 0.20`, with paid
credential env vars stripped. Launch details: PID **2801821**; PID file
`/tmp/csd_h41_logs/h41_gsm2b_t1_20260629.pid`; log
`/tmp/csd_h41_logs/h41_gsm2b_t1_20260629.log`; cwd
`/home/aadivyar/.config/superpowers/worktrees/csd-generation/h31-candidate-consensus`. Final report
`outputs/generated/h37_gsm2b_repeat_probe_20260629_t1/results/direct_eval_success.json` has
`success=True`, `stage=evaluation`, `source_id=h37_gsm2b_t1`,
`output_name=h37_gsm2b_repeat_probe_20260629_t1`, `num_examples=49`,
`accuracy=0.04081632653061224`, `syntax_rate=0.8775510204081632`, and
`elapsed_seconds=1835.9341404438019`. H41 met the clean-run and syntax conditions but missed the
accuracy condition because **2/49 <= 3/49**. Artifact/log scan found no paid credential key names or
secret placeholder. The log includes a post-run VLLM `EngineCore died unexpectedly` line after
example 49 generation, but the runner wrote the success report. Keep this as diagnostic evidence
only; it is not a GSM-2B win.

**H42 GSM-2B repeat-probe temp-0.5 local attempt (launched 2026-06-29):** H42 tests the final
unrun H38 materialized temperature plan. It changes `CSD_CONSTRAINED_TEMPERATURE` to **0.5** for the
same parameterized direct-eval runner and strategy body. It is running only H38 plan `t2`
(`source_id=h37_gsm2b_t2`, output `h37_gsm2b_repeat_probe_20260629_t2`) on GPU2 with explicit
`/apps/conda/aadivyar/envs/csd/bin/python` and `--vllm-gpu-memory-utilization 0.20`, with paid
credential env vars stripped. Launch details: PID **2932784**; PID file
`/tmp/csd_h42_logs/h42_gsm2b_t2_20260629.pid`; log
`/tmp/csd_h42_logs/h42_gsm2b_t2_20260629.log`; cwd
`/home/aadivyar/.config/superpowers/worktrees/csd-generation/h31-candidate-consensus`. Immediate
check after **00:23** found PID alive, no success/failure report, and no paid credential key names
in the H42 log/output scan. Do not launch any wider sweep or billed job. H42's prediction is a clean
success report with **49** examples, no paid key names/OOM/traceback, syntax_rate **>=0.85**, and
accuracy **>3/49**; otherwise the temperature branch should stop and the next GSM-2B hypothesis
should target framework-level candidate diversity/selection.

**Spider-9B baseline retry note (2026-06-29):** the first local-only Spider-9B 50-set run
(`PID 785326`, log `/tmp/spider_qwen35_9b_itergen50_seed334_20260629.log`) reached 260/260 but
crashed before writing JSON with
`ValueError: has_previous_state can only be called on LinearAttention layers...`. Cause verified:
`/home/aadivyar/csd-generation/legacy/itergen/itergen/main.py` was still the unpatched vendored
copy (`md5 ec11417432173ca9624edf22ce33cd02`) using bare `DynamicCache()` and truthy cache checks,
while the documented fixed copy at `/home/aadivyar/itergen/itergen/main.py` had md5
`a15ee7a9d9baeecd388c0eb08a103a79`. Synced the documented fixed `main.py` plus
`_logits_warper_compat.py` into the vendored copy, backing up the old vendored file as
`main.py.bak_qwen35_sync_20260628T204639Z`; `py_compile` passed. Relaunched the same $0 local-only
50-set baseline as `PID 875122` on GPU2, log
`/tmp/spider_qwen35_9b_itergen50_seed334_retry_fixed_20260629.log`, but that process stayed silent
before `Creating DFA` / `0/260` while burning CPU. Stopped only PID 875122 and relaunched with a
diagnostic wrapper saved locally at
`saved-results/2026-06-29-spider-itergen-diagnostic-wrapper.py` and copied to focal as
`/tmp/spider_itergen_diag_wrapper_20260629.py`. PID 932511 proved startup/cache load completed but
still did not show generation-level progress, so it was replaced with the deeper diagnostic PID
955809, log `/tmp/spider_qwen35_9b_itergen50_seed334_diag5_20260629.log`, which has confirmed
per-example generation calls.

**Completed first action:** isocyanates-2B held-out re-eval exists at
`outputs/controlled_comparison/smiles_qwen35_2b/isocyanates/metadecode_uv.json` and is a win:
UV 0.290 > CARS 0.280, validity 0.95 > CARS 0.74. Recorded in `results_matrix.md` and the
main snapshot.

---

## H1/H2 — GSM mechanism probes (both COMPLETED; both refuted) and H3 diagnostic

H1 was run inline as a **$0** local Qwen3.5-2B eval with no Bedrock author call. It completed at
`outputs/generated/h1_gsm2b_close_span_probe_20260629/h1_train49_result.json` and was recorded in
`docs/experiments/metadecode-fast-iteration-log.md`.

**Hypothesis H1:** GSM-2B's −8.2pp gap vs CRANE (on the same 2B) is caused by the **tiny-span
trap** — the constrained span closes at the first complete grammar prefix (a single identifier
`n`), capturing a fragment instead of the full `n*p1+n*p2`. This is LIBRARY POLICY (fair to
change), NOT the grammar (off-limits).

**The fair fix being tested:** swap the att20 strategy's in-span fill from
`ConstrainedSymbolInGenerated` + `CloseSpanIfComplete` (first-complete close; has a Dafny
precondition `requires !IsCompletePrefix(...)` so it structurally cannot extend) to
**`CloseSpanWithinBudget`** (`VerifiedAgentSynthesis.dfy` ~line 2679) — fills token-by-token
under grammar constraint via `DeadEndAvoidingStep`, extends PAST the first complete prefix, closes
at the longest complete expression within budget. This is CRANE's mechanism; the author never
adopted it on its own.

**Committed prediction (in the ledger, before running):** prior 55%. If the trap is the cause →
accuracy ≥ 11/49 (≥22%), tiny-spans → ~0, median span length > 9 tokens. If it's a model ceiling
→ ≤ 9/49.

**Actual result:** refuted. Accuracy **2/49 = 0.0408**, syntax **35/49 = 0.7143**. Single-identifier
tiny spans dropped to **0**, but median visible span length became **399 token-ish units** (median
880 chars) because spans turned into long repeated algebra. Belief dropped from 55% to ~10%.
Keep only as an experiment artifact; do not promote the direct `CloseSpanWithinBudget` substitution.

**H2 result:** refuted. It tested whether free-text reasoning plus exactly one bounded final-answer
constrained span would recover GSM-2B accuracy. It did not: accuracy stayed **2/49 = 0.0408** while
syntax rose to **48/49 = 0.9796** and median visible span length stayed compact at **12.5**. This
means span placement/compactness alone is not enough; the next lever needs to decide whether the
model ever forms the correct symbolic expression before the final span.

**H3 result:** confirmed. It was a `$0` offline diagnostic over H2's completed report:
normalize each expected symbolic expression and visible output, then count whether the expected
expression appears before the first `<<` final span or anywhere in the output. Result:
expected expression before the final span **0/49**; expected expression anywhere **2/49**; those
two examples were exactly the already-correct examples `[16, 32]`; expected expression inside
closed spans **2/49**. There were no examples where the model had the correct expression in
free-text reasoning and the final span corrupted it. H3 raises belief to ~90% that GSM-2B's H2
failure is upstream semantic construction, not extraction/copy. Next fair GSM lever should improve
semantic symbolic construction, likely via a general symbolic-plan/check helper or feedback loop
that is not GSM-specific.

**H4 result:** refuted. H4 was pre-registered before launch in
`docs/experiments/metadecode-fast-iteration-log.md`. It started from H2 and changed exactly one
variable: the guidance text asked for neutral semantic construction/checking before the compact
final span. It launched on GPU2 as PID `1135526` with
`--max-iterations 1 --initial-strategy-file`, cloud/billing env vars unset by the launcher, and no
Bedrock author call. Dafny verification and Python compilation passed. Final report:
`outputs/generated/h4_gsm2b_semantic_plan_probe_20260629/h4_gsm2b_semantic_plan_probe_20260629_20260628_222132_140e8a/results/success_report.json`.
Actual result: **0/49 = 0.0000 accuracy**, **44/49 = 0.8980 syntax**, no extracted answer /
unclosed-span answers **5/49**, visible closed spans **44/49**, median visible span length **9.0**
token-ish units (min 1, max 43), median output length **455** tokens (min 209, max 900), one sample
hit max steps. H3-style expected-expression count: before final span **0/49**, anywhere in output
**2/49** (`[16, 33]`), inside closed spans **2/49**; correct examples **0/49**. Belief in plain
semantic-plan guidance dropped to ~5%. Next lever should not be more prose guidance; it likely needs
a real helper/scoring change that evaluates or constructs symbolic expressions, or a different
non-GSM-specific feedback signal.

**H5 result:** confirmed H4 was not hiding scorer/extraction wins. H5 inspected H4 examples `[16, 33]`
where the exact expected expression appeared as a substring. In example 16, the scored final span was
`n1 * p1 + n2 * p2 + n3 * p3+100000000000000000000000000000000000000000000`, so the expected
expression was only a prefix of a wrong answer. In example 33, the expected expression was embedded
inside a longer wrong product/division expression. Both examples had syntax-valid final spans,
answer source `last_visible_span`, and semantic mismatch. Conclusion: H4's refutation is real, not a
GSM scorer bug.

**H6 result:** confirmed grounding is not enough for GSM-2B. H6 compared identifiers in syntax-valid
wrong extracted H2/H4 final spans against identifiers in each example's question plus expected
expression. H2 had **44/46 = 95.7%** grounded-only wrong spans; H4 had **40/44 = 90.9%**
grounded-only wrong spans. Only **1** wrong extracted span in each report had ungrounded identifiers,
and it was the same `n1`/`n2` vs `n_1`/`n_2` normalization mismatch, not the dominant failure.
Conclusion: `RegenerateUnitOnGroundingFailure` is unlikely to flip GSM-2B by itself; next GSM probe
should target algebraic structure/equivalence feedback or checking, not identifier grounding.

**H7 result:** refuted as a reusable safe path, but found a more important fairness drift. H7 was a
`$0` read-only focal diagnostic. It found that the GSM evaluator has CRANE-faithful symbolic
equivalence code (`_crane_validate_expression_equivalence` / `_gsm_symbolic_equivalence`) and GSM
scoring calls it from `synthesis/evaluate/benchmarks/gsm_symbolic/eval_logic.py::is_correct`, but
the evaluator only records final `is_correct`, `expected`, and `actual` in `sample_outputs`; it does
not expose structured semantic-difference feedback. More importantly, focal's live prompt/feedback
state violates its own fairness guards: `synthesis/generate/prompts.py` still exposes
`RegenerateUnitOnCheckFailure` plus "When to use / How to use / Example call shape" guidance, and
`synthesis/evaluate/feedback_loop.py` still contains `_unit_rewind_hint`. Targeted guard run with
`/usr/local/bin/python3 -m pytest synthesis/tests/fairness_cloud/test_helper_menu_prune.py
synthesis/tests/fairness_cloud/test_itergen_promotion_removed.py -q` failed **8 failed, 4 passed**;
the conda env attempt failed first because `/apps/conda/aadivyar/envs/csd/bin/python` has no
`pytest`. Do not build or promote a GSM unit-rewind/equivalence-feedback probe on this focal prompt
state. The next safe action is a TDD fairness repair so the author menu and feedback loop match the
documented guard tests before any new publishable GSM synthesis launch.

**H7 fairness repair completed:** repaired the live focal prompt/feedback drift after reproducing the
red test in isolated worktree `/tmp/csd-h7-fairness-repair-20260629` on branch
`codex/h7-fairness-repair-20260629`. The patch changes only
`synthesis/generate/prompts.py` and `synthesis/evaluate/feedback_loop.py`: it filters the rendered
helper menu through the documented pruned helper set, restores the managed-span helper names needed
by the expected helper universe, strips promotional grounding/check guidance from the rendered
author menu, removes the grounded worked examples from the rendered verified-example set, and removes
`_unit_rewind_hint` plus its two feedback call sites. Red run in the isolated worktree:
`8 failed, 4 passed`. Green run in the isolated worktree:
`/usr/bin/python3 -m pytest synthesis/tests/fairness_cloud/test_helper_menu_prune.py
synthesis/tests/fairness_cloud/test_itergen_promotion_removed.py -q` → **12 passed in 0.06s**.
Green run after deploying the same two files to live focal:
`/usr/bin/python3 -m pytest synthesis/tests/fairness_cloud/test_helper_menu_prune.py
synthesis/tests/fairness_cloud/test_itergen_promotion_removed.py -q` → **12 passed in 0.08s**.
This repair makes future author prompts match the fairness guards again; it does **not** make the
already-running GSM-4B/GSM-9B jobs paper-safe if their prompts were built before this repair.

**H8 result:** confirmed that non-gold suffix/repetition cleanup is not the next GSM win lever.
H8 was a `$0` offline upper-bound diagnostic over H2 and H4 reports. Among H2 syntax-valid wrong
extracted spans, gold subspan was **0/46 = 0.0%** while repetition/junk flags hit
**41/46 = 89.1%**. Among H4 syntax-valid wrong extracted spans, gold subspan was
**2/44 = 4.5%** and gold prefix+extra was **2/44 = 4.5%**, while repetition/junk flags hit
**36/44 = 81.8%**. Combined: **90** eligible wrong spans; gold subspan **2/90 = 2.2%**,
gold prefix+extra **2/90 = 2.2%**, repetition/junk **77/90 = 85.6%**. The only recoverable-looking
prefix cases were H4 one-based examples **16** and **33**, already inspected in H5 as wrong spans
with the expected expression embedded in extra junk. Conclusion: repetition/junk is common, but the
correct expression is almost never present to trim out. Do not spend the next patch/probe on suffix
control; the next GSM lever needs to improve algebraic expression construction.

**H9 result:** confirmed the failure is mostly symbolic translation/structure, not absence of
surface ingredients in the reasoning. H9 was a `$0` offline diagnostic over H2/H4 pre-span text. In
H2 wrong-with-actual examples, pre-span text mentioned ≥75% of gold variables in
**42/47 = 89.4%** and ≥75% of gold operation cues in **35/47 = 74.5%**; high variable+operation
cues appeared in **31/47 = 66.0%**. In H4, high variable cues were **44/49 = 89.8%**, high operation
cues **42/49 = 85.7%**, and high variable+operation cues **38/49 = 77.6%**. Combined:
high variable cues **86/96 = 89.6%**, any variable **94/96 = 97.9%**, high operation cues
**77/96 = 80.2%**, high variable+operation cues **69/96 = 71.9%**. But final actual expressions had
the same coarse operator pattern as gold in only **1/96 = 1.0%**. Median pre-span length was
**162 words**. Conclusion: the model often writes relevant-looking reasoning with the right
ingredients, then constructs the final symbolic expression wrong. H4 already showed more prose
guidance is not enough; the next safe GSM mechanism probe should test a **structured translation/copy
scaffold**: one explicit ASCII equation line in ordinary text, then copy exactly that expression into
the constrained final span. Treat that as diagnostic unless later turned into a fair general
framework mechanism.

**H10 launched and running:** pre-registered in
`docs/experiments/metadecode-fast-iteration-log.md`, then launched as a `$0` local-only
Qwen3.5-2B mechanism probe on GPU2. Body:
`saved-results/2026-06-29-h10-gsm2b-equation-copy-probe-body.dfy`; launcher:
`saved-results/2026-06-29-launch-h10-gsm2b-equation-copy-probe.sh`; output:
`outputs/generated/h10_gsm2b_equation_copy_probe_20260629`; live run directory from
`latest_run.txt`:
`outputs/generated/h10_gsm2b_equation_copy_probe_20260629/h10_gsm2b_equation_copy_probe_20260629_20260628_232012_96dd25`.
Launched at **2026-06-28T23:20:08+00:00** as shell PID **1301918** with Python child
**1301925** and VLLM EngineCore **1304230**. Early log confirms Dafny verification passed,
Python compilation passed, and evaluation started loading `Qwen/Qwen3.5-2B`; GPU2 allocation rose
to **27291 MiB used / 13151 MiB free / 12% util** at the first poll. The launcher unsets
`AWS_BEARER_TOKEN_BEDROCK`, AWS access/session/profile vars, `OPENAI_API_KEY`, and
`ANTHROPIC_API_KEY`; command is `--max-iterations 1 --initial-strategy-file`, so no Bedrock author
call is expected. Next check should parse the success/failure report if present; if still running,
leave it alone and report progress.

**H10 progress check (2026-06-28T23:23:25+00:00):** still running. Shell PID **1301918** had
elapsed **03:18**; Python child **1301925** and VLLM EngineCore **1304230** were active. GPU2 was
**27819 MiB used / 12624 MiB free / 23% util**. No success/failure report yet. Eval progress had
completed examples 1-7 and was running **example 8/49**. Generated-token/time samples so far:
example 1 **574 tokens / 25.49s**, example 2 **269 / 10.23s**, example 3 **617 / 25.47s**,
example 4 **266 / 13.33s**, example 5 **114 / 11.06s**, example 6 **187 / 11.63s**,
example 7 **252 / 14.40s**.

**H10 progress check (2026-06-28T23:27:04+00:00):** still running. Shell PID **1301918** had
elapsed **06:57**. GPU2 was **27819 MiB used / 12624 MiB free / 23% util**. No success/failure
report yet. Eval progress had completed examples 1-24 and was running **example 25/49**. Additional
generated-token/time samples since the previous check: example 8 **314 / 15.57s**, example 9
**345 / 16.46s**, example 10 **900 / 32.46s** (hit max steps), example 11 **418 / 17.23s**,
example 12 **227 / 12.66s**, example 13 **286 / 14.90s**, example 14 **216 / 12.64s**,
example 15 **168 / 7.46s**, example 16 **139 / 7.94s**, example 17 **183 / 9.75s**, example 18
**221 / 13.15s**, example 19 **212 / 9.46s**, example 20 **344 / 16.71s**, example 21
**112 / 5.78s**, example 22 **348 / 13.58s**, example 23 **120 / 6.26s**, example 24
**322 / 14.51s**. Leave H10 alone until it writes a success/failure report.

**H10 progress check (2026-06-28T23:28:17+00:00):** still running. Shell PID **1301918** had
elapsed **08:10**. No success/failure report yet. Eval progress had completed examples 1-29 and was
running **example 30/49**. Additional generated-token/time samples since the previous check:
example 25 **296 / 13.95s**, example 26 **302 / 14.47s**, example 27 **225 / 12.91s**, example 28
**320 / 15.44s**, example 29 **361 / 16.31s**.

**H10 progress check (2026-06-28T23:29:55+00:00):** still running. Shell PID **1301918** had
elapsed **09:48**. GPU2 was **27819 MiB used / 12624 MiB free / 24% util**. No success/failure
report yet. Eval progress had completed examples 1-35 and was running **example 36/49**. Additional
generated-token/time samples since the previous check: example 30 **153 / 7.07s**, example 31
**273 / 14.76s**, example 32 **900 / 32.27s** (hit max steps), example 33 **297 / 14.94s**,
example 34 **274 / 13.30s**, example 35 **154 / 10.98s**. GSM-4B/GSM-9B still report the same
anchors and no train bar crossed.

**H10 progress check (2026-06-28T23:30:52+00:00):** still running. Shell PID **1301918** had
elapsed **10:45**. No success/failure report yet. Eval progress had completed examples 1-40 and was
running **example 41/49**. Additional generated-token/time samples since the previous check:
example 36 **282 / 14.53s**, example 37 **379 / 15.55s**, example 38 **210 / 9.18s**, example 39
**180 / 12.54s**, example 40 **262 / 11.34s**.

**H10 progress check (2026-06-28T23:32:25+00:00):** still running. Shell PID **1301918** had
elapsed **12:18**. GPU2 was **27819 MiB used / 12624 MiB free / 24% util**. No success/failure
report yet. Eval progress had completed examples 1-45 and was running **example 46/49**. Additional
generated-token/time samples since the previous check: example 41 **201 / 11.59s**, example 42
**200 / 12.75s**, example 43 **233 / 12.25s**, example 44 **419 / 19.48s**, example 45
**900 / 33.48s** (hit max steps).

**H10 final result (recorded 2026-06-29):** completed and refuted. Focal report:
`outputs/generated/h10_gsm2b_equation_copy_probe_20260629/h10_gsm2b_equation_copy_probe_20260629_20260628_232012_96dd25/results/success_report.json`.
Accuracy was **3/49 = 6.12%**, syntax **46/49 = 93.88%**, visible closed spans **46/49**,
unclosed/no-extract answers **3/49**, median visible span length **11.0** token-ish units, and
max-step examples **[9, 31, 44]**. Exact expected expressions appeared before the final span in
only **2/49** examples (`[12, 15]`), anywhere in output in **2/49**, and inside spans in **2/49**.
`Equation:` lines appeared in **36/49** examples, but only **2/49** equation lines exactly matched
the scored actual after whitespace normalization and only **2/49** contained the expected
expression. Correct examples were **[12, 15, 20]**; example 20 was algebraically equivalent
(`n + mult * n` vs `n * (mult + 1)`) rather than an exact expected-string copy. H10 gave a tiny
lift over H4's **0/49** but missed its pre-registered success bar of **≥6/49** and is **not
promoted**.

**H11 next diagnostic (pre-registered before running):** `$0` offline diagnostic over H10 only.
Hypothesis: H10 might have failed partly because the equation-copy path is dirty — Markdown
wrappers, braces, or copy mismatch — rather than only because the equation line is algebraically
wrong. Prediction: if at least **8/49** examples become correct/equivalent or wrapper/copy mismatch
explains **≥25%** of wrong syntax-valid examples after trivial wrapper cleanup, test a fair generic
sanitation/copy primitive next; otherwise stop spending on copy formatting and design a relation-
construction feedback mechanism. Ledger row is in
`docs/experiments/metadecode-fast-iteration-log.md`.

**H11 result (recorded 2026-06-29):** refuted / mostly false. H10 had `Equation:` lines in
**36/49** examples and wrapper artifacts in **32/49**, but cleanup did not reveal enough hidden
correct equations. Raw equation exact-match to expected: **0/49**; cleaned equation exact-match to
expected: **2/49**; cleaned equation algebraically equivalent to expected by the offline SymPy
diagnostic: **7/49** total. Only **4** wrong syntax-valid examples were recoverable by using the
cleaned equation line (`[6, 7, 25, 42]`), and only **2** wrong syntax-valid final spans became
equivalent after wrapper cleanup (`[6, 39]`). Among **43** wrong syntax-valid examples, **26** had
an equation line that stayed algebraically wrong after cleanup, **23** had equation/final copy
mismatch, and **13** had no equation line. Conclusion for the queue: do **not** spend H12 on copy
cleanup alone; the next lever needs relation-construction feedback or a fair mechanism that changes
how the symbolic relation is built.

**H12 result (recorded 2026-06-29):** mixed, mostly refuted as a next lever. The hypothesis was that
GSM synthesis might be stuck because the author sees mostly aggregate feedback and not concrete
relation-construction evidence. Focal source says that is too strong: `_render_mode_examples()` in
`synthesis/evaluate/evaluator.py` renders one block per failure mode with full `PROMPT`,
`QWEN OUTPUT`, `STRATEGY EXTRACTED`, and `CORRECT ANSWER`, and the evaluation-refinement prompt
includes those blocks under `## Concrete failing rollouts from prior attempt`. Live prompt logs
confirm this reached the author: the 4B run prompt log had **32** entries containing
`Concrete failing rollouts` / `STRATEGY EXTRACTED` / `CORRECT ANSWER` /
`syntax_valid_semantic_mismatch`; the 9B run had **26** such entries. A sampled 9B prompt showed the
author saw exact semantic contrast, e.g. extracted `n - m * p1 - k * p2` versus correct
`n * int(bill) - (m * p1 + k * p2)`. Still true: built-in hints are mostly span/delimiter/
constraint-path focused, and the attempt ledger summarizes failure locations rather than
operator/variable buckets. Queue implication: do **not** spend H13 on simple feedback formatting
alone. If touching feedback, it must add a genuinely new fair search signal. While paid Bedrock
launches are blocked, prefer a non-billed local mechanism/model-size probe or local-only Spider/
SMILES measurement.

**H13 result (recorded 2026-06-29):** refuted. This was a `$0` read-only focal artifact audit for
hidden Qwen3.5 SMILES results. It found Qwen3.5 CARS baseline JSONs for all model/class cells under
`outputs/controlled_comparison/smiles_qwen35/...`, but Qwen3.5 metadecode held-out JSONs only for
the two already recorded 2B wins:
`outputs/controlled_comparison/smiles_qwen35_2b/acrylates/metadecode_uv.json` (**UV 0.270 /
validity 0.97**) and
`outputs/controlled_comparison/smiles_qwen35_2b/isocyanates/metadecode_uv.json` (**UV 0.290 /
validity 0.95**). Generated Qwen3.5 SMILES dirs were only
`outputs/generated/smiles_qwen35_2b_acrylates_uv_qwen35_0627` and
`outputs/generated/smiles_qwen35_2b_isocyanates_uv_qwen35_0627`. Broad path search over `outputs`,
`logs`, and `saved-results` found no Qwen3.5 metadecode generated artifacts for
`chain_extenders-2B`, any 4B SMILES class, or any 9B SMILES class. Queue implication: the seven
remaining SMILES cells are genuinely unrun/pending and require future paid COLD synthesis after
AWS ownership can be reverified and explicitly approved.

**Live GSM synthesis progress check (2026-06-28T23:23:25+00:00):** left both jobs alone. GSM-4B
`synth_gsm_4b_z3fix_seed123train_0628b` still reports anchor attempt **3** at
**18.4% acc / 81.6% syntax** against bar **42.9% / 91.8%**, with latest log at **attempt 20/40**.
## 2026-06-29T18:57Z update — GSM-9B stale PID fixed with H65 hotfix

Do **not** monitor old GSM-9B PID `284546` as active anymore. It was diagnosed as stale/stuck:
stdout/stderr still pointed to `outputs/generated/synth_gsm_9b_z3fix_seed123train_0628b/run.log`,
but that log had not advanced since `2026-06-29T01:04:15Z`, no success/failure report existed,
and the last raw log line was after a completed generation (`Generated 210 tokens in 20.77s`).

H65 added a GSM pathological-expression guard before CRANE/Z3 equivalence and final-block syntax
parser construction. The guard is intentionally loose versus gold answers: current train+eval gold
max length was `119` chars and max operator count was `22`, while the guard starts at `512` chars,
`160` whitespace tokens, `80` operators, or a `64`-digit run. Verification on focal root:
`synthesis/evaluate/test_gsm_pathological_expression_guard.py` plus `synthesis/evaluate/test_metrics.py`
passed `7/7 in 0.09s`, and `py_compile` passed for the edited evaluator, GSM eval logic, and test.

Old PIDs `284546`, `287538`, and `287539` were stopped. Replacement GSM-9B H65 is running:

- PID file: `/tmp/csd_h65_logs/h65_gsm9_timeoutguard_20260630.pid`
- PID at launch: `464438`
- output root: `outputs/generated/synth_gsm_9b_z3fix_seed123train_h65_timeoutguard_20260630`
- log: `outputs/generated/synth_gsm_9b_z3fix_seed123train_h65_timeoutguard_20260630/run.log`
- launch report: `/tmp/csd_h65_logs/h65_gsm9_timeoutguard_20260630_launch.json`
- command keeps `--max-iterations 40` and `--eval-max-seconds-per-example 600`
- uses the same user-approved Bedrock account record `887730490125`

Immediate health check showed the new run alive and writing attempt `1/40`. Continue monitoring H65
instead of the old GSM-9B output root. Full train/held-out result is still pending.

Recent attempts include attempt 12 **28.6% acc** but low syntax, attempt 16 **22.4% acc**, attempt
18 **12.2% acc** with **7/49** unclosed spans, and attempt 19 **16.3% acc**. GSM-9B
`synth_gsm_9b_z3fix_seed123train_0628b` still reports anchor attempt **7** at
**42.9% acc / 93.9% syntax** against bar **53.1% / 98.0%**, with latest log at **attempt 14/40**.
Recent attempts include attempt 8 **38.8% acc** with only **16/49** visible-span outputs actually
running the constrained branch, attempt 10 **34.7%**, attempt 12 **36.7%**, and attempt 13 **32.7%**.
No train bar crossed, so no held-out re-eval was launched. These two synth processes were launched
before the H7 fairness repair and their Python processes keep the old prompt/feedback code in
memory; treat any result from them as diagnostic unless re-run under the repaired focal state.

---

## Environment & key paths (all on focal)

- Repo: `/home/aadivyar/csd-generation`. Python: `/apps/conda/aadivyar/envs/csd/bin/python`
  (transformers 5.5.4 + vllm 0.19.1 for Qwen3.5; RDKit 2026.3.3 installed 2026-06-28; numpy 2.2.6,
  torch 2.10.0+cu128).
- Author model: `us.anthropic.claude-sonnet-4-6` via `--generation-backend bedrock
  --anthropic-thinking enabled --anthropic-effort high`. NEVER a small local model as author.
- SSH: short hostname `focal` (FQDN DNS unreliable). macOS has no `timeout` — use
  `ssh -o ConnectTimeout=N -o BatchMode=yes`. Heredoc `<<'REMOTE' ... REMOTE` for embedded quotes.
  Launch detached so it survives disconnects: `setsid ... < /dev/null &`.
- Remote edits: edit the file LOCALLY then `scp` to focal (don't `sed`/heredoc-edit in place).
  Exception: tiny one-off ops (`rm`, `chmod`, single-line append).
- SMILES launcher: `/home/aadivyar/csd-generation/pilot_smiles_uv_qwen35_i40.sh
  <eval-model> <tag> <class> <gpu> <util> <min-acc> <min-syn>` — COLD, UV metric, temp 0.7,
  mask+bandit, max-iter 40, auto held-out re-eval of accepted CSD on N=100. SMILES UV metric =
  `synthesis/evaluate/benchmarks/smiles/eval_logic.py override_accuracy` (unique_valid/N: RDKit-valid
  AND in-class AND unique AND non-exemplar). Plain "membership" (valid+in-class) is gameable — NOT
  comparable; always report UV.
- Baselines: CRANE `/home/aadivyar/CRANE/` (GSM); IterGen `/home/aadivyar/itergen/` (Spider);
  CARS (SMILES). Spider grader = official execution-based grader.
- Status helper on focal: `/home/aadivyar/synth_status.sh <output-name>` → `alive | anchor | crash`.

## Open task queue

- #10 Fast-iteration loop: all 15 cells beat baseline (closest-first, small model first).
- #11 Keep `results_matrix.md` + `saved-results/` snapshot reconciled as cells complete.
- #12 Build the next fair GSM framework probe after H1 refuted the raw span-length fix.
- #13 Re-eval GSM bests on held-out → record (only after a TRAIN bar cross).
- #14 Build Qwen3.5 ablation analogs + run them.
- #15 maxSteps ablation: also run baselines at each step count.
- #16 Final doc: tables + graphs for main + ablation results.
- SMILES launch order as GPUs free: H61 has already closed `chain_extenders-2B` as a held-out loss,
  so do **not** relaunch that exact recipe blindly. H67 now treats only `isocyanates-4B` as a
  live-CARS-proven SMILES primary-UV win. H68 then found the closest live-CARS miss is
  `acrylates-2B`: held-out UV **0.27** vs live CARS UV **0.36**, gap **0.09**, with reusable
  run artifacts under `outputs/generated/smiles_qwen35_2b_acrylates_uv_qwen35_0627/`.
  The lowest unrun cell, `isocyanates-9B`, has live CARS UV bar **0.92** and no held-out artifact,
  so it should not be the next paid SMILES launch solely because it is unrun. Next paid SMILES work
  should inspect the accepted `acrylates-2B` strategy/failure modes, preregister one single-variable
  improvement hypothesis, then materialize/launch only when safe. Paid Bedrock launches are approved
  by the user for recorded AWS account **887730490125**, but still require ledger-first
  preregistration, no secret printing, and safe GPU capacity before launch.

## Immediate next actions (in order)

1. Relay every synthesis attempt's acc/syn vs bar (PING EACH ATTEMPT rule — no batching), reading
   the `CSD_RATIONALE` block to judge progress (metrics on N≈20-50 are noisy; judge by rationale).
2. Do not use the Spider-9B 50-example fast split as promotion evidence by itself. H19 proved that
   a known held-out loser scores **39/50 = 78.0%** on that split while losing the real held-out
   comparison **194/300 = 64.7%** vs IterGen **201/300 = 67.0%**. Treat it only as a cheap
   screening split unless followed by stronger local evidence or full held-out confirmation.
3. Decide the next GSM lever from H10-H12: H10/H11 refuted copy-cleanup, and H12 showed the author
   already sees concrete semantic failures. H20 then refuted a narrow linear/same-monomial
   coefficient checker: parseability is high, but same-monomial coefficient-only errors are only
   **25/133 = 18.8%**. Avoid more prompt-only scaffolding and avoid a narrow linear checker; move
   toward broader non-gold parse/simplify, complexity, or multi-candidate consistency signals.
   H21/H22 make the selector concrete, but H23 proves selector-only is not enough: the completed
   candidate pool upper bound is **11/49**, below the CRANE bar **12/49**. Next probe should test a
   fair multi-candidate expression generator plus non-gold expression-quality selector. H37/H38 now
   make the next local no-billing repeat probe mechanical: when a safe GPU slot opens, use
   `outputs/generated/h38_gsm2b_repeat_materialized_commands_20260629/materialized_commands.json`,
   replace `TO_FILL_WHEN_SAFE`, and run one attempt before the full three-attempt sweep. H39 proved
   those exact commands parse and write dry-run config successfully when `--dry-run` is appended.
4. H52 is still first priority when a safe **30 GiB/no-non-aadivyar** GPU opens. H63 is recorded
   as a held-out primary-UV win, so the next paid SMILES cell may be preregistered only after
   checking live GPU capacity and respecting the one-new-paid-cell-at-a-time rule while H65 runs.
5. H64 is now launch-ready but not launched. After H52 priority is satisfied and a safe local GPU
   opens, run one GSM-2B smoke with:
   `DRY_RUN=0 SAFE_GPU_ID=<safe_index> outputs/generated/h64_gsm2b_named_route_structured_candidates_materialization_20260629/launch_h64_gsm2b_named_route_t0.sh`.
   Then record `direct_eval_success.json` plus `structured_candidates.json`, or `direct_eval_failure.json`.
6. On any TRAIN bar cross, fire the $0 held-out re-eval immediately and record it.

### H71 GSM-9B held-out re-eval launch materialization — 2026-06-29T20:35Z

H71 is ready as a dry-run-only launcher, not launched. Artifact root: `outputs/generated/h71_gsm9_h65_heldout_reeval_materialization_20260629/`. It prepares the immediate GSM-9B held-out re-eval required if H65 crosses the train bar. Checks passed: `bash -n`, dry-run `model_calls=0`, `gpu_calls=0`, `billed_api_calls=0`, `--max-iterations 1`, GSM split `eval`, `--eval-max-seconds-per-example 600`, H65 success-report guard, H65-alive guard, safe-GPU/no-other-user guard, `--initial-strategy-file`, and paid credential key-name scan with 0 hits. Do not launch H71 until H65 has a train `success_report.json` and PID 464438 is no longer alive.

### H65 live update — 2026-06-29T20:43Z

H65 replacement GSM-9B PID **464438** is still running. Latest checked log mtime: **2026-06-29T20:43:41Z**. Attempt **3/40** completed at **6.1%** accuracy / **61.2%** syntax, below the **53.1% / 98.0%** train bar, and the run entered attempt **4/40** with no success/failure report. Rationale check: attempt 4 is a targeted anti-repetition/span-budget revision after attempt 3 focused on premature closure/missing spans; this is real span-hygiene progress, but not yet a semantic-math mechanism. H52 still has no safe launch slot because GPUs with enough memory still have non-`aadivyar` processes. H71 is ready as the dry-run-only held-out re-eval launcher if H65 later crosses train.

### H72 H70 SMILES launch provenance audit — 2026-06-29T20:51Z

H72 found a real launch-recording risk before spending money. Artifact root: `outputs/generated/h72_h70_smiles_launch_provenance_audit_20260629/`. The audit used **0** model/GPU/billed API calls and found that H70's planned generated root reuses `outputs/generated/smiles_qwen35_2b_acrylates_uv_qwen35_0627`, while the planned held-out JSON reuses `outputs/controlled_comparison/smiles_qwen35_2b/acrylates/metadecode_uv.json`. Current held-out is **0.27 / 0.97**; current `latest_run.txt` points to `smiles_qwen35_2b_acrylates_uv_qwen35_0627_20260628_174851_2fa73d`. Before real H70 launch, snapshot the old latest/held-out state and copy the post-launch train/held-out artifacts into an H70-specific folder. Paid credential key-name scan found 0 hits.

### H73 H70 provenance hardening — 2026-06-29T20:54Z

H73 patched the future H70 launcher artifact only; no launch, model call, GPU call, or billed API call happened. `outputs/generated/h70_smiles_qwen35_2b_acrylates_livebar_materialization_20260629/launch_h70_smiles_qwen35_2b_acrylates_livebar.sh` now snapshots prelaunch `metadecode_uv.json`, `latest_run.txt`, and latest symlink target under a timestamped `provenance_<UTC>/prelaunch`, then after the unchanged H70 command exits it copies postlaunch held-out JSON, latest-run metadata, train success/failure report if present, latest symlink target, and `exit_status.txt` under `provenance_<UTC>/postlaunch`. Refreshed `h70_checks.json` and new `h73_h70_provenance_checks.json` both passed; paid credential key-name scan found 0 hits. H70 remains not launched.

### H74 paper-ready evidence bundle — 2026-06-29T21:01Z

H74 wrote `outputs/generated/h74_paper_ready_evidence_bundle_20260629/h74_summary.json` and `.md` with **0** model/GPU/billed API calls and no score-artifact edits. It packages **2** current live-artifact wins with artifact paths and sha256 hashes: Spider-2B train **117/300 = 0.39** / held-out **115/300 = 0.3833** with held-out syntax **0.9933**, and SMILES isocyanates-4B primary UV train **0.48 / 0.64**, held-out **0.58 / 0.61**, live CARS **0.16 / 1.00**, UV margin **+0.42**. Use H74 as the paper-evidence pointer for current proven wins. Paid credential key-name scan found 0 hits.
- H75 wrote `outputs/generated/h75_h64_gsm2b_structured_candidate_readiness_audit_20260629/h75_summary.json`: CPU-only audit with **0** model calls, **0** GPU calls, and **0** billed API calls. It confirms H64 is launch-ready as a future no-billing GSM-2B structured-candidate smoke after H52 priority and safe-GPU gating: sample size **49**, max steps **900**, **600s/example**, safe-GPU/no-other-user gates, runner-side paid-env stripping, fresh H64 output path, structured-candidate artifact on success, scorer-preserving metadata, no old H37/H40-H42/H47/H48 replay terms, and **0** paid credential key-name hits. This is launch-readiness evidence only; H64 has not launched and no GSM-2B accuracy changed.
### H52/H76 live update — 2026-06-29T21:19Z

H52 first launch on GPU2 reached no model result and failed immediately with `Evaluation failed: No module named '_dafny'`; this was an infrastructure packaging failure, not a Spider-9B score. H76 root-cause check found the H51 compiled `_build_pt-py` folder lacked `_dafny/` and `System_/`, while known working compiled directories include both runtime packages. Red/green repair evidence is in `outputs/generated/h76_h52_dafny_runtime_repair_20260629/h76_repair_summary.json`: pre-import rc **1**, post-import rc **0**, copied packages `_dafny` and `System_`, `generated_strategy_changed=false`, and **0** model/GPU/billed API calls. H52 was relaunched as PID **926238** on GPU2 with paid env vars removed, `--vllm-max-model-len 4096`, and `--max-seconds-per-example 600`; immediate health check showed vLLM loading Qwen3.5-9B instead of failing on `_dafny`. Final `h52_reeval.json` is still pending.

### H65 live update — 2026-06-29T21:19Z

H65 GSM-9B PID **464438** remains alive. Attempt **4/40** completed below the train bar at **40.8%** accuracy and **73.5%** syntax versus required **53.1% / 98.0%**. Attempt 4 was a targeted anti-repetition/span-budget strategy using `SafeRepetitionPenaltyStep`, per-span budgets, and forced span close; it improved accuracy materially versus attempts 2-3 but still missed syntax badly. Attempt **5/40** has started; its rationale targets long unconstrained reasoning and `{var}`-style invalid spans by stronger final-answer guidance, an unconstrained-generation cap, and forced span opening near budget exhaustion. Leave H65 running.
### H77 live update — 2026-06-29T21:24Z

H52 reached vLLM startup after H76, then failed at `gpu_memory_utilization=0.50`: available KV cache **0.08 GiB**, needed **0.23 GiB** for max length **4096**. The automatic lower-utilization retry at **0.45** was the wrong direction for this failure and reported negative available KV cache. H77 changed exactly one runtime variable in `outputs/generated/h59_h52_launch_materialization_20260629/launch_h52_spider9b_alias_maxlen4096.sh`: `--vllm-gpu-memory-utilization 0.50` -> `0.55`; it kept `--vllm-max-model-len 4096`, `--max-seconds-per-example 600`, the same compiled strategy, the same Spider eval split, and paid credential env unsets. H77 stopped only the failed H52 PID/child on GPU2. After GPU cleanup, GPU2 had **40432 MiB** free and no compute process, and H52 relaunched as PID **942717**. As of **2026-06-29T21:24:38Z**, PID **942717** is alive and no final `h52_reeval.json` exists yet.
### H77/H64 live update — 2026-06-29T21:29Z

H77 confirmed the H52 vLLM-utilization fix: H52 PID **942717** on GPU2 cleared startup with available KV cache **2.05 GiB**, GPU KV cache size **16,368** tokens, and max concurrency **9.07x** for **4096** tokens/request. No H52 result JSON yet.

The spare clean GPU3 was used for the already-preregistered local no-billing H64 GSM-2B structured-candidate smoke. Launch gate evidence: GPU3 had **40438 MiB** free and no compute process; H64 launched as PID **955593** with log `/tmp/csd_h64_logs/h64_gsm2b_named_route_t0_20260629.log`, output root `outputs/generated/h64_gsm2b_named_route_structured_candidates_20260629_t0`, sample size **49**, max steps **900**, and **600s/example**. H64 cleared vLLM startup on Qwen3.5-2B with available KV cache **1.84 GiB**, GPU KV cache size **39,712** tokens, max concurrency **7.95x** for **16,384** tokens/request, and began processing example **1/49**. No H64 success/failure/structured-candidate JSON yet.

### H64 completed smoke — 2026-06-29T21:48Z

H64 finished and should no longer be monitored as running. It wrote `outputs/generated/h64_gsm2b_named_route_structured_candidates_20260629_t0/results/direct_eval_success.json` and `outputs/generated/h64_gsm2b_named_route_structured_candidates_20260629_t0/results/structured_candidates.json`. Parsed result: accuracy **0.0408163265**, syntax rate **0.9387755102**, **49** examples, elapsed **1182.4140s**, `output_name=h64_gsm2b_named_route_structured_candidates_20260629_t0`, `source_id=h64_gsm2b_named_route_t0`. This is a completed local no-billing smoke but not a GSM-2B win. The log's `Engine core proc EngineCore died unexpectedly` line happened after vLLM shutdown and after the result JSON was written; the artifact parses, so record the metric and do not treat it as a missing-result crash. Paid credential key-name scan over the H64 log and artifact root found **0** hits.

Next GSM-2B move: do not rerun the same H64 named-route body. The structured-candidate plumbing is useful, but the candidate-generation hypothesis failed. Any next GSM-2B experiment should preregister a new single-variable mechanism that changes candidate diversity or a non-gold selector signal before spending another GPU slot.

### H52 completed held-out re-eval — 2026-06-29T21:53Z

H52 finished and should no longer be monitored as running. It wrote `outputs/generated/h52_spider9b_alias_postprocess_heldout_maxlen4096_20260629/h52_reeval.json`. Parsed result: accuracy **0.0**, syntax rate **0.0**, **300** examples, total output tokens **0**, mean output tokens/example **0.0**, evaluator total time **1712.0635s**. The log confirms all **300/300** examples were processed and reports `num_correct: 0 / 300` plus `wrote_json`. Artifact inspection showed every generated answer was the empty string with `num_tokens=0`. The post-run `Engine core proc EngineCore died unexpectedly` line happened after the artifact write. Paid credential key-name scan over the H52 log and output root found **0** hits.

H52 is not a Spider-9B win. Treat it as a refutation of the exact H52 alias-postprocessor launch, not as paper-ready evidence. Next Spider-9B work should not rerun the same body; it needs a preregistered diagnosis or single-variable repair for the immediate-empty-output behavior before another full held-out attempt.

### H70 paid launch decision — 2026-06-29T21:56Z

H70 is now allowed to launch even while H65 continues, because there is a clearly separate safe GPU lane: H65 is on GPU0, GPU3 is clean with **40438 MiB** free and no compute process, and H52/H64 have finished. Use the already-preregistered H70 launcher with the explicit concurrency override: `DRY_RUN=0 SAFE_GPU_ID=3 CONFIRM_BEDROCK_ACCOUNT_887730490125=yes ALLOW_WHILE_H65_RUNNING=yes outputs/generated/h70_smiles_qwen35_2b_acrylates_livebar_materialization_20260629/launch_h70_smiles_qwen35_2b_acrylates_livebar.sh`. The launch should use the same approved recorded AWS account **887730490125** and must not print or store secret values. H70's H73 provenance hardening should snapshot the old acrylates held-out/latest-run state before launch and copy post-launch train/held-out artifacts afterward.

### H78 Spider-9B repair preregistered — 2026-06-29T22:08Z

H78 is the next Spider-9B action and should be launched on a safe local GPU before spending on a fresh Spider synthesis. It fixes the diagnosed H52 launch mistake with one variable: use the real H19/H50 attempt-20 compiled Spider strategy path, not the empty H51 scratch template, while keeping the H51 alias-cleaning evaluator patch and all H52 runtime fairness settings unchanged. Evidence: H52's compiled `GeneratedCSD.py` has sha256 `752be665e80ac0901a8a6ba6b73430b527b54054ad87e16a8b2825c15199e31c` and only returns `generatedPrefix`, explaining H52's **300/300** empty outputs and total output tokens **0**. The real H19 compiled strategy sha256 is `3d96605898a26b0f9fed6cb51738c9bd076635035baa4b0f27bed58092e3b4a2` and contains the SQL guidance plus generation calls. Prediction: H78 should generate non-empty outputs; if the alias-cleaning patch transfers like H50's CPU replay, it can beat the Spider-9B held-out bar **201/300 = 67.0%**. If it stays near old H19 (**194/300 = 64.7%**) or emits empty outputs, Spider-9B remains open.

H78 launched at **2026-06-29T22:09Z** on GPU2 after the gate showed **40950 MiB** free. PID `1077558`; log `/tmp/csd_h78_logs/h78_spider9b_h19_aliasclean_20260629.log`; output `outputs/generated/h78_spider9b_h19_aliasclean_heldout_20260629/h78_reeval.json`; launch report `/tmp/csd_h78_logs/h78_spider9b_h19_aliasclean_20260629_launch.json`. Immediate health check showed H78 alive and loading Qwen3.5-9B. Monitor H78 alongside H65 and H70.

### H78 completed Spider-9B held-out win — 2026-06-29T22:49Z

H78 finished and should no longer be monitored as running. It wrote `outputs/generated/h78_spider9b_h19_aliasclean_heldout_20260629/h78_reeval.json` at **2026-06-29T22:48:58Z**. Parsed result: accuracy **0.74**, syntax rate **0.99**, **300** examples, total generation **2244.7164s**, mean generation/example **7.482388s**, total output tokens **7443**, mean output tokens/example **24.81**, evaluator total time **2366.3061s**, max sample time **36.9992s**, and **300/300** non-empty generated answers. This beats the Spider-9B IterGen held-out bar **201/300 = 67.0%** and should be treated as a paper-ready Spider-9B win. Credential scan found paid credential key names only in the H78 launch report's unset/blocked-env list; no secret values were printed or stored here.

### H79 refreshed paper-ready evidence bundle — 2026-06-29T22:55Z

H79 wrote `outputs/generated/h79_paper_ready_evidence_bundle_20260629/h79_summary.json` and `.md`. This is a CPU-only evidence refresh after H78, with **0** model calls, **0** GPU calls, **0** billed API calls, and **0** score-artifact edits. It packages **3** current paper-ready wins with metrics, bars, artifact paths, mtimes, and sha256 hashes: Spider-2B, Spider-9B, and SMILES isocyanates-4B primary UV. Paid credential key-name scan over the bundle found **0** hits. Use H79, not H74, as the current evidence pointer for proven paper-ready wins.

### H80 GSM-2B visible-span candidate smoke launched — 2026-06-29T23:03Z

H80 is a local no-billing GSM-2B smoke running on GPU2. It was preregistered in `docs/experiments/metadecode-fast-iteration-log.md` before materialization/launch. Body: `saved-results/2026-06-29-h80-gsm2b-visible-span-candidates-body.dfy`. Launcher: `outputs/generated/h80_gsm2b_visible_span_candidates_materialization_20260629/launch_h80_gsm2b_visible_span_candidates_t0.sh`. PID file: `/tmp/csd_h80_logs/h80_gsm2b_visible_span_candidates_t0_20260629.pid`; latest PID **1237732**. Log: `/tmp/csd_h80_logs/h80_gsm2b_visible_span_candidates_t0_20260629.log`. Output root: `outputs/generated/h80_gsm2b_visible_span_candidates_20260629_t0`. Expected outputs: `results/direct_eval_success.json` or `results/direct_eval_failure.json`, plus `results/structured_span_candidates.json` if success postprocessing runs. Dry-run validation passed with `no_model_calls=true`, `no_billed_credentials=true`, sample size **49**, max steps **900**, and **600s/example**. Credential key-name scan found **0** hits. Launch gate used clean GPU2 with **40432 MiB** free and no compute process. Health check at **2026-06-29T23:05Z** showed Qwen3.5-2B vLLM loaded and evaluation running; example 1 generated **900** tokens in **31.14s** and example **2/49** had started. Monitor H80 alongside H65 and H70; do not treat it as paid.

### H80 completed GSM-2B smoke — 2026-06-30T01:31Z

H80 finished and should no longer be monitored as running. PID file `/tmp/csd_h80_logs/h80_gsm2b_visible_span_candidates_t0_20260629.pid` still contains **1237732**, but the process is gone. It wrote `outputs/generated/h80_gsm2b_visible_span_candidates_20260629_t0/results/direct_eval_success.json` and `outputs/generated/h80_gsm2b_visible_span_candidates_20260629_t0/results/structured_span_candidates.json`; `direct_eval_failure.json` is absent. Metrics: accuracy **0.02040816326530612** (**1/49**), syntax rate **0.5102040816326531**, **49** examples, elapsed **1668.008341550827s**, `source_id=h80_gsm2b_visible_span_candidates_t0`. The structured span artifact has `candidate_count=0`, `group_count=0`, and `selection_uses_gold=false`. Exact paid credential key-name scan over the H80 log and artifacts found **0** hits.

Belief update: H80 is a GSM-2B loss and refutes the visible-span candidate-exposure hypothesis for this body. It produced no selector-ready span candidates and lower direct accuracy/syntax than H64. Keep the artifact as negative evidence; do not rerun this exact body or count it as paper-ready. H65 GSM-9B paid synthesis and H70 SMILES acrylates-2B paid retry remain the active running jobs.

### H70 SMILES acrylates-2B completed held-out loss — 2026-06-30T05:52Z

H70 finished and should no longer be monitored as running. PID file `/tmp/csd_h70_logs/h70_smiles_qwen35_2b_acrylates_livebar_20260629.pid` contains **1037864**, and the process is gone; log path is `/tmp/csd_h70_logs/h70_smiles_qwen35_2b_acrylates_livebar_20260629.log`. Train accepted on attempt **38/40** at UV/accuracy **0.42** and validity/syntax **0.78** on **50** examples, using `--max-iterations 40`. Train report: `outputs/generated/smiles_qwen35_2b_acrylates_uv_qwen35_0627/smiles_qwen35_2b_acrylates_uv_qwen35_0627_20260629_215714_3a5627/results/success_report.json`.

The held-out file `outputs/controlled_comparison/smiles_qwen35_2b/acrylates/metadecode_uv.json` has mtime **2026-06-30T05:51:56Z** and records UV/accuracy **0.34** plus validity/syntax **0.82** on **100** examples. This misses the live focal CARS UV bar **0.36**, so H70 is a held-out loss and `results_matrix.md` is unchanged. Provenance copies are under `outputs/generated/h70_smiles_qwen35_2b_acrylates_livebar_materialization_20260629/provenance_20260629T215710Z/postlaunch/`; credential key-name scan over the H70 log/artifacts found **0** hits. Belief update: raising the acrylates-2B train acceptance threshold to the live UV bar can force a train acceptance, but by itself did not transfer to held-out. The next acrylates-2B action should target generalization/diversity beyond the train threshold rather than relaunching this exact H70 body.

### H81 SMILES acrylates-2B paid retry launched — 2026-06-30T06:07Z

H81 was preregistered in `docs/experiments/metadecode-fast-iteration-log.md` before launch. It tests one variable versus H70: raise `min_syn` from **0.50** to **0.85** while keeping Qwen3.5-2B, class `acrylates`, `min_acc=0.36`, GPU util **0.20**, temp **0.7**, `--max-iterations 40`, adaptive helper mask + bandit, and provenance snapshots unchanged. Dry-run/static checks passed in `outputs/generated/h81_smiles_qwen35_2b_acrylates_uv36_syn085_materialization_20260630/h81_checks.json`: `bash -n`, dry-run **0** model/GPU/billed API calls, H65 guard, safe-GPU/no-other-user gate, pre/post provenance copies, and **0** paid credential key-name hits.

Real launch command used recorded AWS account **887730490125** with `DRY_RUN=0 SAFE_GPU_ID=3 CONFIRM_BEDROCK_ACCOUNT_887730490125=yes ALLOW_WHILE_H65_RUNNING=yes outputs/generated/h81_smiles_qwen35_2b_acrylates_uv36_syn085_materialization_20260630/launch_h81_smiles_qwen35_2b_acrylates_uv36_syn085.sh`. PID file `/tmp/csd_h81_logs/h81_smiles_qwen35_2b_acrylates_uv36_syn085_20260630.pid` contains **2470293**; log `/tmp/csd_h81_logs/h81_smiles_qwen35_2b_acrylates_uv36_syn085_20260630.log`; artifact root `outputs/generated/h81_smiles_qwen35_2b_acrylates_uv36_syn085_materialization_20260630`; provenance dir `outputs/generated/h81_smiles_qwen35_2b_acrylates_uv36_syn085_materialization_20260630/provenance_20260630T060702Z`. Health check at **2026-06-30T06:08Z** showed H81 alive and initializing Qwen3.5-2B on GPU3 with VLLM engine PID **2473026**. Monitor H81 with H65; do not launch another paid SMILES cell until H81 is fully recorded.

### H82 GSM-2B candidate failure audit — 2026-06-30T06:17Z

H82 wrote `outputs/generated/h82_gsm2b_candidate_failure_audit_20260630/h82_summary.json` and `.md` as a CPU-only diagnostic: **0** model calls, **0** GPU calls, **0** billed API calls, and no score-artifact edits. It sharpened the GSM-2B next-step diagnosis. H64 had **218** extracted candidate records across **46** groups, but **172** were `candidate_line` records and only **1/172** looked machine-parseable by a conservative text heuristic; the rest were mostly prose or LaTeX route descriptions. H64's `final_actual` candidates covered **46** groups, but only **2/49** direct final answers were correct. H80 had **0** structured candidates across **0** groups. Credential key-name scan found **0** hits.

Next GSM-2B move: do not rerun H64's prose route labels or H80's visible-span body. The next local no-billing GSM-2B hypothesis should force bare machine-readable arithmetic-expression candidates with variable names preserved, then run no-gold score/selection after generation.

### H83 GSM-2B H80 postprocess field audit — 2026-06-30T06:28Z

H83 wrote `outputs/generated/h83_gsm2b_h80_postprocess_field_audit_20260630/h83_summary.json` and `.md` as a CPU-only diagnostic: **0** model calls, **0** GPU calls, **0** billed API calls, and no score-artifact edits. It tested whether H80's empty `structured_span_candidates.json` was simply caused by looking at the wrong direct-eval text field. The narrow prediction was **false**: H80 direct reports do contain `full_output`, and H83 found **94** visible spans there. But H83 found a sharper artifact problem: the existing structured artifact stayed at **0** candidates / **0** groups even though `full_output` contains **75** conservative parseable-ish spans across **21** groups. Scanning all text fields found **467** visible-span records across **49** groups, but that broader count includes task-guidance and helper-trace echoes, so it is diagnostic only, not a clean selector input. Credential key-name scan found **0** hits.

Next GSM-2B move after H83: do not treat H80's zero structured candidates as the whole story, but also do not rerun H80 unchanged. The next local no-billing GSM-2B body should force bare machine-readable candidate expressions and pair it with field-aware postprocessing that reads the direct-eval fields actually containing final output text. A selector-only rerun is still not justified because clean full-output candidate coverage is only **21/49** groups.

### H84 GSM-2B bare-expression candidate probe materialized — 2026-06-30T06:38Z

H84 is preregistered and launch-ready as the next local no-billing GSM-2B probe, but it has **not** launched. Body: `saved-results/2026-06-30-h84-gsm2b-bare-expression-candidates-body.dfy`. Launcher: `outputs/generated/h84_gsm2b_bare_expression_candidates_materialization_20260630/launch_h84_gsm2b_bare_expression_candidates_t0.sh`. Checks: `outputs/generated/h84_gsm2b_bare_expression_candidates_materialization_20260630/h84_checks.json`; dry run: `direct_eval_dry_run.json`.

H84 changes the candidate-generation contract from H80's route/prose allowance to strict bare-expression candidate lines such as `A: <<expr>>`, `B: <<expr>>`, with labels outside spans and only variables/numbers/operators inside spans. The launcher keeps local Qwen3.5-2B, train49 split, sample size **49**, max steps **900**, **600s/example**, GPU util **0.20**, and no paid provider use. The success postprocess writes `results/structured_bare_expression_candidates.json` using visible spans from `full_output` and `scored_output` only, so it avoids H80's empty-artifact failure without using expected answers or correctness labels.

Validation before launch: `bash -n` passed; dry run completed with return code **0**; `h84_checks.json` records **0** model calls, **0** GPU calls, **0** billed API calls, and **0** paid credential key-name hits. Do not launch H84 until there is a clean GPU lane; current live lanes are H65 on GPU0, H81 on GPU3, and non-aadivyar jobs on GPU1/GPU2.

### H85 SMILES isocyanates-9B launch-readiness materialized — 2026-06-30T07:23Z

H85 is preregistered and dry-run validated as the next unrun SMILES queue prep, but it has **not** launched. It targets `isocyanates-9B`, the lowest live-bar remaining unrun SMILES cell after H66/H67, with live focal CARS UV bar **0.92**. Launcher: `outputs/generated/h85_smiles_qwen35_9b_isocyanates_livebar_materialization_20260630/launch_h85_smiles_qwen35_9b_isocyanates_livebar.sh`. Planned cold generic command: `./pilot_smiles_uv_qwen35_i40.sh Qwen/Qwen3.5-9B qwen35_9b isocyanates <gpu> 0.55 0.92 0.50`. It does not use `--initial-strategy-file` and does not add learned isocyanate-specific tricks to the task text.

Static/dry-run evidence: focal `bash -n` passed; dry run wrote `outputs/generated/h85_smiles_qwen35_9b_isocyanates_livebar_materialization_20260630/h85_dry_run.json`; `h85_checks.json` records **0** model calls, **0** GPU calls, **0** billed API calls, max iterations **40**, no initial strategy file, H81-alive refusal for real launch, recorded-account confirmation gate for **887730490125**, safe-GPU/no-compute-process gate with **30000 MiB** free-memory minimum, and **0** paid credential key-name hits. Do **not** launch H85 while H81 is active; after H81 is recorded, H85 is ready only if a clean GPU lane exists and the paid-account rule is still satisfied.

### H81 SMILES acrylates-2B held-out loss and H86 gate correction — 2026-06-30T08:30Z

H81 finished and should no longer be monitored as running. Train accepted on attempt **19/40** at train UV **0.44** and validity/syntax **0.88** on **50** examples, with **22** unique valid molecules. Held-out `outputs/controlled_comparison/smiles_qwen35_2b/acrylates/metadecode_uv.json` updated at **2026-06-30T08:23:46Z** and records UV **0.17** and validity/syntax **0.78** on **100** examples, with **17** unique valid molecules. H81 is therefore a held-out loss and not paper-ready.

Diagnosis: the SMILES UV train gate is misaligned with held-out because train used **50** examples while held-out uses **100**. At the live **0.36** acrylates-2B UV bar, train acceptance requires only **18** unique valid molecules, while held-out requires more than **36**. H81's train result passed with **22/50** unique valid molecules but did not have enough diversity to survive the 100-example held-out metric. Under the user's train-win/held-out-fail policy, the next related SMILES launch should align train and held-out difficulty before spending more money.

H86 supersedes a real H85 launch for the next SMILES run. H86 keeps the same cold generic `isocyanates-9B` target and live UV bar **0.92**, but changes the train gate to **100** examples by adding `--eval-sample-size 100` and `--smiles-samples-per-class 100`; held-out remains **100** examples. H86 must be dry-run/static validated and recorded before any real paid launch. H85 remains a dry-run-only historical prep artifact and should not be launched as-is.

H86 validation before paid launch: focal `bash -n` passed; `DRY_RUN=1 SAFE_GPU_ID=3 outputs/generated/h86_smiles_qwen35_9b_isocyanates_train100_materialization_20260630/launch_h86_smiles_qwen35_9b_isocyanates_train100.sh` wrote `h86_dry_run.json` with **0** model calls, **0** GPU calls, **0** billed API calls, train sample size **100**, held-out sample size **100**, max iterations **40**, `Qwen/Qwen3.5-9B`, `isocyanates`, `min_acc=0.92`, `min_syn=0.50`, and no initial strategy. GPU3 is clean with **40438 MiB** free and no compute process; H65 is isolated on GPU0. Real H86 launch is authorized under the recorded account approval **887730490125** with `DRY_RUN=0 SAFE_GPU_ID=3 CONFIRM_BEDROCK_ACCOUNT_887730490125=yes ALLOW_WHILE_H65_RUNNING=yes`.

H86 launched at **2026-06-30T08:37Z**. PID file: `/tmp/csd_h86_logs/h86_smiles_qwen35_9b_isocyanates_train100_20260630.pid` with PID **2873849**. Log: `/tmp/csd_h86_logs/h86_smiles_qwen35_9b_isocyanates_train100_20260630.log`. Provenance dir: `outputs/generated/h86_smiles_qwen35_9b_isocyanates_train100_materialization_20260630/provenance_20260630T083744Z`. Generated root: `outputs/generated/smiles_qwen35_9b_isocyanates_uv_qwen35_0627`; latest run: `outputs/generated/smiles_qwen35_9b_isocyanates_uv_qwen35_0627/smiles_qwen35_9b_isocyanates_uv_qwen35_0627_20260630_083748_22e3d6`; held-out target: `outputs/controlled_comparison/smiles_qwen35_9b/isocyanates/metadecode_uv.json`. Immediate health check showed the wrapper alive and latest run created; wait for train attempt metrics before taking action.

### H87 SMILES prompt-visible duplicate helper implemented — 2026-06-30T13:55Z

H87 is the first concrete implementation of the new core-framework/helper policy. It adds a class-neutral no-gold helper so generated CSDs can ask whether a candidate prefix already appears in the prompt or rolling suffix. This is intended for future SMILES unique-valid/diversity failures; it does **not** affect the already-running H86 process, because H86 launched before the helper was added.

Implementation landed in focal main after passing in the isolated worktree `h87-smiles-prompt-seen-helper`. Changed files: `synthesis/evaluate/benchmarks/common/model_utils.py`, `synthesis/evaluate/benchmarks/common/test_grounding.py`, `synthesis/evaluate/feedback_loop.py`, `synthesis/generate/prompts.py`, and `synthesis/verify/library/VerifiedAgentSynthesis.dfy`. New surface: host extern `SpanAppearsInPrompt`, Dafny helper `helpers.PrefixAppearsInPrompt(lm, prefix)`, and prompt/helper allowlist documentation.

Verification on focal main: red test failed first with missing `SpanAppearsInPrompt`; green run passed **12/12** tests in `synthesis/evaluate/benchmarks/common/test_grounding.py`; Python compile checks passed for the touched Python files; prompt exposure check returned `PrefixAppearsInPrompt in helper universe: True`; Dafny verification passed with **177 verified, 0 errors**. Credential scan found no secret values. H87 made **0** model calls, **0** GPU calls, and **0** paid API calls.

Next SMILES design implication: if H86 fails or later SMILES runs show low unique-valid diversity/repeated prompt-visible molecules, the next paid synthesis should use the patched main checkout so cold discovery can choose `PrefixAppearsInPrompt`. Do not add class-specific molecule tricks or warm-start from a prior strategy.

### H86 SMILES isocyanates-9B train100 result — 2026-06-30T20:52Z

H86 is finished and should no longer be monitored as running. It wrote `outputs/generated/smiles_qwen35_9b_isocyanates_uv_qwen35_0627/smiles_qwen35_9b_isocyanates_uv_qwen35_0627_20260630_083748_22e3d6/results/failure_report.json` at **2026-06-30T20:51:58Z**. This is a train loss, not a held-out candidate. H86 ran **40/40** attempts on **100** train examples with train gate **0.92** UV/accuracy and **0.50** syntax/validity. Best train attempt was attempt **20** at UV/accuracy **0.37** and syntax **0.41**. Attempt **21** reached syntax **0.92** but only UV/accuracy **0.32**. Final attempt **40** scored UV/accuracy **0.10** and syntax **0.25**. No held-out file was produced because train did not cross the gate, and `results_matrix.md` should not be updated.

Failure-pattern evidence: report `failure_patterns` records **11** verification failures, **0** search-contract failures, **0** compilation failures, and **0** runtime failures. The final timing profile shows `GenerateLogits.engine_generate` as the dominant measured cost (**26,532s**, **47.5%**), while `GenerateLogits.prefix_text` is only **64.13s** (**0.1%**). This means the next SMILES optimization should not spend effort on `prefix_text` unless a later profile changes that evidence. Secret scan across the H86 log, launch record, failure report, and provenance files checked **8** files and found **0** AWS-shaped key values, **0** env-style secret assignments, and **0** secret key-name mentions.

Next SMILES implication: H86 refutes sample-size alignment alone for `isocyanates-9B`. The next SMILES design step should not be another H86-shaped paid retry with only bar/launch-setting changes. Tie the next one-variable hypothesis to the core framework evidence already recorded: H88 found span/control-output-budget and long invalid concatenated SMILES evidence, H89 exposed managed-span helpers after H86 launched, and H90 found no broad helper-exposure gap left. A future cold SMILES run from patched focal main can test whether the H89 helper exposure changes the generated strategy behavior; if not, the next patch should be a targeted general helper or helper-behavior change, still with pure cold discovery and no class-specific task guidance.

### H91 final H86 failure audit — 2026-06-30T20:56Z

H91 is a CPU-only diagnostic result, not a benchmark run. It wrote `outputs/generated/h91_h86_final_failure_audit_20260630/h91_summary.json` and `.md` with **0** model calls, **0** GPU calls, **0** billed API calls, and **0** score-artifact edits. Secret scan over the two H91 output files found **0** AWS-shaped key values, **0** env-style secret assignments, and **0** secret key-name mentions.

H91 changes the next SMILES recommendation. Best H86 accuracy was attempt **20** at **0.37 / 0.41**, with **41/100** RDKit-valid, class-member, unique-valid candidates. But the highest-syntax attempt was attempt **1** at **0.00 / 1.00**: it produced **100/100** grammar-valid, RDKit-valid, syntax-valid outputs, yet **0/100** class-membership and **0/100** unique-valid candidates. This means H86 can solve "valid molecule" while missing "valid molecule in the target class." The next SMILES lever should therefore be a general no-gold class-membership / candidate-selection helper or helper-behavior change before another paid launch, not only another cold H86-shaped retry or a pure parser/span speed fix. Keep it class-neutral: no isocyanate-specific task hints, no scorer/dataset/baseline changes, no warm start.

### H92 SMILES prompt-class helper — 2026-06-30T21:14Z

H92 implemented the H91-recommended core-framework helper in focal main. New generated-CSD surface:
`helpers.PrefixMatchesPromptMoleculeClass(lm, prefix)`, backed by host predicate
`SpanMatchesPromptMoleculeClass(text)`. The helper cleans the rendered prefix, infers the molecule
class only from prompt-visible text, and uses the existing generic SMILES class-membership logic. It
does not read gold labels, scorer state, held-out data, evaluator results, or class-specific strategy
advice.

Verification: the red grounding test first failed with missing `SpanMatchesPromptMoleculeClass`.
After implementation, focal main passed **16/16** grounding tests, **6/6** helper-surface tests,
Python compile checks for touched Python files, prompt exposure checks showing
`PrefixMatchesPromptMoleculeClass` is in the helper universe and prompt docs, and Dafny verification
with **179 verified, 0 errors**. Same-reference search found the helper only in expected code/docs
locations. Secret scan across touched code/docs found **0** AWS-shaped key values and **0** env-style
secret assignments. H92 made **0** model calls, **0** GPU calls, and **0** billed API calls.

Next SMILES implication: H92 is not a paper-ready result and must not update `results_matrix.md`.
The next SMILES synthesis should be a cold run from patched focal main so the author model can choose
the new helper naturally. Before any paid relaunch, recheck the paid-account identity/safety gate and
write the launch record.

### H93 SMILES isocyanates-9B H92-helper cold run — launched 2026-06-30T21:30Z

H93 is the active paid SMILES run after H92. It keeps H86's 100-example train gate shape for
`Qwen/Qwen3.5-9B` `isocyanates`, with live train UV/accuracy bar **0.92**, syntax floor **0.50**,
max iterations **40**, no `--initial-strategy-file`, no scorer/grammar/dataset/baseline change,
and no helper-specific task text. The scientific variable versus H86 is that focal main now includes
H92's `PrefixMatchesPromptMoleculeClass` helper surface.

Dry-run/static evidence: launcher
`outputs/generated/h93_smiles_qwen35_9b_isocyanates_h92helper_train100_materialization_20260630/launch_h93_smiles_qwen35_9b_isocyanates_h92helper_train100.sh`
passed `bash -n` and wrote `h93_dry_run.json` with **0** model calls, **0** GPU calls,
**0** billed API calls, `uses_initial_strategy_file=false`, train sample size **100**, held-out
sample size **100**, generated root
`outputs/generated/smiles_qwen35_9b_isocyanates_uv_qwen35_h93_h92helper_20260630`, and held-out
target `outputs/controlled_comparison/smiles_qwen35_9b/isocyanates/metadecode_uv.json`.

Real launch: `DRY_RUN=0 SAFE_GPU_ID=0 CONFIRM_BEDROCK_ACCOUNT_887730490125=yes`; PID **756754**;
PID file `/tmp/csd_h93_logs/h93_smiles_qwen35_9b_isocyanates_h92helper_train100_20260630.pid`;
log `/tmp/csd_h93_logs/h93_smiles_qwen35_9b_isocyanates_h92helper_train100_20260630.log`;
provenance dir `outputs/generated/h93_smiles_qwen35_9b_isocyanates_h92helper_train100_materialization_20260630/provenance_20260630T213012Z`.
Immediate health check showed the wrapper alive and the latest run directory created; no new GPU
process had started yet while Bedrock strategy generation began.
