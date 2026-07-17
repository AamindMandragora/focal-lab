# Dynamic CSD Gen win strategy across SMILES, GSM, and Spider

**Date:** 2026-06-30  
**Source of truth:** focal checkout at `/home/aadivyar/csd-generation`  
**Purpose:** make the campaign logic explicit enough that you can critique it and redirect it before we spend more time or paid Bedrock calls.  
**Scope:** the current Qwen3.5 Dynamic CSD Gen campaign: SMILES, GSM, Spider, regression checks, and ablations.  

## One-page summary

The campaign should be run as a staged win-building loop, not as a pile of opportunistic experiments.

Current order:

1. Finish SMILES paper-ready wins.
2. Then finish GSM wins.
3. Then finish Spider wins.
4. Then rerun prior wins to check for regressions.
5. Then run ablations once the win matrix is stable.

Within each stage, the core loop is:

1. Pick the next cell from the stage policy.
2. Read the live baseline bar and prior artifacts.
3. Write a hypothesis before changing code or launching.
4. Classify the next lever by how directly it changes the CSD's ability.
5. Prefer a core framework/helper change when the failure is not just measurement, provenance, or launch safety.
6. Change one thing.
7. Run the smallest fair test that can answer the question.
8. If training wins, immediately run held-out.
9. If held-out fails, diagnose split difficulty and bar margin before launching another search.
10. Only promote held-out wins to the paper matrix.

The biggest rule is that a training win is only a search signal. It is not a paper-ready result. The paper result is the held-out comparison against the live baseline artifact.

The second biggest rule, added from the 2026-06-30 user correction and strengthened after H86/H91, is that run settings are not enough. If a cell keeps missing, the next serious hypothesis should usually change what the generated CSD can express, test, repair, or select. Bar changes, sample-size alignment, and provenance fixes are still valid when they explain the failure, but they are experiment management. They should not become the default way to search for wins.

The sharper version of that rule, added after your latest correction, is: attack the core framework earlier. A higher bar, longer timeout, cleaner launch, bigger sample, or better artifact copy can make a result more trustworthy, but it rarely gives the CSD a new ability. New wins are more likely to come from changing the core framework: helper behavior, helper availability, feedback signals, candidate representation, repair operations, and no-gold selection operations.

In plain terms: do not just squeeze the same framework harder. If the CSD cannot build useful candidates, cannot tell duplicates apart, cannot repair invalid outputs, or cannot choose among candidates without gold labels, the next serious change should attack that directly. That usually means editing a helper, exposing an existing helper better, or adding a new general helper.

This is now a stronger default, not just another option. For capability-shaped failures, a core framework change should be treated as the main path to a win. Around-framework changes such as higher bars, bigger samples, longer timeouts, and cleaner launch wrappers are still useful, but they mostly make evidence cleaner. They usually do not give the generated CSD a new operation. New helper behavior and new general helpers are riskier, but they are also the changes most likely to move a stuck cell because they change what the generated strategy can actually do.

The correction changes the default question after a failed run:

- Weak question: "What threshold, timeout, sample size, or launch setting should I try next?"
- Better question: "What operation did the generated CSD need at the failure point, and does the framework currently give it that operation?"

This does not mean every failure gets a large rewrite. It means the first explanation should be about capability. If the evidence shows the capability is already present and the problem is measurement, then change the measurement. If the evidence shows the CSD is missing an operation, change the helper surface or helper implementation before spending another paid run on the same setup.

The practical default is now:

1. Look at the failed artifacts.
2. Identify the first handoff where the useful signal was lost.
3. Ask what operation would have preserved or recovered that signal.
4. Check whether that operation already exists as a helper, feedback signal, or candidate artifact.
5. If it exists but was not used, improve helper surfacing or feedback.
6. If it exists but returns the wrong shape, repair the helper behavior.
7. If it does not exist, add the smallest fair framework operation before another paid retry.

The burden of proof is now on non-core changes. If the next proposed move is only a bar, timeout, sample size, launch wrapper, or artifact-copy change, the ledger should say why a helper/framework change is not the right response to the observed failure.

What counts as a core framework change:

- Editing helper behavior.
- Adding a new general helper.
- Changing how helpers are shown to the author model.
- Changing feedback so the author model can see which operation failed.
- Changing candidate representation so useful partial work is not lost.
- Changing repair, filtering, duplicate-checking, or no-gold selection operations.

What does not count as a core framework change:

- Raising or lowering a score bar.
- Changing train sample size, held-out sample size, or timeout by itself.
- Relaunching the same setup cold and hoping search variance finds a better strategy.
- Moving artifacts around without changing what the generated CSD can do.
- Adding class-specific task hints, changing the scorer, changing the grammar, changing the dataset, or changing the baseline. Those are still disallowed.

This is the main operational change: for hard cells, assume the largest gains will come from changing the framework's fair operations, not from changing the conditions around the same framework. A cold relaunch with the same helper surface is justified only when the failure audit says the CSD already had the operation it needed and the miss was caused by measurement, provenance, runtime, or random search variance.

### User correction incorporated on 2026-07-01

Your correction is now part of the campaign policy:

- I had been spending too much effort on things around the framework: bars, sample sizes, timeouts, launch wrappers, provenance, and monitoring.
- Those changes make the evidence cleaner, but they usually do not change what the generated CSD can do.
- The next hard-cell hypotheses should attack the core of the framework earlier: helper implementation, helper visibility, feedback, candidate representation, repair operations, and no-gold candidate selection.
- This is higher risk than changing run settings, but it is also more likely to produce a real step change because it gives generated strategies new fair operations.

The working rule is:

1. If the failure is about trust, recording, billing safety, or train/held-out mismatch, fix that wrapper issue first.
2. If the failure is about runtime, profile the active path and optimize the measured slow helper.
3. If the failure is about what the generated CSD can build, check, repair, compare, or select, make the next serious change a framework/helper change.
4. Do not spend another paid retry on the same helper surface unless the ledger explains why the current framework already has the needed operation.

### Latest correction: do not confuse cold discovery with a frozen framework

Pure cold discovery means the generated strategy starts from scratch. It does not mean the framework must stay frozen while we watch it fail in the same way.

The corrected policy is:

1. Keep every synthesis run cold.
   - No seeded strategy.
   - No `--initial-strategy-file` for synthesis.
   - No class-specific task guidance.

2. Improve the general framework between cold runs when the evidence points there.
   - Helper implementations are allowed to change.
   - New general helpers are allowed.
   - Feedback and candidate artifacts are allowed to change.
   - The change must be task-general, no-gold, and available to future cold runs.

3. Treat repeated wrapper-only changes as a warning sign.
   - A better train bar, bigger sample, longer timeout, cleaner launch, or safer artifact copy can make evidence more trustworthy.
   - Those changes do not usually give the generated CSD a new ability.
   - If the failure trace shows the CSD lacked an operation, another wrapper-only retry is the fallback, not the default.

The short rule: cold strategy, improving framework. The strategy should not inherit old answers, but the framework should learn from failures by adding fair operations.

### Hypothesis source rule

The next hypothesis should come from the first broken operation in the run, not from the final score.

#### Inputs

- The generated CSD code.
- The helper calls it used.
- The helpers it could see but did not use.
- The candidate strings or structured outputs.
- The first metric bucket that failed: validity, syntax, uniqueness, class membership, exact answer, schema correctness, output packaging, or runtime.
- The feedback sent back to the author model.

#### Outputs

The ledger entry should name:

- The first broken operation.
- Whether the operation is missing, hidden, slow, wrong-shaped, or hard to verify.
- The smallest fair framework change that attacks that operation.
- The smallest local effect expected before a paper-ready win.

#### Algorithm

1. Write the intended path.
   - SMILES: prompt -> candidate strings -> valid molecules -> canonical unique molecules -> prompt-class members -> selected final answer.
   - GSM: question -> clean expressions -> evaluated candidates -> scorer-ready final field.
   - Spider: question/schema -> SQL candidates -> parseable SQL -> valid schema references -> official grader input.

2. Write the actual path from artifacts.
   - Use generated code, helper calls, candidate artifacts, logs, and score buckets.

3. Find the first broken handoff.
   - Stop where useful information first became missing, malformed, repeated, invisible, or unusable.

4. Ask what operation would have fixed that handoff.
   - Examples: canonicalize candidates, check prompt-derived molecule class, repair invalid strings, extract clean expressions, preserve scorer-ready output fields, check SQL schema references, choose among no-gold candidates.

5. Check the helper surface.
   - If the helper exists and was used, inspect the implementation and return shape.
   - If the helper exists but was not used, inspect helper visibility, docs, helper-selection rules, and feedback.
   - If no helper exists, add the smallest general helper.

6. Test the local effect first.
   - The first test should prove the helper changes the broken operation.
   - The paid cold run comes after the helper is tested and visible.

## Updated stance: use experiments to improve the framework

Your correction changes how I should read failures.

The campaign is not just trying to get lucky with a generated strategy. It is using each failed run to identify what the current framework does not let a generated CSD do. Once that missing operation is visible, the next serious move should usually be a targeted framework/helper change, not another run with only a new bar, timeout, sample size, or launch wrapper.

### Inputs

For every failed or weak run, inspect:

- The generated strategy code.
- The helper calls it used.
- The helpers available but not used.
- The candidate strings or structured outputs it produced.
- The metric bucket that failed: accuracy, syntax, validity, uniqueness, class membership, schema correctness, exact answer extraction, or runtime.
- The feedback sent back to the author model.

### Outputs

The next hypothesis should produce one of these outputs:

- A helper implementation fix.
- A clearer helper surface so the author model can discover and use an existing helper.
- A new general helper that supplies a missing operation.
- A candidate-contract or feedback-loop change that makes an existing signal usable.
- A written reason why the failure was only measurement/provenance/runtime, not missing CSD ability.

### Algorithm

1. Treat the failed run as a trace through the framework.
   - Do not start from "accuracy was low."
   - Start from the generated code, helper calls, candidates, score buckets, and feedback.

2. Find the first operation the CSD needed but could not reliably do.
   - Examples: build a candidate pool, clean candidate text, reject duplicates, check prompt-class membership, repair invalid candidates, select among no-gold candidates, preserve scorer-ready fields, or check schema references.

3. Decide whether the missing operation is already in the framework.
   - If it exists and was used, inspect the helper implementation and return shape.
   - If it exists and was not used, inspect prompt docs, helper naming, helper-selection rules, and feedback.
   - If it exists but is too slow, inspect the algorithm before adding new behavior.
   - If it does not exist, add the smallest fair helper that supplies it.

4. Prefer the core framework lever when the failure is capability-shaped.
   - A launch-setting change is appropriate when the result cannot be trusted.
   - A sample-size or bar change is appropriate when train and held-out were not comparable.
   - A speed change is appropriate when runtime prevents enough attempts.
   - A helper/framework change is appropriate when the CSD could not build, check, repair, compare, or select the right candidate.

5. Test the framework change before another paid launch.
   - Helper behavior change: write a direct test first.
   - Helper visibility change: check the generated prompt/feedback surface.
   - Candidate-contract change: check the artifact shape on a small probe.
   - Speed change: profile before and after.

6. Relaunch synthesis cold only after the framework change is visible and recorded.
   - No warm starts.
   - No class-specific tricks in task text.
   - No scorer, grammar, dataset, split, or baseline changes.

This is the key principle: a failed hard cell should usually become a framework-improvement question before it becomes another paid-search question.

### Failure pattern -> preferred response

| Observed failure pattern | First thing to inspect | Preferred next lever |
|---|---|---|
| Valid outputs but wrong class | Prompt-derived class checks and candidate-selection helpers | Repair/expose/add class-membership helper or no-gold selector |
| Many valid outputs but few unique candidates | Canonicalization and duplicate filtering | Add or repair canonical uniqueness helper |
| Invalid outputs dominate | Validity checks, repair operations, parser preconditions | Add or repair safe validity/repair helper |
| Correct-looking answer buried in prose | Candidate extraction contract and final-field writer | Add or repair clean candidate/expression extraction |
| Empty or malformed final output | Output packaging helper and feedback text | Repair scorer-ready field contract |
| Syntax is high but task accuracy is low | Semantic no-gold checks, candidate diversity, and selector signals | Add helper that tests the missing semantic property without gold labels |
| Runtime blocks iteration | The measured hot helper, not all helpers | Optimize the measured helper only |
| Train wins but held-out fails | Split/sample/bar comparability | Align difficulty first, then raise train bar if the drop is real |
| Artifacts are missing or stale | Provenance paths and launch wrapper | Fix recording before interpreting the score |

This table is a guardrail against overusing around-framework changes. The default is not "change a knob." The default is "find the missing operation."

## Core-first default

This is the operating policy change from your latest feedback.

### Main correction

The campaign should treat failed runs as evidence about the framework, not only as evidence about a bad strategy.

If a run fails because the generated CSD cannot build, check, repair, compare, or select the right kind of candidate, the next hypothesis should change the framework surface. That means helper behavior, helper visibility, feedback, candidate artifacts, or a new general helper. It should not default to another cold relaunch with a new threshold, timeout, or sample size.

The key distinction is:

- Around-framework changes make the same CSD process easier to run or easier to trust.
- Core-framework changes change what the generated CSD can actually do.

Around-framework changes are still necessary for fair measurement. H81's 50-example train gate versus 100-example held-out metric is a real example where sample-size alignment mattered. But once the measurement is fair, a hard cell should usually move through core-framework hypotheses.

### How to find the core hypothesis

#### Inputs

For a failed run, use these artifacts:

- Generated strategy code.
- Helper calls used by that strategy.
- Helpers available to the author model but not used.
- Candidate outputs and any candidate-pool artifacts.
- Score buckets such as syntax, validity, uniqueness, class membership, exact answer, or SQL grader result.
- Verifier failures and precondition failures.
- Feedback text sent back to the author model.

#### Outputs

The next hypothesis should name:

- The first step where the intended data flow broke.
- The operation missing at that step.
- Whether the operation is absent, hidden, too slow, wrong-shaped, or hard to verify.
- The smallest fair framework change that would supply that operation.
- A local test that can pass or fail before the next paid run.

#### Algorithm

1. Write the intended data flow for the task.
   - SMILES: prompt -> candidate strings -> valid molecules -> canonical unique molecules -> prompt-class members -> selected final answer.
   - GSM: question -> clean expressions -> evaluated candidates -> scorer-ready final field.
   - Spider: question plus schema -> SQL candidates -> parseable SQL -> valid schema references -> official grader input.

2. Write the actual data flow from the artifacts.
   - Use generated code, candidates, logs, score buckets, and feedback.
   - Do not infer this from the final score alone.

3. Find the first bad handoff.
   - A handoff is where one step passes data to the next step.
   - Examples: generated text to candidate extractor, candidate to parser, parser-valid molecule to class checker, SQL text to schema checker.
   - Stop at the earliest point where useful information became missing, malformed, repeated, unverifiable, or invisible to the next step.

4. Name the missing operation as a verb phrase.
   - Good: "canonicalize candidates before duplicate filtering."
   - Good: "check whether this candidate matches the molecule class named in the prompt."
   - Good: "repair a complete but wrong-class SMILES candidate without violating parser preconditions."
   - Weak: "improve accuracy."

5. Check whether the framework already gives the CSD that operation.
   - If the helper exists and was used, inspect its behavior and return shape.
   - If the helper exists but was not used, inspect helper visibility, prompt docs, feedback, and helper-selection rules.
   - If the helper exists but generated code cannot verify with it, fix the contract or expose a safer wrapper.
   - If no helper exists, add the smallest general helper that supplies the operation.

6. Ask the "human with one tool" question.
   - If a human can look at the artifacts and say, "I could fix this if the CSD had one fair operation," that operation is the next helper hypothesis.
   - Example: "I can see the outputs are valid molecules but all wrong-class; the CSD needs a prompt-derived class-membership helper."
   - Example: "I can see many strings are duplicates; the CSD needs canonical duplicate filtering."
   - Example: "I can see the arithmetic answer is buried in prose; the CSD needs clean expression extraction."

7. Prefer a framework change over another run-setting change when the missing operation is real.
   - First fix an existing helper if it is wrong, slow, or hard to use.
   - Then improve helper surfacing or feedback if the helper exists but was ignored.
   - Then add a new general helper if the operation is absent.
   - Relaunch cold only after the helper change is tested and visible.

8. Keep the helper fair.
   - Allowed: task-general checks, parser state, schema references, molecule validity, prompt-derived class labels, canonical forms, no-gold candidate selection.
   - Not allowed: answer-specific tricks, scorer changes, grammar changes, split changes, baseline changes, warm starts, or class-specific strategy advice in the task text.

9. Predict a local effect before predicting a win.
   - A class helper should raise class-member candidates before it is expected to raise held-out UV.
   - A duplicate helper should raise unique-valid candidates before it is expected to raise final accuracy.
   - A clean-expression helper should increase structured expression candidates before it is expected to raise GSM held-out accuracy.

This is the hypothesis principle I should use from now on: explain the failure by finding the first broken operation in the CSD data flow, then change the fair framework surface that controls that operation.

### Inputs

For every failed cell, read:

- The generated CSD code.
- The helper calls it used.
- The helper calls that were available but not used.
- The candidates it produced.
- The score buckets that failed.
- The feedback message sent back to the author model.

### Outputs

The next hypothesis should name one of these:

- A helper implementation to repair.
- An existing helper to expose more clearly.
- A new general helper to add.
- A feedback or candidate-contract change that lets the CSD use an existing signal.
- A reason the failure is truly only measurement, provenance, or launch control.

### Algorithm

1. First ask what the generated CSD was trying to do.
   - Use the strategy code and helper calls, not just the final score.

2. Then ask where it lacked an operation.
   - Could it build enough candidates?
   - Could it reject duplicates?
   - Could it check the prompt-derived target class?
   - Could it repair invalid outputs?
   - Could it choose among no-gold candidates?
   - Could it return scorer-ready fields?

3. If the operation exists but the strategy did not use it, improve helper surfacing or feedback.

4. If the operation exists but is hard to verify, slow, or returns the wrong shape, repair the helper.

5. If the operation does not exist, add the smallest general helper that supplies it.

6. Only after that should I spend another paid synthesis run on the same hard cell.

7. If I choose a non-core change instead, I should record why the evidence points to measurement or launch control instead of missing CSD ability.

This makes core helper work the default for capability failures. It does not mean reckless rewrites. It means the main design question becomes: "What fair operation does the CSD need that the framework does not currently give it?"

### Core-framework escalation rule

This rule turns your correction into a default decision path.

#### Inputs

- The last failed strategy body.
- The failed examples or candidate artifacts.
- The helper calls the strategy used.
- The helper calls exposed to the author model.
- The exact metric bucket that failed.
- Any verification failures from the generated Dafny strategy.

#### Outputs

The next design step must be one of:

- Repair a helper implementation.
- Improve helper visibility or feedback so the author model can use an existing helper.
- Add a small general helper that supplies a missing operation.
- Record why the failure is not a framework-capability failure.

#### Algorithm

1. Treat another cold relaunch with the same framework as the fallback, not the default.

2. Ask what the generated CSD could not do at the failure point.
   - Could it build enough candidates?
   - Could it recognize candidates that matched the prompt?
   - Could it remove duplicates?
   - Could it repair a bad candidate without breaking parser or verifier rules?
   - Could it choose among no-gold candidates?
   - Could it emit the exact field the scorer needed?

3. If the answer is "no" for any of those, make the next hypothesis a framework/helper hypothesis.

4. Prefer this order:
   1. Fix an existing helper that is wrong, slow, or awkward to use.
   2. Improve the prompt or feedback surface for an existing helper.
   3. Add the smallest general helper that gives the missing operation.
   4. Relaunch cold only after the helper is tested and visible.

5. Do not let "cold discovery" mean "unchanged framework."
   - Cold discovery bans seeded strategies and class-specific tricks.
   - It does not ban improving the general helper library that every cold run can use.

6. Keep the fairness guardrail strict.
   - Allowed: helper behavior, helper visibility, feedback, candidate artifacts, no-gold selection operations.
   - Disallowed: scorer changes, grammar changes, split changes, baseline changes, warm starts, and class-specific task guidance.

7. Before the next paid retry, write one sentence in the ledger answering:
   - "Why is this not just another relaunch of the same failing framework?"

## Inputs → Outputs → Algorithm

### Inputs

The campaign uses these inputs:

- Focal repo state: `/home/aadivyar/csd-generation`.
- Live baseline artifacts, not old notes.
- Active run logs and result JSONs.
- The hypothesis ledger: `docs/experiments/metadecode-fast-iteration-log.md`.
- The handoff/audit docs:
  - `saved-results/2026-06-29-campaign-handoff.md`
  - `saved-results/2026-06-29-paper-ready-gap-audit.md`
  - `saved-results/2026-06-30-parser-helper-big-o-audit.md`
- User policy:
  - SMILES paper-ready means primary UV beats CARS UV.
  - SMILES task descriptions stay pure cold discovery.
  - No class-specific tricks in task text.
  - No `--initial-strategy-file` for synthesis.
  - If train wins but held-out fails and the drop is real, use strict `+0.10` train-bar margin next.
  - Work one SMILES class across model sizes before switching classes.
  - Prefer core framework/helper hypotheses over repeated threshold or launch-setting tweaks when the observed failure is about what the CSD can generate or select.
  - After a capability-shaped failure, audit the helper surface before another paid retry.
  - Paid Bedrock is approved only for recorded AWS account `887730490125`, with launch records and no secret printing.

### Outputs

The campaign is done only when these outputs exist:

- A paper-ready win/loss record for every target SMILES, GSM, and Spider cell.
- Held-out result JSONs for every claimed win.
- Updated `results_matrix.md` for paper-ready wins only.
- A refreshed paper-ready evidence bundle.
- Regression checks showing prior wins still win after framework changes.
- Ablation results that isolate which framework changes mattered.
- A record of which successful changes were core framework/helper changes, not only run-setting changes.

### Algorithm

1. Determine the active stage.
   - We are currently in the SMILES stage.
   - GSM and Spider design work waits unless it is required to record an already-running job.

2. Pick the next cell using the stage policy.
   - SMILES: stay within one class across model sizes before switching classes.
   - GSM: after SMILES, return to the queued GSM hypotheses and recorded H65 evidence.
   - Spider: after GSM, return to unresolved Spider cells and regression of existing Spider wins.

3. Read the live bar.
   - Do not use stale short notes.
   - Use the current controlled-comparison artifacts.

4. Write the prediction before the run.
   - Each run needs a hypothesis, prior, single variable, prediction, and stop condition.
   - This is required even during autonomous operation.

5. Classify the lever before choosing the next run.
   - Level 1: record/provenance fix.
   - Level 2: bar, sample size, timeout, or launch setting.
   - Level 3: speed-only helper implementation.
   - Level 4: helper behavior change that changes what the CSD can test or choose.
   - Level 5: new general helper primitive that gives the CSD a new fair operation.
   - Level 6: disallowed change, such as changing the scorer, dataset, baseline, or adding class-specific win tricks.
   - Also label the lever as either around-framework or core-framework:
     - Around-framework: changes the run conditions around the same CSD, such as bars, timeouts, GPU choice, sample size, launch wrappers, or artifact copying.
     - Core-framework: changes what the generated CSD can do, such as helper behavior, helper docs, candidate artifact shape, feedback signals, no-gold selection primitives, or strategy-visible operations.

6. Prefer the deepest allowed lever that matches the failure.
   - Use Level 1 or 2 only when the evidence says the problem is provenance, unfair measurement, split mismatch, or an underpowered gate.
   - Use Level 3 only when runtime is blocking iteration.
   - Use Level 4 or 5 when the generated CSD is producing the wrong kind of answer, lacks candidate diversity, cannot select among candidates, or cannot repair a common failure.
   - Never use Level 6.

7. Choose the smallest fair test.
   - Use small/train tests to move quickly.
   - Use held-out tests for paper claims.
   - Use CPU-only or dry-run checks when the question is about provenance, artifact shape, or launch safety.

8. Run one paid synthesis cell at a time unless capacity is clearly separate.
   - The active paid SMILES job is H93.
   - H65 is already recorded, so it should not start a new GSM design branch during the SMILES stage.

9. Record the result.
   - If below train bar: record as a training loss and diagnose the failure mode.
   - If above train bar: immediately run held-out.
   - If held-out wins: update the matrix and evidence bundle.
   - If held-out fails: apply the train-win/held-out-fail policy before launching another search.

10. Repeat until the stage is complete or clearly blocked.

## How hypotheses are chosen

### Inputs

For each failed or pending cell, the hypothesis step reads:

- The latest train and held-out scores.
- Per-example failure details when available.
- The generated strategy body and helper calls.
- Parser/helper timing and call counts if runtime is slowing the loop.
- Prior wins and losses on the same dataset family.
- The fairness rules: no scorer changes, no dataset changes, no class-specific prompt tricks, no synthesis warm start.

### Outputs

A good hypothesis produces a short ledger entry with:

- The specific observed behavior to explain.
- The suspected failure layer.
- The single lever to change.
- A prediction that can be false.
- The reason this lever is allowed under the fairness rules.

### Algorithm

1. Locate the first place the run loses useful information.
   - Example: GSM H64/H80 produced text, but not clean machine-readable candidate expressions.
   - Example: H81 trained on 50 examples and held out on 100, so the train signal was too weak for the held-out claim.

2. Separate run-control problems from CSD-capability problems.
   - Run-control problems include wrong files, stale compiled strategies, too-short timeouts, bad GPU settings, or mismatched train/held-out sample sizes.
   - CSD-capability problems include weak candidate pools, repeated candidates, bad no-gold selection, missing repair ability, or semantic wrongness despite valid syntax.

3. Ask whether the CSD had the operation it needed at the point of failure.
   - If outputs are invalid, did it have a general repair or validity-check operation?
   - If outputs repeat, did it have canonical duplicate detection?
   - If one sample is weak, did it have a fair way to build and compare a candidate pool?
   - If the final answer is hidden in messy text, did it have a helper that returns clean machine-readable candidates?
   - If a strategy keeps choosing badly, did it have no-gold selection signals beyond raw parser validity?

4. Ask whether a human could solve the same local problem with one missing tool.
   - Example: "I can see these are duplicates, but the CSD has no canonical duplicate helper."
   - Example: "I can see the molecule is valid but wrong-class, but the CSD has no prompt-derived class check."
   - Example: "I can see the arithmetic answer inside prose, but the CSD has no clean expression extractor."
   - If that sentence is true, the hypothesis should usually be a helper or candidate-contract change.

5. If the problem is run-control, fix the run-control problem first.
   - This keeps results fair and auditable.
   - It should not become the main way to hunt for wins.

6. If the problem is CSD capability, change the helper surface or helper implementation before turning more knobs.
   - Existing helper behavior changes are preferred when one helper is clearly too slow, too weak, or returning the wrong shape.
   - New helper primitives are preferred when the generated strategy lacks a fair operation it needs, such as no-gold candidate pooling, canonical deduplication, static validity checks, or no-gold selection.

7. Keep the change general.
   - A SMILES helper may expose generic molecule validity, canonical form, duplicate detection, or candidate-pool operations.
   - It must not encode "how to make isocyanates win" or any other class-specific trick in the prompt or helper contract.

8. Test the helper before using it in a launch.
   - For code changes, use red/green tests.
   - For launch/materialization changes, use dry-run/static checks.
   - For paid runs, record the account and launch record before the run.

This means future failures should not automatically lead to another bar tweak. The default question should be: "What operation did the generated CSD need but not have?"

### How to decide what could explain the behavior

The hypothesis is not chosen by guessing from the final score. It is chosen by comparing the desired data flow to the actual data flow.

#### Inputs

- Desired data flow for the task.
- Actual generated code.
- Actual helper outputs.
- Actual candidate artifacts.
- Actual score buckets.

#### Outputs

- One concrete mechanism that explains the behavior.
- One framework operation that would change that mechanism.
- One test or probe that could show the mechanism is wrong.

#### Algorithm

1. Write the desired data flow in plain terms.
   - SMILES: prompt -> candidate strings -> valid molecules -> unique molecules -> prompt-class members -> selected final answer.
   - GSM: question -> clean expressions -> variables preserved -> evaluated answer -> scorer-ready final field.
   - Spider: question/schema -> SQL candidate -> parseable SQL -> valid schema references -> official grader input.

2. Write the actual data flow from artifacts.
   - Use the generated strategy, candidate files, logs, and score buckets.

3. Compare the two flows step by step.
   - The first mismatch is the best place to form the hypothesis.

4. Name the missing operation at that mismatch.
   - This should be a verb phrase, not a score phrase.
   - Good: "canonicalize candidates before duplicate filtering."
   - Good: "check whether a SMILES candidate matches the prompt-derived molecule class."
   - Weak: "raise accuracy."

5. Check whether the framework already provides that operation.
   - If yes, inspect why the strategy did not use it or could not verify with it.
   - If no, design the smallest general helper.

6. Predict the smallest visible effect.
   - A good helper hypothesis should predict a local metric before a paper win.
   - Example: class-membership should rise above `0/100`.
   - Example: unique-valid candidates should rise above `0/100`.
   - Example: clean expression candidates should appear in structured fields.

7. Test the local effect before claiming the larger win path.
   - The local effect does not prove a paper win.
   - It only proves the framework change attacked the mechanism it was supposed to attack.

### Around-framework changes versus core-framework changes

The campaign should separate two kinds of changes:

1. Around-framework changes.
   - Inputs: launch command, sample size, train bar, timeout, GPU lane, artifact path, or recording rule.
   - Output: the same generated CSD process runs under cleaner or stricter conditions.
   - Algorithm:
     1. Check whether the run was measured fairly.
     2. Fix mismatched sample sizes, stale artifacts, unsafe launch state, or missing records.
     3. Re-run only if the fixed condition could change the interpretation of the result.

2. Core-framework changes.
   - Inputs: helper library, helper implementation, helper docs, feedback messages, candidate artifact contract, strategy-visible operations, or no-gold selection rules.
   - Output: generated CSDs can express, test, repair, or choose strategies they could not reliably express before.
   - Algorithm:
     1. Identify the missing operation from the failure trace.
     2. Add or repair the smallest general helper that provides that operation.
     3. Test the helper without a paid run first.
     4. Launch a cold synthesis run only after the helper is visible to generated strategies and passes tests.

The practical bias is now toward core-framework changes after repeated failures. Around-framework fixes are still required for fairness and provenance, but they should not substitute for giving the CSD a better set of fair operations.

The risk is higher for core-framework work because it can introduce bugs or accidental unfairness. The guardrails are:

1. Write the hypothesis first.
2. Keep the helper general and no-gold.
3. Do not change scorer, grammar, dataset, baseline, or class-specific task text.
4. Prove the helper behavior with a failing test before implementation and a passing test after implementation.
5. Run a small targeted experiment before treating it as a win path.

### Principles for explaining behavior

When I decide what could explain a failure, I should use these principles:

1. **Find the first bad handoff.**
   - Input: the model output, helper outputs, parser result, scorer result, and final metric.
   - Output: the earliest step where useful information became missing, malformed, repeated, or unusable.
   - Algorithm: trace one failed example from raw model text to final score and stop at the first step that lost the needed signal.

2. **Prefer a mechanism over a surface symptom.**
   - "Accuracy is low" is a symptom.
   - "The candidate pool has 100 valid strings but 0 unique molecules" is closer to a mechanism.
   - "The helper exposes validity but not canonical uniqueness" is a framework-level mechanism.

3. **Look for missing operations, not only bad parameters.**
   - A parameter change asks the same system to search harder.
   - A helper change gives the CSD a new fair operation or a better version of an existing operation.
   - After repeated failures, missing operations are more likely to matter than another threshold.

4. **Stay inside fair-comparison rules.**
   - It is allowed to improve the CSD's internal helper library.
   - It is not allowed to change the scorer, grammar, dataset, baseline, or task text with class-specific win tricks.

5. **Make the explanation falsifiable.**
   - Bad: "The model needs better reasoning."
   - Good: "If repeated candidates are the bottleneck, then adding canonical duplicate detection should increase unique valid candidates on a small train probe without changing the evaluator."

6. **Use the smallest test that can disprove the mechanism.**
   - For a helper behavior claim, write a unit test first.
   - For a candidate-pool claim, run a small no-gold probe.
   - For a train/held-out mismatch claim, compare sample size and difficulty before launching another paid run.

7. **Prefer an operation-level explanation over a knob-level explanation.**
   - Knob-level explanation: "The bar was too high" or "the run needed more attempts."
   - Operation-level explanation: "The CSD generated valid molecules but had no safe way to repair complete wrong-class candidates."
   - Use knob-level explanations only when the artifacts show the framework already had the needed operation.

8. **Ask whether the framework gave the author model a safe move.**
   - A helper that exists in code but is not visible in the prompt is not practically available.
   - A helper that returns the right answer but is too hard to verify may still be unusable.
   - A repair path that breaks parser/verifier preconditions is not a safe move.
   - These are framework problems, not reasons to keep changing thresholds.

### Core-first hypothesis generator

This is the more explicit version of your correction: the next hypothesis should usually come from the CSD process itself, not from the launch wrapper around it.

#### Inputs

- A failed run or weak attempt.
- The generated CSD strategy code.
- The output candidates it produced.
- The helpers it used.
- The helpers it could have used but did not.
- The feedback message sent back to the author model.
- The metric bucket that failed: class membership, validity, syntax, accuracy, uniqueness, exact answer extraction, SQL execution, or another task-specific but fair metric.

#### Outputs

- One framework-level mechanism that could explain the failure.
- One smallest fair framework change to test that mechanism.
- One small test or probe that should fail before the change and pass after it.
- A decision on whether a paid relaunch is justified after the framework change.

#### Algorithm

1. Trace the failed attempt through the CSD process.
   - Start with generated code.
   - Then inspect helper calls.
   - Then inspect generated candidates.
   - Then inspect parser/scorer buckets.
   - Then inspect the feedback sent to the author model.

2. Name the missing operation in plain language.
   - SMILES examples: build a candidate pool, reject duplicates, check prompt-derived molecule class, repair invalid strings, choose among no-gold candidates.
   - GSM examples: extract clean numeric expressions, preserve variable names, compare candidate expressions, keep scorer-ready fields.
   - Spider examples: preserve non-empty SQL, check schema references, repair aliases, avoid empty outputs.

3. Map that operation to the framework surface.
   - If a helper already exists and was used, inspect helper behavior.
   - If a helper exists but was not used, inspect helper docs, helper surfacing, and feedback text.
   - If no helper exists, design the smallest general helper.
   - If the operation would require changing scorer, grammar, dataset, baseline, or class-specific task guidance, reject it.

4. Prefer the highest-impact fair lever.
   - First choice: repair an existing helper that returns weak, slow, or hard-to-use information.
   - Second choice: expose an existing helper better through docs, examples, feedback, or helper-selection rules.
   - Third choice: add a new general helper.
   - Later choice: change bars, timeouts, sample sizes, or launch settings, unless the failure was clearly a measurement/fairness problem.

5. Make the hypothesis testable before another paid run.
   - A helper implementation change needs a direct unit test.
   - A helper-surfacing change needs a prompt/feedback surface check.
   - A candidate-contract change needs a small artifact-shape check.
   - A speed change needs a before/after profile on the measured bottleneck.

6. Relaunch cold only after the framework change is visible and tested.
   - No warm starts.
   - No task-specific tricks.
   - No scorer, grammar, dataset, or baseline changes.
   - One changed mechanism per synthesis run.

The point is not to make the framework bigger for its own sake. The point is to stop spending paid search on a CSD that cannot perform the operation the failure trace says it needs.

## Core framework/helper change policy

The campaign now ranks possible changes by how directly they attack the framework.

| Rank | Lever type | Use when | Example |
|---:|---|---|---|
| 1 | Provenance or recording fix | The result cannot be trusted or found. | Copy postlaunch artifacts into a run-specific folder. |
| 2 | Gate or launch setting | The measurement itself is unfair or too weak. | Align train and held-out sample sizes after H81 showed 50-vs-100 mismatch. |
| 3 | Speed-only helper implementation | Runtime blocks iteration, but scoring behavior should stay the same. | Incremental parser completeness instead of full parse on every prefix. |
| 4 | Existing helper behavior change | A helper gives the CSD too weak a signal or the wrong shape. | Make candidate extraction field-aware instead of reading only one output field. |
| 5 | New general helper primitive | The CSD lacks an operation needed to solve the task fairly. | Generic no-gold candidate pooling, canonical uniqueness, or selector helpers. |
| 6 | Disallowed shortcut | It changes the benchmark instead of the CSD. | Changing the scorer, dataset, baseline, or adding class-specific task hints. |

Ranks 4 and 5 should get more attention than they have so far. They are higher risk than launch-setting changes, but they are also more likely to create real wins because they change what strategies can do.

The updated bias is stronger than "consider ranks 4 and 5." After a capability-shaped failure, ranks 4 and 5 are the default next place to look. Ranks 1 and 2 are still required when the evidence is about trust, fairness, or measurement, but they should not be used as a substitute for giving the generated CSD the missing operation.

### Change size: attack the biggest gaps with the biggest structural changes first

Match the size of the change to the size of the gap. When a cell is far from its
baseline (a large gap), do not open with small tweaks. Start with the biggest core
changes to the framework — the ones that change what the CSD fundamentally can do:

- **Introduce a new CSD** (a new constrained-decoding primitive/operation the author
  model can call), rather than only re-tuning an existing helper's threshold or shape.
- **Change the iteration style itself** (how the synthesis/feedback loop runs — e.g. the
  span-close discipline, when/how spans open, the search/feedback mechanics), rather
  than only adjusting run settings.

The reasoning: a large gap is unlikely to be closed by a small-behavior tweak; it usually
means the CSD is missing a whole operation or the loop is shaped wrong. Small helper-shape
and run-setting changes are for closing the *last* small distance once a big structural
change has already moved the cell most of the way. Go big-first, then refine. This must stay
fair — a new CSD or iteration-style change is only allowed if it changes the mechanism, never
if it leaks dataset-specific "how to win" guidance (that stays rank-6 disallowed).

The practical rule:

1. If a failure is caused by bad measurement, fix measurement.
2. If a failure is caused by slow iteration, fix speed.
3. If a failure is caused by the generated strategy lacking a useful operation, change or add helpers.
4. If a proposed change would leak dataset-specific answer guidance, reject it.

The important update is priority, not permission. Helper and framework changes were already allowed when they were fair. They are now the default response to capability failures. Wrapper changes need a clear reason after a hard miss; helper changes need tests and a fairness check.

### Direct framework intervention checklist

Before I launch another paid retry after a failed cell, I should answer this checklist in the ledger:

1. Did the failure come from bad measurement, missing artifacts, or unsafe provenance?
   - If yes, fix that first.

2. Did the train gate differ from held-out in sample size, difficulty, or target metric?
   - If yes, align the gate or raise the bar before more search.

3. Did the generated CSD lack a fair operation it needed?
   - Examples: candidate pooling, duplicate detection, validity repair, schema-aware SQL checks, clean expression extraction, no-gold selection.
   - If yes, the next hypothesis should be a helper implementation, helper exposure, or new helper primitive.

4. Is the proposed helper general?
   - It can know about the task type, such as SMILES validity or SQL schema references.
   - It cannot know the answer pattern for a specific class, split, or benchmark cell.

5. Can the helper be tested without a paid run?
   - If yes, write the test first and prove the helper works before launching synthesis.

6. What should improve if the helper hypothesis is right?
   - Examples: more unique valid molecules, more clean GSM candidate expressions, fewer empty Spider outputs, faster parser checks.

7. Am I changing the core behavior, or only changing the wrapper around it?
   - If only the wrapper, record why that is enough.
   - If the failure trace shows missing CSD ability, wrapper-only changes should be rejected.

8. Did I inspect sibling helpers that could have solved the same local problem?
   - Search for nearby helpers in the same helper family.
   - Decide whether each is already sufficient, poorly surfaced, too slow, or missing behavior.
   - Do not audit every helper equally; audit the helpers near the failed operation first.

9. Did I make a real helper-level change when the evidence called for one?
   - If the failure is capability-shaped, the next serious move should usually edit helper behavior, expose an existing helper better, or add a new general helper.
   - A paid retry with only a new threshold, timeout, sample count, or launch wrapper needs an explicit reason in the ledger.
   - The reason cannot be "this is easier to launch." It has to explain why the CSD already has the operation it needs.

### Helper audit workflow

This is the concrete workflow for your correction that I should stop only changing things around the framework.

This audit should be complete for the active failure path. It should not stop after the first helper that looks suspicious. It should also not rewrite unrelated helpers just because they exist.

Complete means:

1. List every helper on the path where the failure happened.
   - SMILES candidate failures: candidate extraction, parser checks, RDKit validity, class checks, canonical form, duplicate checks, repair helpers, and no-gold selection helpers.
   - GSM candidate failures: expression extraction, span cleanup, variable preservation, symbolic checks, field writing, and no-gold selection helpers.
   - Spider output failures: SQL parsing, schema-reference checks, alias checks, output packaging, and no-gold SQL selection helpers.

2. For each helper on that path, write one status:
   - works and is visible;
   - exists but is poorly shown to the author model;
   - exists but returns the wrong shape;
   - exists but is too slow on the active path;
   - missing and should be added as a new general helper;
   - irrelevant to this failure path.

3. Only then choose the next framework change.
   - Fix a broken helper first.
   - Then improve helper visibility or feedback.
   - Then add a new general helper if the operation is missing.
   - Use a wrapper-only retry only if the helper audit shows the CSD already had the operation it needed.

#### Inputs

- The generated strategy code.
- The helper calls used by that strategy.
- The helper calls that were available but not used.
- The failed outputs and the score buckets.
- Any parser/helper timing counters.
- The prompt docs that tell the author model which helpers exist.
- The feedback message that the author model saw after failure.

#### Outputs

- A short list of missing or weak operations.
- A decision for each operation: existing helper is enough, helper docs are weak, helper behavior is weak, helper is too slow, or a new general helper is needed.
- One proposed framework change, written as a hypothesis before implementation.
- A small test that can fail before the change and pass after the change.

#### Algorithm

1. Start from a failed example or failed attempt, not from a general complaint.
2. Write the desired data flow in plain terms.
   - Example: raw SMILES text -> cleaned candidate -> valid molecule -> canonical molecule -> unique molecule -> target class member -> selected final answer.
   - Example: raw GSM text -> clean expression -> variables preserved -> expression evaluates -> no-gold selection -> final answer.
   - Example: raw SQL text -> parseable SQL -> schema references exist -> aliases are valid -> official grader input is clean.
3. Mark the first step where the actual run diverged from that desired data flow.
4. Check whether the framework exposes a helper for that step.
5. If a helper exists, inspect four things:
   - Is it visible in the synthesis prompt?
   - Is its contract clear enough for the author model to use?
   - Does its implementation return the data shape the strategy needs?
   - Is it fast enough for the active loop?
6. If no helper exists, design the smallest general helper that supplies that operation.
7. Keep the helper no-gold.
   - It can use the prompt, parser state, schema, molecule validity, canonicalization, or duplicate checks.
   - It cannot encode a hidden answer pattern for one class, split, or benchmark cell.
8. Test the helper directly before launching synthesis.
9. Launch a cold run only after the helper is implemented, visible, and recorded.

The point of this audit is not to touch every helper equally. The point is to avoid another paid retry when the previous evidence says the CSD was missing an operation.

The audit must still be complete for the relevant path. If the failed path is SMILES candidate selection, inspect the SMILES candidate helpers, class-check helpers, validity helpers, canonicalization helpers, duplicate helpers, and selection helpers that sit on that path. If the failed path is GSM expression extraction, inspect the expression, span, field-writing, and candidate-selection helpers. If the failed path is Spider output packaging, inspect SQL parsing, schema-reference, alias, and final-output helpers. Do not stop after the first helper that looks suspicious.

The expected output of this audit is a framework change or a written reason not to make one. A failed hard cell should not end with only "try a different threshold" unless the audit shows the helper surface was already adequate.

### Helper audit priority list

I should audit helper methods in this order:

1. Helpers used by the failed strategy.
   - These are highest priority because the run already depended on them.

2. Helpers that would have addressed the first bad handoff.
   - Example: if SMILES outputs are valid but wrong-class, inspect class-membership and candidate-selection helpers before parser helpers.

3. Helpers documented in the prompt but not chosen by the author model.
   - This checks whether the helper exists but is poorly described or poorly surfaced.

4. Helpers that dominate runtime.
   - These matter when slow helper calls stop us from running enough examples or attempts.

5. Stage-specific dormant helpers.
   - `CompletedSchemaSymbolCount` matters in GSM/Spider stages when schema-symbol rollback helpers are active.
   - `GetTopKTokens` matters only if the chosen strategy uses top-k helpers heavily.
   - `GenerateLogits.prefix_text` matters if profiling shows it is still a meaningful SMILES bottleneck after H93.

This is different from a broad rewrite. A broad rewrite risks bugs and fairness drift. A targeted helper audit asks: "Which operation did this run need, and was the helper surface good enough to provide it?"

## What counts as a win

### SMILES

A SMILES paper-ready win means:

- Held-out primary UV is greater than the live CARS primary UV bar.
- The held-out result is recorded under the controlled-comparison path.
- The run used pure cold discovery.
- The synthesis task text did not include class-specific tricks.
- Validity/syntax is recorded, but it does not need to beat CARS unless the user changes the policy.

Current important SMILES evidence:

- H63: isocyanates-4B is a paper-ready primary-UV win: held-out UV `0.58` versus live CARS UV `0.16`.
- H70: acrylates-2B held-out loss: train `0.42` UV / `0.78` validity, held-out `0.34` UV / `0.82` validity versus `0.36` UV bar.
- H81: acrylates-2B train win but held-out loss: train `0.44` UV / `0.88` syntax on 50 examples, held-out `0.17` UV / `0.78` syntax on 100 examples.
- H86: isocyanates-9B train100 loss: best train attempt was UV/accuracy `0.37` and syntax `0.41` on 100 examples; final attempt was `0.10` / `0.25`; it did not cross the `0.92` / `0.50` train gate, so no held-out re-eval ran.
- H87: implemented the general `PrefixAppearsInPrompt` helper, so generated SMILES strategies can reject prompt-visible duplicate spans without gold labels or class-specific tricks.
- H91: CPU-only H86 audit found a framework-level class/candidate-selection problem: one high-syntax attempt produced `100/100` grammar-valid and RDKit-valid outputs but `0/100` class-membership and `0/100` unique-valid candidates.
- H92: implemented the general prompt-derived `PrefixMatchesPromptMoleculeClass` helper in focal main. This is a core framework/helper change, not a benchmark result.
- H93: active cold isocyanates-9B train100 run from patched focal main after H92. Its purpose is to test whether the new helper surface improves class-member or unique-valid candidate rates without changing task text, scorer, grammar, dataset, baseline, or warm-start policy.

### GSM

A GSM paper-ready win means:

- Held-out accuracy beats the live CRANE accuracy bar for that model/cell.
- Syntax also meets the required bar used for that cell.
- The result is from a held-out re-eval or full fair run, not just a local probe.

Current important GSM evidence:

- H65 GSM-9B is recorded as a train loss: best visible train accuracy was `44.9%` with `95.9%` syntax, below the `53.1%` accuracy and `98.0%` syntax train bars.
- H71 did not run because H65 never crossed its train bar.
- H80/H82/H83 showed that GSM-2B failures are not just a trivial postprocess issue.
- H84 is the queued GSM-stage bare-expression candidate probe, but it should not launch during the SMILES stage.

### Spider

A Spider paper-ready win means:

- Held-out official-grader accuracy beats the live IterGen baseline for that model/cell.
- Syntax is recorded and should not regress.
- Alias/output-format fixes must be no-gold and must not change the scorer.

Current important Spider evidence:

- Spider-2B already has a paper-ready held-out win.
- H78 recorded Spider-9B as a paper-ready held-out win: `0.74` accuracy / `0.99` syntax on 300 examples.
- H52 failed as an empty-output/eval-failure path and should not be treated as a running win.
- Spider-4B remains unresolved.

## Current SMILES strategy

The SMILES stage is the active design stage. The main reason is simple: the user explicitly changed the campaign policy to finish SMILES first, then GSM, then Spider.

### SMILES selection rule

Work one class across model sizes before switching classes.

That means the right shape is:

1. Choose a class.
2. Work through the model sizes for that class.
3. Only switch class when that class is complete or clearly blocked with evidence recorded.

Current class focus is isocyanates because:

- isocyanates-4B has a real held-out primary-UV win.
- isocyanates-9B produced the H86 train100 loss and H91 failure audit.
- H92 added the missing general prompt-class membership helper identified by H91.
- H93 is now the active cold isocyanates-9B run testing that helper surface.
- isocyanates-2B is historically not paper-ready under the live bar and must be revisited later if we are trying to complete all model sizes.
- the next isocyanates step after H93 should depend on the H93 failure mode: if it does not use the new helper or still lacks candidate diversity, inspect helper surfacing, candidate pooling, canonical uniqueness, and no-gold selection before another paid retry.

### SMILES levers I am using

These are the allowed levers:

- Train/held-out sample-size alignment.
  - H81 showed that 50-example train UV can overstate readiness for 100-example held-out UV.
  - H86 fixed the train gate side by using 100 train examples; the matching 100-example held-out path did not run because H86 never crossed the train gate.

- Train bar margin.
  - If a train win fails held-out and train/held-out difficulty is comparable, raise the next train bar by strict `+0.10`.

- Core framework/helper changes.
  - The next serious SMILES design change after H93 is recorded should usually be a general helper or helper-behavior change, not another threshold-only retry.
  - If H93 does not use `PrefixMatchesPromptMoleculeClass`, inspect helper surfacing, helper docs, helper-selection rules, and feedback before adding a new helper.
  - If H93 uses the helper but still produces weak or repeated candidates, inspect the full relevant helper path: candidate pooling, RDKit validity checks, prompt-class membership checks, canonical duplicate checks, and no-gold candidate selection.
  - Good next helper candidates are generic no-gold candidate pooling, canonical duplicate checks, validity checks, repair operations, and candidate-selection helpers.
  - These helpers must stay class-neutral. They can help the strategy manage molecules; they cannot encode class-specific win tricks in the task text or helper contract.
  - A paid retry with only a new threshold, timeout, sample count, or launch wrapper is not the default after a capability-shaped SMILES failure.

- Pure runtime changes.
  - Parser helper speedups are allowed if they do not change scoring semantics.
  - Launch provenance hardening is allowed when provenance is the failure.
  - Better artifact recording is allowed when the artifact shape is blocking diagnosis.

- Model size and class queue.
  - Stay in the current class until complete or blocked.
  - Then move to the next class by live-bar and evidence.

These are not allowed:

- No class-specific hints in the task description.
- No learned isocyanate/acrylate tricks in the prompt.
- No warm-start synthesis from a prior strategy.
- No changing the SMILES evaluator, grammar, or dataset split to get a win.
- No repeated paid relaunches with the same helper surface after a capability-shaped failure unless the ledger explains why the framework already has the needed operation.

### Post-H86/H91/H92 decision rule

H86 is finished and recorded as a train loss, not a paper-ready result. The important fact is not only that the score missed the bar. The useful fact is that H91 split the miss into two different behaviors:

1. Some attempts could make valid molecules but not the right class.
   - H91 found one high-syntax attempt with `100/100` grammar-valid, RDKit-valid outputs and `0/100` class-membership.

2. The best-accuracy attempt still had weak class-valid candidate coverage.
   - H86's best train attempt reached only UV/accuracy `0.37` with syntax `0.41` on 100 examples.

The next SMILES hypothesis was therefore core-framework-first:

1. Add or repair a general no-gold helper that lets generated CSDs test whether a candidate matches the molecule class named in the prompt.
2. Keep the helper class-neutral in its contract and docs.
3. Use only prompt-visible class names and generic evaluator-side class-membership logic.
4. Do not add isocyanate-specific strategy advice to the task text.
5. Prove the helper with unit tests and helper-surface checks before any paid relaunch.

H92 implemented that helper. H93 is the cold run testing whether the synthesis author can use the stronger helper surface.

After H93 finishes:

1. If H93 uses `PrefixMatchesPromptMoleculeClass` and class-member / unique-valid rates improve, continue with candidate-pool and no-gold selection helpers only if accuracy still misses.
2. If H93 does not use the helper, inspect helper surfacing and feedback text before adding another helper.
3. If H93 uses the helper but still repeats or selects weak candidates, inspect canonical duplicate checks, candidate pooling, and no-gold selection.
4. If H93 uses the helper but hits verifier/precondition failures while trying to repair wrong-class complete prefixes, treat that as a core repair-operation gap before another paid retry.
5. If H93 crosses train, run the 100-example held-out re-eval immediately.
6. If H93 loses train, record it as evidence about the helper surface before launching another paid SMILES run.

## GSM strategy once the campaign enters the GSM stage

GSM failures so far look less like a parser-only problem and more like a candidate-construction problem.

The GSM plan is:

1. Treat H65 as recorded evidence: it missed the GSM-9B train bar, so H71 did not run.
2. Do not launch new GSM design work until SMILES is complete or blocked.
3. When GSM starts, begin with the highest-information queued probe:
   - H84 bare-expression candidate generation for GSM-2B.
4. Use structured candidate artifacts, not old report text.
5. Force machine-readable arithmetic expressions instead of prose labels.
6. Preserve variable names and scorer-ready metadata.
7. Use no-gold selection rules.
8. Promote only held-out wins.

The reason is evidence from H80/H82/H83:

- H80 produced weak structured artifacts.
- H83 showed that useful partial spans existed in `full_output`, but the structured artifact failed to capture them cleanly.
- Broad all-field scanning recovered many spans, but that included prompt/helper echoes and is not clean enough to claim a candidate-pool win.

So the next GSM lever should change candidate generation itself, not just the selector.

For GSM, the core-helper direction is:

1. Add or repair general helpers that make candidate expressions machine-readable.
2. Preserve variable names and scorer metadata.
3. Generate multiple independent bare expressions.
4. Select using no-gold consistency or symbolic checks.
5. Only then tune bars or sampling if the candidate pool is strong enough but held-out still drops.

## Spider strategy once the campaign enters the Spider stage

Spider already has real wins, so the Spider stage is partly about completing missing cells and partly about preventing regressions.

The Spider plan is:

1. Preserve Spider-2B and Spider-9B wins.
2. Revisit Spider-4B as the unresolved cell.
3. Treat output-format fixes as allowed only when they are no-gold and do not change the scorer.
4. Do not over-trust 50-example Spider wins.
   - H19 showed a fast 50-set path can look strong while full held-out still loses.
5. Use full held-out official-grader results for claims.

The Spider failure mode is different from GSM:

- GSM mostly needs better symbolic answer construction.
- Spider often needs semantic SQL correctness and clean output packaging.

So I should not blindly port GSM candidate-generation ideas to Spider unless the Spider failure audit says that is the bottleneck.

For Spider, the core-helper direction is:

1. Improve helpers that keep SQL tied to schema symbols during generation.
2. Add no-gold static checks for parseability, schema references, aliases, and output packaging.
3. Repair common format mistakes without changing the official grader.
4. Use candidate pools only if the failure audit shows semantic alternatives or output packaging are the bottleneck.

## Speed and helper optimization policy

Speed work is useful only if it makes the experiment loop faster without changing what counts as correct.

The parser helper patch already did the first safe step:

- Cache prefix-to-text conversion by prefix object.
- Make completeness incremental before falling back to full parse.
- Cache valid-next counts.
- Override `IsDeadPrefix` to avoid duplicate wrapper calls.

Next optimization steps are staged:

1. SMILES stage:
   - After H93 is recorded, profile the next SMILES run before doing another optimization patch.
   - Inspect `GenerateLogits.prefix_text` in `synthesis/evaluate/benchmarks/common/model_utils.py` only if it remains a meaningful timing bucket.
   - Do not assume every helper needs work. Use the H93 profile to decide.

2. GSM/Spider stages:
   - Inspect `CompletedSchemaSymbolCount` only when schema-symbol rollback helpers are active again.

3. Any stage:
   - Inspect `GetTopKTokens` only if the chosen strategy uses top-k helpers heavily.

I should not do a broad helper rewrite just because some helpers are theoretically O(n). The right rule is: profile first, patch the active-stage bottleneck, write a failing test, then implement the smallest behavior-preserving fix.

This speed policy does not replace the core-helper policy above. Speed work makes iteration cheaper. Core helper work changes the search space and is the likely source of new wins when current strategies keep producing the wrong outputs.

## How failures should change the plan

### Failure type 1: training never crosses the bar

Interpretation:

- The current search space or helper menu is not finding a good enough strategy.

Response:

- Do not run held-out.
- Diagnose whether the miss is accuracy, syntax, validity, timeout, or artifact failure.
- If the miss is not a measurement/provenance issue, prefer a Level 4 or Level 5 helper/framework lever over another threshold-only retry.
- Preregister the next attempt.

### Failure type 2: training wins but held-out fails

Interpretation:

- Either the train gate is too weak, the split is easier than held-out, or the strategy overfit.

Response:

1. Check train/held-out difficulty and sample-size alignment.
2. If mismatched, fix the split/bar/gate first.
3. If comparable, raise the train bar by strict `+0.10`.
4. Record the diagnosis before another launch.

### Failure type 3: syntax/validity wins but accuracy fails

Interpretation:

- The parser is doing its job, but the generated content is semantically wrong.

Response:

- Do not add more parser constraints unless the failure is actually parser-related.
- For GSM, improve candidate construction/selection helpers.
- For Spider, inspect semantic SQL failure buckets, then improve schema/output helpers if the audit points there.
- For SMILES, inspect unique-valid diversity, repeated candidates, and invalid candidates, then prefer a generic molecule helper if that is the bottleneck.

### Failure type 4: artifact/provenance failure

Interpretation:

- The experiment may have run, but the result cannot be trusted or traced.

Response:

- Fix provenance first.
- Snapshot prelaunch state.
- Copy postlaunch train/held-out artifacts into run-specific folders.
- Do not promote ambiguous results.

## What I need feedback on

These are the policy points where your feedback can change the campaign speed and direction:

1. **How strict should “all SMILES wins” be?**
   - Current assumption: every SMILES class/model cell should eventually beat live CARS primary UV.

2. **When a live CARS UV bar is near 1.00, should we treat that as a normal target or a likely dead-end threshold?**
   - Current assumption: keep trying until evidence says blocked, but record dead ends honestly.

3. **How much paid search is acceptable per hard cell before switching class or model size?**
   - Current assumption: paid budget is approved, but each paid run still needs a clean one-variable hypothesis and launch record.

4. **Should speed work pause launches, or only apply between runs?**
   - Current assumption: do not interrupt active paid jobs; apply speed patches to new launches after tests pass.

5. **For GSM, do you want me to prioritize small-model mechanism discovery or the closest high-model win once SMILES is done?**
   - Current staged policy says GSM after SMILES, but the GSM internal ordering could still be tuned.

6. **How aggressive should core helper work be after the current active jobs finish?**
   - Answered by your latest correction: core helper/framework work should be more aggressive than it has been.
   - Current policy: if a failure is about what the CSD can generate, validate, repair, or select, prefer a helper implementation, helper-surfacing change, candidate-contract change, or new general helper before another bar-only retry.
   - Current operational version: one failed hard cell should trigger a failure audit; if that audit points to a missing or weak operation, the next design step should be a tested helper/framework change, not another paid run with only a new bar or timeout.

## What I am intentionally not doing

I am not doing these unless you change the policy:

- I am not adding dataset-specific solution tricks into task descriptions.
- I am not counting training wins as paper-ready.
- I am not changing graders, grammars, splits, or baselines to create wins.
- I am not launching GSM/Spider design work during the SMILES stage.
- I am not doing broad helper rewrites without profiling and tests, but I am now prioritizing targeted helper behavior changes and new general helpers when the failure is a CSD-capability failure.
- I am not using prior strategies as synthesis warm starts.

## My current belief

The fastest path is not “try random cells until one wins.” The faster path is:

1. Finish SMILES with aligned gates and cold generic runs.
2. Use held-out failures to calibrate train bars when the failure is a train/held-out mismatch.
3. When the failure is the generated strategy itself, change helper behavior or add a general helper primitive.
4. Keep speed work targeted to measured bottlenecks.
5. Preserve every win with full provenance so the paper matrix does not collapse later.

The current correction is that step 3 should happen earlier and more deliberately than it has so far. If the failure trace says the CSD lacks a fair operation, the next serious move should be a helper or feedback-loop change. More paid retries with the same helper surface should come after that, not before it.

The newest correction is even more direct: when the evidence points at a core CSD limitation, I should treat launch settings as secondary. The campaign should spend more design effort on changing the framework's fair operations than it has so far, because those changes are more likely to move hard cells than another cold relaunch with the same helper surface.

Put plainly: the framework is the product being improved. The experiments are not just trying to find a lucky strategy. They are supposed to reveal which fair operation the current framework fails to give the generated CSD. Once the missing operation is visible, the next move should usually be to build or repair that operation, then relaunch cold to test whether the author model can use it.

The main risk is that some live bars, especially SMILES bars near `1.00`, may be naturally hard to beat under pure cold discovery. If that happens, the right output is not to keep spending blindly. The right output is a recorded dead-end/blocker with evidence: what was tried, why it was fair, what failed, and what lever would be needed to continue.
