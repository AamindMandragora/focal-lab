---
name: Qwen→Dafny→Eval Loop
overview: Add an automated pipeline that prompts Qwen via vLLM (OpenAI-compatible API) to generate Dafny CSD programs, injects them into a Dafny template, parse-checks/verifies/compiles them with Dafny 4.11.0, runs them on GSM-symbolic to produce parsed outputs and metrics, and iterates on failures by feeding compiler/verifier errors back to Qwen.
todos:
  - id: template-dafny
    content: "Add Dafny template module `proofs/generated/GeneratedCSD.dfy` with a single `GeneratedProgram(): CSD.Program` slot for Qwen output."
    status: completed
  - id: qwen-synth-driver
    content: Implement `scripts/synthesize_csd.py` to call Qwen via vLLM (OpenAI-compatible API), write candidate Dafny, and iterate with error feedback.
    status: completed
  - id: dafny-build-step
    content: Implement parse-check + verify + compile-to-Python steps using `./.tools/dafny/dafny/dafny` and capture diagnostics.
    status: completed
  - id: runtime-externs
    content: Implement Python extern bindings (`LLM_Generate`, `ParseOk`, `ConstrainedGenerate`, etc.) in the generated module runtime so compiled Dafny can execute.
    status: completed
  - id: gsm-eval
    content: Add `scripts/eval_gsm_symbolic_csd.py` to run GSM-symbolic with a compiled GeneratedProgram and log outputs/metrics.
    status: completed
  - id: closed-loop
    content: "Integrate a failure-repair loop: on Dafny/compile/runtime errors, feed diagnostics back to Qwen and retry until success or max iters."
    status: completed
---

# Implement Qwen-generated CSD program pipeline

## Goal

Implement the end-to-end loop you described:

- Qwen generates Dafny code for a `CSD.Program` (composed from constrained-decoding sub-parts)
- The generated Dafny is inserted into a template module
- We **parse-check / verify / compile** it via Dafny 4.11.0
- We **run GSM-symbolic** to produce parsed outputs + useful metrics
- If anything fails (parse/verify/compile/runtime), we feed back errors to Qwen and iterate

## What already exists

- `proofs/CSD.dfy` now defines the **whole-loop** `CSD.Program` / `Attempt` / `Policy` surface.
- Local Dafny 4.11.0 binary exists at `/.tools/dafny/dafny/dafny`.

## Implementation plan

### 1) Add a Dafny “generated CSD” template

- Create a template file like [`proofs/generated/GeneratedCSD.dfy`](/Users/aadivyaraushan/Documents/GitHub/focal-lab/proofs/generated/GeneratedCSD.dfy) containing:
- `import opened CSD`
- a single function (or const) that returns a `CSD.Program`, e.g. `function GeneratedProgram(): Program { <QWEN_SNIPPET> }`
- strict formatting rules so Qwen outputs only the `Program` expression.

### 2) Implement Qwen synthesis driver (vLLM, OpenAI-compatible API)

- Add [`scripts/synthesize_csd.py`](/Users/aadivyaraushan/Documents/GitHub/focal-lab/scripts/synthesize_csd.py):
- Calls your Qwen server via **vLLM’s OpenAI-compatible API** (base URL + API key from env vars)
- (Optional alternative) If you later prefer Hugging Face Inference/Endpoints, swap the client implementation; the rest of the pipeline stays the same.
- Prompts Qwen with:
- the allowed constructors (`Policy`, `Attempt`, `Program`) **with exact signatures**
- a concrete “tool menu” of available constrained-decoding techniques (names + 1–2 line intent):
- Policy-level (token-level constrained decoding components; compose these into constrained strategies):
- `Base(strategy: ProposalStrategy)`: unconstrained token proposal base over the vocabulary (used as a starting point).
- `MaskWithSynCode(base: Policy)`: grammar-align tokens (SynCode-style masking; intersection with grammar-allowed next tokens).
- `MaskWithCRANE(base: Policy)`: grammar-align tokens (CRANE-style masking placeholder; currently same intersection semantics in Dafny).
- `WithRejection(base: Policy)`: semantic filter via `Accept(prefix,tok)` (models rejection sampling as an abstract filter).
- `Fallback(primary: Policy, backup: Policy)`: if primary has candidates, use it; else backup.
- `Intersect(a: Policy, b: Policy)`: combine constraints by intersecting allowed-next sets.
- (Forbidden) `Union(a: Policy, b: Policy)`: disallowed for Qwen (may break grammar alignment).
- Attempt-level (whole-output attempts; these produce a full candidate output string/sequence):
- `Unconstrained(strategy: ProposalStrategy)`: call `LLM_Generate(...)` to propose a full output.
- `Constrained(policy: Policy)`: call `ConstrainedGenerate(...)` (guarantees `ParseOk`).
- `Repair(base: Attempt)`: post-process a candidate using `RepairTransform(s)` (e.g., whitespace normalization, delimiter fixes).
- `ConstrainedSearch(policy: Policy, beamWidth: nat)`: constrained search-mode generation (beam/DFS-like) via `ConstrainedSearchGenerate(...)`.
- Program-level (whole-loop orchestration; these define the dynamic constrained decoding strategy):
- `TryThenElse(g, attempt, check, onFail)`: run an attempt, accept if check passes, else continue.
- `TryK(g, k, attempt, check, onFail)`: bounded repetition of an attempt without unrolling.
- `BestOfNThenElse(g, n, strategy, check, onFail)`: generate N candidates, select one passing via `BestOfNSelectPassing`, else continue.
- `CompleteIfPrefixOkElse(g, strategy, policy, onFail)`: if `ParseOkPrefix` holds for an unconstrained candidate, complete it with `ConstrainedCompleteFromPrefix`, else continue.
- `ReturnParsed(g, policy)`: final backstop; must end the chain to ensure `ParseOk` by construction.
- Check-level (acceptance criteria):
- `ParseOnly`: accept if `ParseOk(g,s)` holds.
- `ParseAndSemantic`: accept if `ParseOk(g,s) && SemanticOk(g,s)` holds.
- requirement: output must be a single Dafny expression of type `CSD.Program`
- forbidden: defining new modules/functions/loops; no `Union` at policy-level; do not invent new constructors
- a few canonical example strategies to pattern-match (these are examples, not requirements), e.g.:
- TryK + fallback: `TryK(g, 4, Unconstrained(Temperature), ParseOnly, ReturnParsed(g, fallbackPolicy))`
- BestOfN: `BestOfNThenElse(g, 8, Temperature, ParseOnly, ReturnParsed(g, fallbackPolicy))`
- Prefix-completion: `CompleteIfPrefixOkElse(g, Temperature, fallbackPolicy, ReturnParsed(g, fallbackPolicy))`
- Repair wrapper: `TryThenElse(g, Repair(Unconstrained(Temperature)), ParseOnly, ReturnParsed(g, fallbackPolicy))`
- the hard contract: generated `Program` must ensure `ParseOk(g, Run(program,...))` by construction (end with `ReturnParsed`)
- Writes the response into `GeneratedCSD.dfy` (or a timestamped file under `proofs/generated/`)
- Exposes key controls as CLI flags:
- `--max-iters` (default: 10): maximum number of repair attempts (Qwen regeneration cycles) before giving up
- `--smoke-n` (default: 10): number of GSM examples to run for the post-compile smoke test during synthesis
- `--out-dir` (default: `proofs/generated/`): where to write the generated Dafny files / artifacts
- `--template-path` (default: `proofs/generated/GeneratedCSD.dfy`): the Dafny template file to overwrite with the generated snippet
- `--compiled-out-dir` (default: `GeneratedCSD-py/`): output directory for Dafny-to-Python compilation artifacts
- `--max-steps` (default: 200): generation budget passed to `CSD.Run(..., maxSteps)` for both smoke test and evaluation
- `--dataset-path` (default: `./datasets/ml-gsm-symbolic/generated_data/GSM_symbolic.jsonl`): GSM-symbolic dataset file (same as existing runner conventions)
- `--results-dir` (default: `./outputs/generated-csd/`): where to write evaluation outputs and metrics
- `--qwen-model` (default: from env `QWEN_MODEL`): model name sent to vLLM/OpenAI-compatible API
- `--qwen-base-url` (default: from env `QWEN_BASE_URL`): base URL of vLLM server
- `--qwen-api-key` (default: from env `QWEN_API_KEY`, or empty): API key if required by your vLLM deployment

### 3) Parse-check / verify / compile generated Dafny

- In `synthesize_csd.py`, for each candidate:
- **Parse-check (parse + resolve only)**: run `./.tools/dafny/dafny/dafny /compile:0 /noVerify proofs/CSD.dfy proofs/generated/GeneratedCSD.dfy`.\n+  This step is a fast gate that checks **syntax + name/type resolution** (imports, constructor names/arity, type errors) but does **not** run verification and does **not** compile.
- **Verify** (required): run Dafny verification on the same files and reject any candidate that does not verify (keep the spec light so this is fast)
- **Note on verification feasibility (termination)**: ensure the core combinator library (`proofs/CSD.dfy`) verifies under Dafny 4.11. In particular, the `TryK` combinator should be implemented in a way that Dafny can prove termination (e.g., recursion that clearly decreases `k`, or an explicit `decreases` measure). If verification fails due to termination, refactor `TryK` before relying on it in the synthesis loop.
- **Compile to Python**: compile `proofs/CSD.dfy` + `GeneratedCSD.dfy` to a new folder `GeneratedCSD-py/`.
- On failure: capture stderr, truncate/summarize, feed it back to Qwen as “compiler feedback”, retry up to `--max-iters`.

### 4) Add Python runtime bindings for externs

Implement extern functions used by `CSD.Run`:

- `LLM_Generate` (calls a model endpoint for unconstrained generation)
- `ParseOk` / `ParseOkPrefix` (calls your parser/Lark grammar)
- `ConstrainedGenerate` / `ConstrainedCompleteFromPrefix`:
- `ConstrainedGenerate(g, policy, prompt, maxSteps)`: the **primary constrained decoding backend**. Given a grammar handle `g` and a `Policy` (masking/rejection/fallback composition), it generates an output that is guaranteed to satisfy `ParseOk(g, output)`. This is what `ReturnParsed(...)` ultimately uses as the “backstop”.
- `ConstrainedCompleteFromPrefix(g, policy, prefix, maxSteps)`: a **prefix-guided completion** backend. It assumes the input is already a parse-ok *prefix* (as checked by `ParseOkPrefix`) and completes it to a fully parse-ok output under grammar `g`. This is used by `CompleteIfPrefixOkElse(...)` to turn a partially-valid unconstrained candidate into a guaranteed parse-ok output.
- (Out of scope for the initial pipeline) We can later add advanced techniques (semantic validators, repair transforms, constrained search, best-of-N) once the basic end-to-end loop is working.

This will live in the generated Python module’s `module_.py` (same pattern as existing `ConstrainedDecoding-py/module_.py`).

### 5) Control flow (closed-loop synthesis + build + smoke-eval)

- Wire `scripts/synthesize_csd.py` to implement the full control loop:
- generate candidate Dafny snippet with Qwen
- write into `--template-path`
- parse-check → verify → compile-to-Python
- run a GSM-symbolic **smoke test** on `--smoke-n` examples:
- execute the compiled `GeneratedProgram()` via `CSD.Run(program, prompt, --max-steps)`
- ensure runtime completes without exceptions
- record basic metrics (parse-ok, calls, tokens, latency)
- if any step fails (parse/verify/compile/runtime), feed back diagnostics to Qwen and retry (up to `--max-iters`)
- if smoke test passes, optionally proceed to full dataset evaluation (next section)

### 6) Input dataset + full evaluation (GSM-symbolic)

- Add [`scripts/eval_gsm_symbolic_csd.py`](/Users/aadivyaraushan/Documents/GitHub/focal-lab/scripts/eval_gsm_symbolic_csd.py):
- Loads GSM-symbolic dataset from `--dataset-path` (default: `./datasets/ml-gsm-symbolic/generated_data/GSM_symbolic.jsonl`)
- Runs the **compiled generated** `GeneratedProgram()` via `CSD.Run(program, prompt, --max-steps)` for each example
- Writes outputs + metrics to `--results-dir` (default: `./outputs/generated-csd/`)

**Metrics to compute** (initial set):

- Parse success rate (should be ~100% by spec; measure anyway)
- Number of attempt nodes evaluated before success (how many `TryThenElse`/`TryK` steps)
- Whether success came from an unconstrained vs constrained attempt (where available)
- Total tokens generated, latency, and total model calls (unconstrained + constrained)
- (Optional) answer correctness if you implement extraction/scoring for GSM-symbolic

## Files added/changed

- Add [`proofs/generated/GeneratedCSD.dfy`](/Users/aadivyaraushan/Documents/GitHub/focal-lab/proofs/generated/GeneratedCSD.dfy)
- Add [`scripts/synthesize_csd.py`](/Users/aadivyaraushan/Documents/GitHub/focal-lab/scripts/synthesize_csd.py)
- Add [`scripts/eval_gsm_symbolic_csd.py`](/Users/aadivyaraushan/Documents/GitHub/focal-lab/scripts/eval_gsm_symbolic_csd.py)