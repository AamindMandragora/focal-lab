# CSD Generation Project — Agent Instructions

## Project Overview

This project synthesizes **Constrained Decoding Strategies (CSD)** using LLMs with formal verification via Dafny. The pipeline generates, verifies, compiles, and tests strategies that guarantee valid output from language models according to specified grammars.

## Critical Rules

### 1. No Strategy Guidance in Synthesis Prompts

The synthesis system is a **controlled study**. The only valid inputs to the LLM are:
1. The task description (what the model should accomplish)
2. The list of available tools — signatures, preconditions, postconditions, types only

**NEVER add to any prompt or task description:**
- Recommendations on which tools to use (e.g. "use TemperatureConstrainedStep")
- Patterns to avoid (e.g. "avoid CRANE", "don't use ConstrainedStep")
- Hints about strategy structure (e.g. "try always-constrained", "penalize >> early")
- Comparisons to baseline (e.g. "your previous attempt was CRANE-like")
- Usage hints on tools (e.g. "use IsTokenValidNext to inspect '>>'", "more precise than DeadEndDetection")
- Notes about which structural patterns are preferred or required
- Any "NOTE:" or prose that explains when/why to apply a tool beyond its formal contract

**This restriction applies to the task description itself** — the task description should only describe what the model should accomplish, not how to accomplish it.

**What IS allowed in tool descriptions:**
- Signature, argument types, ranges (e.g. `amount: real in [0.0, 1e8]`)
- Preconditions (e.g. `Requires: parser.IsValidPrefix(prefix) && !parser.IsCompletePrefix(prefix)`)
- Postconditions (e.g. `Ensures: forall t in ValidNextTokens(prefix + [next]) ==> t in lm.Tokens`)
- Return value description (e.g. "returns true if fewer than minValidCount valid continuations exist")

This applies to ALL prompts: `INITIAL_GENERATION_PROMPT`, `EVALUATION_FAILURE_REFINEMENT_PROMPT`, `VERIFICATION_ERROR_REFINEMENT_PROMPT`, and any other synthesis prompts.

### 2. Goal: Improve Over CRANE Baseline

The synthesis objective is to find a CSD strategy that **outperforms CRANE** on the evaluation dataset. Always establish CRANE's baseline accuracy/format/syntax on the same model and sample before declaring a synthesized strategy successful. A synthesized CSD only counts as a result if it beats CRANE on accuracy while maintaining comparable format and syntax rates.

To measure CRANE baseline: run the evaluator on a strategy body of just `generated := helpers.CraneGeneration(lm, parser, prompt, maxSteps, 10, eosToken); cost := helpers.cost;`.

## GPU Assignment

- Use GPUs 1 and 2 (`CUDA_VISIBLE_DEVICES=1,2`) — GPU 0 and 3 are often occupied by others.

## vLLM Eval Default on `focal`

- For `Qwen/Qwen2.5-Coder-14B-Instruct` evaluation on the remote `focal` machine, prefer the vLLM eval configuration that was validated on April 10, 2026:
  - `CUDA_VISIBLE_DEVICES=0,1`
  - `backend='vllm'`
  - `tensor_parallel_size=2`
  - `gpu_memory_utilization=0.6`
  - `max_model_len=3072`
  - `enforce_eager=True`
- This setting was the first one that successfully ran the real synthesis eval loop end to end on `focal` with 14B.
- The purpose of this config is eval/runtime stability, not quality. The model can still produce low-quality or malformed outputs even when the eval loop itself is functioning correctly.
- If GPUs 0 and 1 are too busy, re-check `nvidia-smi` before changing the config. Do not drop back to single-GPU 14B on `focal` unless memory availability clearly supports it.

## Working Run Command (GSM-Symbolic)

```bash
CUDA_VISIBLE_DEVICES=1,2 python run_synthesis.py \
    --task "Solve math word problems step by step, writing each arithmetic computation inside << >> delimiters." \
    --dataset gsm_symbolic \
    --max-iterations 15 \
    --generation-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --eval-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --output-name "gsm_new_tools_csd" \
    --temperature 0.7 \
    --max-tokens 1024 \
    --device cuda \
    --min-accuracy 0.3 \
    --min-format-rate 1.0 \
    --min-syntax-rate 1.0 \
    --eval-sample-size 10
```

## Project Structure

```
csd-generation/
├── run_synthesis.py          # Main CLI entry point
├── synthesis/                # Core synthesis pipeline
│   ├── generator.py          # Qwen-based strategy generation (Dafny code)
│   ├── verifier.py           # Dafny verification wrapper
│   ├── compiler.py           # Dafny → Python compilation
│   ├── runner.py             # Runtime testing
│   ├── feedback_loop.py      # Main orchestration
│   └── prompts.py            # LLM prompt templates
├── evaluations/              # Evaluation framework
│   ├── gsm_symbolic/         # GSM-Symbolic math reasoning
│   └── folio/                # FOLIO first-order logic reasoning
├── dafny/                    # Dafny source files
│   ├── GeneratedCSD.dfy      # Template for generated strategies
│   └── VerifiedAgentSynthesis.dfy  # Core verification module
├── grammars/                 # Lark grammar files
└── outputs/                  # Generated outputs
    └── generated-csd/runs/   # Individual run directories
```

## Key Conventions

### Code Style
- Python 3.10+
- Type hints preferred
- Follow existing patterns in the codebase
- No comments unless explicitly requested

### Dafny Code Generation
- Generated strategies must satisfy verification constraints
- Must include `// CSD_RATIONALE_BEGIN ... // CSD_RATIONALE_END` block
- Must use proper method signatures and contracts
- Must handle termination with `decreases` clauses

### Testing
- Run tests with: `pytest tests/`
- Test pipeline manually with: `python test_pipeline.py`
- Verify Dafny code with: `dafny verify <file.dfy>`

### Evaluation
- Use `evaluations/gsm_symbolic/cli.py` for GSM-Symbolic evaluation
- Use `evaluations/folio/cli.py` for FOLIO evaluation
- Always compare against CRANE baseline

## Common Tasks

### Run Synthesis
```bash
CUDA_VISIBLE_DEVICES=1,2 python run_synthesis.py --task "<task>" --dataset gsm_symbolic
```

### Run Evaluation
```bash
python -m evaluations.gsm_symbolic.cli --run-dir <run-dir> --model <model> --device cuda
```

### Verify Dafny Code
```bash
dafny verify dafny/GeneratedCSD.dfy
```

### Compile Dafny to Python
```bash
dafny build dafny/GeneratedCSD.dfy outputs/compiled/
```

## Research Documentation

- **Document Insights**: When a new insight is discovered during synthesis or evaluation, add it to `RESEARCH_NOTES.md` with a clear heading, date, and supporting evidence.
- **Track Experiments**: Log all experiments with model, dataset, parameters, results, and conclusions.

## Important Notes

- **Model Scale**: 7B models struggle with CSD synthesis. Use 14B+ models for better results.
- **Verification Failures**: Most synthesis attempts fail at Dafny verification stage, not at evaluation.
- **CRANE Baseline**: Expected ~70% accuracy on GSM-Symbolic with Qwen2.5-Coder-7B-Instruct.
- **Format Validity**: CSD ensures 100% format validity and syntax validity.
- **Parser Performance**: The `ValidNextTokens` method in the grammar parser MUST use syncode's `DFAMaskStore` for fast token validity checks. NEVER use brute-force O(vocab) Lark parsing — each call would take 15-150 seconds. The DFA mask store reduces this to ~0.01-0.1s per call. When modifying `evaluations/common/parser_utils.py`, always use `syncode.dfa_mask_store.DFAMaskStore.get_accept_mask()` for validity checks, passing the tokenizer to `create_lark_dafny_parser()`.
