# CSD Generation Project — Claude Instructions

## Critical Rule: No Strategy Guidance in Synthesis Prompts

The synthesis system is a **controlled study**. The only valid inputs to the LLM are:
1. The task description (what the model should accomplish)
2. The list of available tools — signatures, preconditions, postconditions, types only

**NEVER add to any prompt:**
- Recommendations on which tools to use (e.g. "use TemperatureConstrainedStep")
- Patterns to avoid (e.g. "avoid CRANE", "don't use ConstrainedStep")
- Hints about strategy structure (e.g. "try always-constrained", "penalize >> early")
- Comparisons to baseline (e.g. "your previous attempt was CRANE-like")
- Usage hints on tools (e.g. "use IsTokenValidNext to inspect '>>'", "more precise than DeadEndDetection")
- Notes about which structural patterns are preferred or required
- Any "NOTE:" or prose that explains when/why to apply a tool beyond its formal contract

Adding any such guidance invalidates the study by biasing the synthesis model's choices. The model must discover effective strategies autonomously from the task description and tool contracts alone.

**What IS allowed in tool descriptions:**
- Signature, argument types, ranges (e.g. `amount: real in [0.0, 1e8]`)
- Preconditions (e.g. `Requires: parser.IsValidPrefix(prefix) && !parser.IsCompletePrefix(prefix)`)
- Postconditions (e.g. `Ensures: forall t in ValidNextTokens(prefix + [next]) ==> t in lm.Tokens`)
- Return value description (e.g. "returns true if fewer than minValidCount valid continuations exist")

This applies to ALL prompts: the initial-generation, evaluation-failure, verification-error, runtime-error, compilation-error, and format-repair author prompts rendered from `synthesis/prompt_rendering/templates/author_prompts/*.j2`, and any other synthesis prompts.

## Goal: Improve Over CRANE Baseline

The synthesis objective is to find a CSD strategy that **outperforms CRANE** on the evaluation dataset. Always establish CRANE's baseline accuracy/format/syntax on the same model and sample before declaring a synthesized strategy successful. A synthesized CSD only counts as a result if it beats CRANE on accuracy while maintaining comparable format and syntax rates.

To measure CRANE baseline: run the evaluator on a strategy body of just `generated := helpers.CraneGeneration(lm, parser, prompt, maxSteps, 10, eosToken); cost := helpers.cost;`.

## GPU Assignment
- Use GPUs 1 and 2 (`CUDA_VISIBLE_DEVICES=1,2`) — GPU 0 and 3 are often occupied by others.

## Working Run Command (GSM-Symbolic)
```bash
CUDA_VISIBLE_DEVICES=1,2 python run_synthesis.py \
    --task "Solve math word problems step by step. Write the final numeric answer inside << >> delimiters." \
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
