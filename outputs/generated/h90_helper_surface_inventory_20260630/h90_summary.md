# H90 helper surface inventory

Created: 2026-06-30T16:26:29.009521+00:00

## Inputs → Outputs → Algorithm

**Inputs:** `VerifiedAgentSynthesis.dfy`, `prompts.py`, `feedback_loop.py`, and Python tests.

**Outputs:** `h90_helper_inventory.json` and this summary, with helper tags for future framework work.

**Algorithm:** parse helper definitions, prompt docs, `_ALL_HELPER_NAMES`, feedback classification sets, and test references; then tag each helper as patch-now, expose-now, profile-later, stage-later, or leave-alone.

## Safety

- model_calls: 0
- gpu_calls: 0
- billed_api_calls: 0
- score_artifact_edits: 0

## Counts

- expose_now: 9
- leave_alone: 34
- profile_later: 7
- stage_later: 55
- prompt_universe_count: 64
- prompt_universe_gap_count: 0
- untested_prompt_helper_count: 51

## Key findings

- All helpers in _ALL_HELPER_NAMES are now implemented, prompt-documented, and feedback-classified.
- 51 prompt-universe helper(s) have no direct test reference found by text scan; prioritize only when tied to an active failure path.
- Speed-sensitive helpers should remain profile-gated: GenerateLogits.prefix_text after H86, CompletedSchemaSymbolCount in GSM/Spider stages, GetTopKTokens only if a strategy uses top-k heavily.

## Top actionable records

- `BoostValidNextAndEos` — leave_alone: no active-stage gap detected
- `ChooseNextToken` — leave_alone: no active-stage gap detected
- `ChooseNextTokenUnconstrained` — leave_alone: no active-stage gap detected
- `Contains` — leave_alone: Dafny definition not currently exposed to generated helper universe
- `DeadEndAvoidingStep` — stage_later: generated/classified helper has no direct test reference found
- `GenerateLogits` — profile_later: known or suspected runtime-sensitive helper; patch only if active profile shows material cost
- `GenerateUnconstrainedChunk` — leave_alone: no active-stage gap detected
- `HasUnmaskedToken` — leave_alone: implemented/internal-looking helper not exposed to generated CSDs
- `IdToLogit` — leave_alone: implemented/internal-looking helper not exposed to generated CSDs
- `IdToToken` — leave_alone: implemented/internal-looking helper not exposed to generated CSDs
- `IdsToLogits` — leave_alone: implemented/internal-looking helper not exposed to generated CSDs
- `IsCompletePrefix` — profile_later: known or suspected runtime-sensitive helper; patch only if active profile shows material cost
- `IsDeadPrefix` — leave_alone: implemented/internal-looking helper not exposed to generated CSDs
- `IsMasked` — leave_alone: Dafny definition not currently exposed to generated helper universe
- `IsValidPrefix` — profile_later: known or suspected runtime-sensitive helper; patch only if active profile shows material cost
- `MaskToken` — leave_alone: implemented/internal-looking helper not exposed to generated CSDs
- `MaskTokens` — leave_alone: implemented/internal-looking helper not exposed to generated CSDs
- `MaskTokensExcept` — leave_alone: implemented/internal-looking helper not exposed to generated CSDs
- `MaskValidNextAndEos` — profile_later: known or suspected runtime-sensitive helper; patch only if active profile shows material cost
- `ParseG` — leave_alone: no active-stage gap detected

## Interpretation

H89 appears to have closed the main name-only helper exposure gap in the prompt universe. The next helper patch should therefore be driven by active H86/H65 failure evidence rather than another broad exposure cleanup.

This is not a benchmark result and does not change `results_matrix.md`.
