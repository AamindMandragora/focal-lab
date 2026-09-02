# Parser helper big-O audit

**Date:** 2026-06-30  
**Scope:** CSD parser and masking helper paths used by SMILES/GSM/Spider constrained decoding.  
**Worktree:** `/home/aadivyar/.config/superpowers/worktrees/csd-generation/parser-big-o-20260630`

## Result

I implemented and tested the first safe speed patch in an isolated focal worktree, then copied it into the main focal checkout at `/home/aadivyar/csd-generation` after the user explicitly asked for that.

Main focal verification after copy:

- `python -m pytest tests/test_parser_helper_big_o.py -q` -> `2 passed`
- `python -m pytest tests/test_helper_surface_contract.py tests/test_mask_brace_block.py tests/test_smiles_rolling_suffix.py -q` -> `10 passed, 6 skipped`
- `python -m py_compile synthesis/evaluate/benchmarks/common/parser_utils.py synthesis/evaluate/benchmarks/common/model_utils.py` -> passed

Important caveat: H86 was already running before this copy. The running Python process may keep the old imported helper code; the copied patch is guaranteed for new imports/new launches from main, not guaranteed to speed the already-live H86 process.

## What changed

1. `IsCompletePrefix(prefix)` now uses Syncode's incremental parser accept state first, and only falls back to full Lark parse on unexpected internal errors.
2. Repeated calls on the same Dafny prefix object now reuse a prefix-object cache before converting the prefix to text.
3. `ValidNextTokenCount(prefix)` now caches the full-mask count instead of repeating `accept_mask.sum().item()`.
4. `IsDeadPrefix(prefix)` is overridden in the runtime parser bridge so generated CSDs do not have to call `IsCompletePrefix` and `ValidNextTokenCount` through two separate wrappers.

## Verification

- RED test before patch:
  - `tests/test_parser_helper_big_o.py::test_complete_prefix_uses_incremental_end_state_before_full_parse` failed because the old helper made zero incremental-parser calls for completeness.
  - `tests/test_parser_helper_big_o.py::test_valid_next_token_count_caches_sum_for_same_prefix` failed because the old helper summed the same mask twice.
- GREEN tests after patch:
  - `python -m pytest tests/test_parser_helper_big_o.py -q` -> `2 passed`
  - Real SMILES completeness sample check matched old full-parse answers for `''`, `C`, `N`, `N=C=O`, `CCN=C=O`, `C(`, `C1`, `XYZ`, and `C.CN=C=O`.
  - `python -m pytest tests/test_helper_surface_contract.py tests/test_mask_brace_block.py tests/test_smiles_rolling_suffix.py -q` -> `10 passed, 6 skipped`
  - `python -m py_compile synthesis/evaluate/benchmarks/common/parser_utils.py synthesis/evaluate/benchmarks/common/model_utils.py` -> passed

## Hot helper audit

| Helper path | Current cost before this patch | Patch status | Remaining risk |
|---|---:|---|---|
| `IsCompletePrefix(prefix)` | O(prefix length) text conversion plus full parse on text-cache miss | Fixed in worktree: prefix-object cache + incremental accept-state completeness | Need broader real-grammar sample check before merging into live main |
| `IsValidPrefix(prefix)` | O(prefix length) text conversion, then incremental parser that still lexes the current text | Partially fixed: repeated same-prefix calls skip text conversion | Full lexing is still O(prefix length) on a new text prefix |
| `ValidNextTokens(prefix)` | O(prefix length) text conversion, valid-prefix check, accept-mask lookup, then materialize valid tokens | Partially fixed: prefix text and mask caching | Materializing valid tokens remains O(number of valid tokens) |
| `ValidNextTokenCount(prefix)` | O(prefix length) text conversion plus O(vocab) mask sum | Fixed in worktree for repeated prefixes/texts | First count for a new text still needs full-mask sum |
| `ValidNextToken(prefix, token)` | O(prefix length) text conversion plus accept-mask lookup | Partially fixed: prefix text/cache reuse | Per-token checks are still worse than group/full-mask checks when used in loops |
| `GroupHasValidMember(prefix, group)` | O(prefix length) text conversion plus one accept-mask lookup plus O(group size) | Partially fixed: prefix text/cache reuse | Still linear in group size, but already avoids per-token parser calls |
| `MaskValidNextAndEos(parser, prefix, eos)` | Uses parser full mask directly when available | Already good; benefits from `_get_accept_mask_for_prefix` cache | Tensor copy/subset remains O(vocab/subset size) |

## Complete helper-surface audit

I audited the helper surface in these authoritative files on focal main:

- `synthesis/verify/library/VerifiedAgentSynthesis.dfy`
- `synthesis/evaluate/benchmarks/common/parser_utils.py`
- `synthesis/evaluate/benchmarks/common/model_utils.py`

This is the complete method/function inventory I found, grouped by cost shape.

### Parser bridge helpers

| Helper(s) | Cost shape | Finding |
|---|---:|---|
| `_tokens_to_text`, `_prefix_to_text`, `IsValidPrefix`, `IsCompletePrefix` | O(prefix length), cached by prefix object and text | Patched. Repeated calls on the same prefix no longer rescan the Dafny prefix; completeness now uses incremental accept state before full parse fallback. |
| `_is_valid_prefix`, `_is_complete` | Syncode call still lexes current text; old completeness used full parse | Patched for completeness. Remaining first-call cost is still at least O(prefix length) because Syncode lexes text. |
| `_get_accept_mask_for_text`, `_get_accept_mask_for_prefix`, `_get_valid_token_indices`, `ValidNextTokens`, `ValidNextToken`, `GroupHasValidMember` | O(prefix length) first call plus O(vocab or group size), then cached by text/prefix | Already uses DFA mask store; patch adds prefix-mask cache and group bulk path. Fallback brute force is O(vocab * parser-check), but only used when Syncode mask store is absent. |
| `_valid_next_token_count_for_text`, `ValidNextTokenCount`, `IsDeadPrefix` | O(vocab) first count, cached after patch | Patched. `IsDeadPrefix` now avoids separate wrapper calls and reuses cached count/completeness. |
| `CompletedSchemaSymbolCount`, `_compute_unit_rollback_info` | O(prefix length) text/render scan | Not hot for SMILES. Relevant to Spider/GSM unit rollback; can be optimized later with a running counter only if profiling shows it matters. |
| `_tokenizer_cache_fingerprint`, `_ensure_syncode_import_path`, `_load_grammar_text`, `_get_parser_components`, `_get_cached_dfa_mask_store`, `create_lark_dafny_parser`, `get_builtin_grammar`, `print_parser_timings` | Setup/reporting cost | Not per-token hot path. No change. |

### Python model bridge helpers

| Helper(s) | Cost shape | Finding |
|---|---:|---|
| `_prefix_text`, `GenerateLogits` in HF/vLLM LMs | O(full prompt prefix length) before every model call | Still a possible cost. But model/Bedrock/vLLM calls dominate many steps, and object-cache benefit is less clear because Dafny often builds fresh `prompt + prefix` sequences. I did not patch this yet. |
| `MaskValidNextAndEos`, `BoostValidNextAndEos`, `_parser_full_mask`, `_expand_full_mask`, `_subset_mask_from_full_mask` | O(vocab/subset size) tensor operations | Already vectorized in Python and uses parser full masks directly. Good current shape. |
| `MaskTokensExcept`, `MaskToken`, `IsMasked`, `_token_indices_for_token` | O(valid token count) or O(duplicate ids for token) | Already uses `_token_str_to_indices`; acceptable. `MaskTokensExcept` remains linear in valid set, but most hot paths should prefer parser full masks. |
| `_token_str_from_id`, `_token_strs_from_text`, `_dafny_prefix_from_token_strs`, `_build_tokens_dafny`, `_build_unconstrained_chunk_result` | O(token count) conversion | Cached token-id decode exists. No obvious safe patch without changing semantics. |
| `SpanGrounded`, `FirstUngroundedIdentifierTokenIdx`, `_grounding_support_set`, `_parse_schema_support`, `_candidate_identifiers`, `_candidate_identifiers_with_pos`, `_first_ungrounded_token_idx` | O(unit text/support size) | Already caches support set per instruction text. Not hot for SMILES; more relevant to Spider grounding. |
| `PenalizeTriedTokenAt`, `_apply_recurrence_penalty` | O(number of tried penalties) plus prefix text handling | Bounded by rollback attempts; not the current H86 bottleneck. |
| `_finalize_full_logits`, `_sample_full_token_id`, `_finalize_from_logprob_dict`, `_select_constrained_index`, `ChooseNextToken`, `ChooseNextTokenUnconstrained`, `IdToLogit` | Tensor/top-k/logit operations | Already mostly vectorized. `_finalize_from_logprob_dict` has an existing top-k optimization. |
| GPU/runtime helpers: `get_model_input_device`, `get_max_input_length`, `_configure_vllm_multiprocessing`, `_get_cached_vllm_engine`, `max_cuda_devices_from_env`, `limit_cuda_visible_devices`, `visible_cuda_device_ids`, `pick_cuda_device_index_with_most_free_memory`, `clear_vllm_engine_cache` | Setup/cleanup cost | Not per-token CSD helper hot path. No change. |

### Dafny helper library

| Helper(s) | Cost shape | Finding |
|---|---:|---|
| `ConstrainedStep`, `DeadEndAvoidingStep`, `AdaptiveConstrainedStep`, `AdaptiveConstrainedStepWithPenalties`, `PenalizedConstrainedStep`, `BoostedConstrainedStep`, `SafeBoostedConstrainedStep`, `SafePenalizedConstrainedStep`, `SoftConstrainedStep`, `SafeSoftConstrainedStep`, `ConfidenceGatedStep`, `RepetitionPenaltyStep`, `SafeRepetitionPenaltyStep`, `TemperatureConstrainedStep`, `SafeTemperatureConstrainedStep` | Mostly one model-logit call plus parser mask calls | These benefit directly from the parser patch because their repeated `IsValidPrefix` / `IsDeadPrefix` / mask checks now reuse text/mask/count caches. |
| `GroupHasValidMember`, `BoostValidGroups`, `ValidTokenCount`, `TopValidCandidates`, `IsTokenValidNext` | Parser/group/top-candidate queries | Runtime bridge already bulk-optimizes `GroupHasValidMember`; `TopValidCandidates` still has loop work over valid candidates and logits but is not the main SMILES H86 path. |
| `RollbackConstrainedSpan`, `RollbackConstrainedSuffix`, `RollbackConstrainedToComplete`, `RollbackAndRegenerate`, `RollbackAndContinue`, `RollbackToCompletePrefix` | Repeated suffix checks; worst case O(tokens * parser-check) | Parser patch helps repeated checks. A deeper optimization would track last complete prefix while generating, but that changes verified Dafny helper behavior and should be done with a separate proof/test pass. |
| `CloseSpanIfComplete`, `RegenerateUnitOnCheckFailure`, `RegenerateUnitOnGroundingFailure`, `CloseSpanWithinBudget` | Loop over budget; repeated completeness/schema checks | Parser patch helps completeness. `CompletedSchemaSymbolCount` remains O(prefix length) and is a later Spider/GSM-stage optimization candidate, not current SMILES priority. |
| `UnconstrainedStep`, `UnconstrainedChunk`, `UnconstrainedGeneration`, `ConstrainedGeneration`, `RolloutConstrainedWithPenalties`, `SpeculativeConstrainedRollout`, `CraneGeneration`, `ManagedStep`, `GenerateWithManagedSpan`, `GenerateWithPrefixAndManagedSpan` | Loop over generated tokens; cost dominated by model calls and parser masks | No obvious single helper bug beyond parser bridge repeated work. |
| Logit/token helpers: `ValidTokensIdsLogits`, `IdToToken`, `TokenToId`, `TokenToIdRecursive`, `IdToLogit`, `TokenToLogit`, `TokensToLogits`, `IdsToLogits`, `MaskToken`, `MaskTokens`, `MaskTokensExcept`, `IsMasked`, `HasUnmaskedToken`, `BoostTokenLogits`, `SafeBoostTokenLogits`, `PenalizeTokenLogits`, `SafePenalizeTokenLogits`, `MaskTokensInPrefix`, `GetHighestLogitToken`, `GetLogitGap`, `GetTopKTokens`, `GetTokenLogit`, `ScaleAllLogits`, `SaveLogitsSnapshot`, `RestoreLogitsSnapshot` | Mostly O(vocab), O(k*vocab), or O(token-list length) in spec | The runtime Python bridge already handles the hottest vocab masks with tensors. `GetTopKTokens` is O(k*vocab) in Dafny spec and would be a future target only if profiling shows strategies call it often. |
| String/span helpers: `Contains`, `RenderPrefix`, `RenderedEndsWith`, `AppendTaskGuidance`, `ConstrainedSymbol`, `ConstrainedSymbolInGenerated`, `OpenConstrainedSpan`, `EnterObservedConstrainedSpan`, `AppendConstrainedToken`, `CloseConstrainedSpan`, `LastTokenBefore`, `DeadEndDetection` | O(token/string length) | No immediate big-O fix found. They are small compared with parser/model calls in current SMILES runs. |

## Recommended next step

For the next optimization pass, I would not touch all helpers equally. I would profile the new SMILES run after H86 finishes and look specifically at:

1. `GenerateLogits.prefix_text` time in `model_utils.py`.
2. `CompletedSchemaSymbolCount` only when we enter the GSM/Spider stages.
3. `GetTopKTokens` only if a chosen strategy uses top-k helpers heavily.

Those are the remaining plausible big-O levers after the parser bridge patch.
