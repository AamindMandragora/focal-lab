// =============================================================================
// GeneratedCSD.dfy - Template for Qwen-generated CSD strategies
// =============================================================================
//
// This file is a template where Qwen generates constrained decoding strategies.
// The synthesis script will:
//   1. Prompt Qwen to generate code for the QWEN_SNIPPET section
//   2. Insert the generated code between the markers
//   3. Run `dafny verify` to check validity
//   4. Compile to Python if verification passes
//
// =============================================================================

include "../proofs/VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  // ===========================================================================
  // QWEN_SNIPPET_START
  // ===========================================================================
  // Qwen generates a Strategy function here.
  //
  // Available primitives (from VerifiedAgentSynthesis.py):
  //
  // Token-Level Constraints (TokenConstraint):
  //   - GrammarMask                    : Only grammar-valid tokens
  //   - Lookahead(depth)               : Avoid dead ends within depth steps
  //   - LengthBound(min, max)          : Enforce length constraints
  //   - BanTokens(banned)              : Blacklist specific tokens
  //   - AllowOnlyTokens(allowed)       : Whitelist specific tokens
  //   - Intersect(a, b)                : Must pass both constraints
  //   - Union(a, b)                    : Can pass either constraint
  //   - NoConstraint                   : Allow all tokens
  //
  // Repair Rules (RepairRules):
  //   - BracketBalance                 : Fix mismatched brackets
  //   - QuoteFix                       : Fix unclosed quotes
  //   - WhitespaceNormalize            : Normalize whitespace
  //   - ComposedRepair(a, b)           : Apply multiple repairs
  //   - NoRepair                       : No repair
  //
  // Attempts (Attempt):
  //   - Unconstrained                  : Free LLM generation
  //   - ConstrainedAttempt(constraint) : Constrained generation
  //   - WithRepair(base, rules)        : Attempt + repair
  //
  // Check Predicates (CheckPredicate):
  //   - ParseOk                        : Output parses under grammar
  //   - Both(a, b)                     : Must pass both checks
  //   - Either(a, b)                   : Must pass at least one
  //
  // Strategies (Strategy):
  //   - Window(start, end, inside, outside) : CRANE-style windowing
  //   - TryK(k, attempt, check, fallback)   : Retry k times, then fallback
  //   - Cascade(strategies, check)          : Try strategies in order
  //   - BestOfN(n, base, check)             : Generate n, pick first valid
  //   - Constrained(constraint)             : Terminal constrained decode
  //   - Free                                : Terminal free generation
  //
  // Convenience functions:
  //   - CRANEStyle(startDelim, endDelim)    : CRANE windowing with << >>
  //   - RetryThenConstrained(k)             : Try k unconstrained, then constrained
  //   - BestOfNWithRepair(n, rules)         : Best-of-N with repair
  //
  // IMPORTANT: The strategy MUST satisfy GuaranteesValidOutput(strategy)
  // This means it must eventually fall back to Constrained or Window.
  // ===========================================================================

  function GeneratedStrategy(): Strategy
  {
    QWEN_INSERT_STRATEGY_HERE  // <-- Qwen replaces this line
  }

  // ===========================================================================
  // QWEN_SNIPPET_END
  // ===========================================================================

  // Lemma: Verify that the generated strategy guarantees valid output
  lemma GeneratedStrategyIsValid()
    ensures GuaranteesValidOutput(GeneratedStrategy())
  {
    // This lemma will fail verification if GeneratedStrategy() doesn't
    // satisfy GuaranteesValidOutput. Qwen must generate a valid strategy.
  }
}

