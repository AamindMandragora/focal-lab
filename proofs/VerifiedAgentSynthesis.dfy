module VerifiedDecoderAgent {
  type Token = string
  type Prefix = seq<Token>
  type Id = nat
  type Logit = real

  class LM {
    const Tokens: seq<Token>
    const Ids: seq<Id>
    var Logits: seq<Logit>

    predicate ValidTokensIdsLogits() reads this
    {
      ((|Tokens| == |Ids|) && (|Ids| == |Logits|) && (|Ids| > 0 && Ids[0] == 0)) &&
      (forall i, j :: 0 <= i < |Tokens| && 0 <= j < |Tokens| && i != j ==> Tokens[i] != Tokens[j]) &&
      (forall token: Token :: token in Tokens ==> (exists i :: 0 <= i < |Ids| && Tokens[i] == token)) &&
      (forall i :: 0 <= i < |Ids| ==> (i == Ids[i]) && (i in Ids)) && 
      (forall i :: 0 <= i < |Logits| ==> Logits[i] <= 1e9 && Logits[i] >= -1e9)
    }

    constructor {:extern} {:axiom} ()
      ensures ValidTokensIdsLogits()

    function IdToToken(id: Id) : (token: Token)
      reads this
      requires ValidTokensIdsLogits()
      requires id in Ids
      ensures token in Tokens
      ensures ValidTokensIdsLogits()
    {
      Tokens[id]
    }

    function TokenToId(token: Token) : (id: Id)
      reads this
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures id in Ids
      ensures ValidTokensIdsLogits()
    {
      TokenToIdRecursive(token, Tokens)
    }

    function TokenToIdRecursive(token: Token, tokens: seq<Token>) : (id: Id)
      reads this
      requires ValidTokensIdsLogits()
      requires forall t: Token :: t in tokens ==> t in Tokens
      requires token in tokens
      requires 0 < |tokens| <= |Ids|
      ensures id in Ids
      ensures 0 <= TokenToIdRecursive(token, tokens) < |tokens|
      ensures ValidTokensIdsLogits()
    {
      if tokens[0] == token then 0
      else 1 + TokenToIdRecursive(token, tokens[1..])
    }

    function IdToLogit(id: Id) : (logit: Logit)
      reads this
      requires ValidTokensIdsLogits()
      requires id in Ids
      ensures logit in Logits
      ensures ValidTokensIdsLogits()
    {
      Logits[id]
    }

    function TokenToLogit(token: Token): (logit: Logit)
      reads this
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
    {
      IdToLogit(TokenToId(token))
    }

    function TokensToLogits(tokens: seq<Token>): (logits: seq<Logit>)
      reads this
      requires ValidTokensIdsLogits()
      requires |tokens| > 0
      requires forall token: Token :: token in tokens ==> token in Tokens
      ensures ValidTokensIdsLogits()
    {
      if (|tokens| == 1) then [TokenToLogit(tokens[0])]
      else [TokenToLogit(tokens[0])] + TokensToLogits(tokens[1..])
    }

    function IdsToLogits(ids: seq<Id>): (logits: seq<Logit>)
      reads this
      requires ValidTokensIdsLogits()
      requires |ids| > 0
      requires forall id: Id :: id in ids ==> id in Ids
      ensures ValidTokensIdsLogits()
    {
      if (|ids| == 1) then [IdToLogit(ids[0])]
      else [IdToLogit(ids[0])] + IdsToLogits(ids[1..])
    }

    method MaskToken(token: Token)
      modifies this
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
    {
      var id := TokenToId(token);
      Logits := Logits[..id] + [0.0] + Logits[id+1..];
    }

    method MaskTokens(tokens: seq<Token>)
      modifies this
      requires ValidTokensIdsLogits()
      requires forall token: Token :: token in tokens ==> token in Tokens
      ensures ValidTokensIdsLogits()
    {
      var N := |tokens|;
      var i := 0;
      while (i < N)
        invariant ValidTokensIdsLogits()
        decreases N - i
      {
        MaskToken(tokens[i]);
        i := i + 1;
      }
    }

    method {:extern} {:axiom} GenerateLogits(input: Prefix) modifies this
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()

    method {:extern} {:axiom} ChooseNextToken(input: Prefix) returns (token: Token)
      requires ValidTokensIdsLogits()
      ensures token in Tokens
      ensures ValidTokensIdsLogits()
  }

  class Parser {
    predicate {:extern} {:axiom} IsValidPrefix(prefix: Prefix)
      ensures forall k: nat :: 0 <= k < |prefix| - 1 ==> IsValidPrefix(prefix[k..])

    predicate {:extern} {:axiom} IsCompletePrefix(prefix: Prefix)
      ensures IsValidPrefix(prefix)

    function {:extern} {:axiom} ValidNextTokens(prefix: Prefix): seq<Token>
      requires IsValidPrefix(prefix)
      ensures forall t :: t in ValidNextTokens(prefix) ==> IsValidPrefix(prefix + [t])
      ensures IsValidPrefix(prefix) ==> (IsCompletePrefix(prefix) || |ValidNextTokens(prefix)| > 0)
  }

  // ============================================================================
  // COMPOSITIONAL CONSTRAINED DECODING STRATEGIES (CSD)
  // ============================================================================
  // 
  // Layer 1: Token-Level Constraints
  // Layer 2: Sequence-Level Operations  
  // Layer 3: Loop-Level Orchestration
  //
  // These primitives can be composed by an LLM (Qwen) to form complete CSDs.
  // ============================================================================

  // ---------------------------------------------------------------------------
  // Layer 1: Token-Level Constraints
  // ---------------------------------------------------------------------------
  // Constraints applied at each decoding step to filter which tokens are allowed.

  datatype TokenConstraint =
    | GrammarMask                                        // Only tokens continuing valid parse
    | Lookahead(depth: nat)                              // Filter tokens leading to dead ends within depth steps
    | LengthBound(min: nat, max: nat)                    // Track length, force EOS at max, block EOS before min
    | BanTokens(banned: set<Token>)                      // Explicitly ban tokens
    | AllowOnlyTokens(allowed: set<Token>)               // Whitelist tokens
    | Intersect(a: TokenConstraint, b: TokenConstraint)  // Must pass both constraints
    | Union(a: TokenConstraint, b: TokenConstraint)      // Can pass either constraint
    | NoConstraint                                       // Allow all tokens

  // AllowedNext: compute the set of allowed next tokens given a constraint and prefix
  function {:extern} {:axiom} AllowedNext(c: TokenConstraint, parser: Parser, prefix: Prefix, allTokens: set<Token>): set<Token>
    reads parser
    // GrammarMask ensures tokens continue a valid parse
    ensures c.GrammarMask? ==> 
      (forall t :: t in AllowedNext(c, parser, prefix, allTokens) ==> 
        (parser.IsValidPrefix(prefix) ==> parser.IsValidPrefix(prefix + [t])))
    // Lookahead ensures tokens continue a valid parse AND avoid dead ends
    ensures c.Lookahead? ==> 
      (forall t :: t in AllowedNext(c, parser, prefix, allTokens) ==> 
        (parser.IsValidPrefix(prefix) ==> parser.IsValidPrefix(prefix + [t])))
    // BanTokens removes banned tokens from consideration
    ensures c.BanTokens? ==> 
      AllowedNext(c, parser, prefix, allTokens) == allTokens - c.banned
    // AllowOnlyTokens restricts to whitelist
    ensures c.AllowOnlyTokens? ==> 
      AllowedNext(c, parser, prefix, allTokens) == allTokens * c.allowed
    // Intersect is set intersection
    ensures c.Intersect? ==> 
      AllowedNext(c, parser, prefix, allTokens) == 
        AllowedNext(c.a, parser, prefix, allTokens) * AllowedNext(c.b, parser, prefix, allTokens)
    // Union is set union
    ensures c.Union? ==> 
      AllowedNext(c, parser, prefix, allTokens) == 
        AllowedNext(c.a, parser, prefix, allTokens) + AllowedNext(c.b, parser, prefix, allTokens)
    // NoConstraint allows all tokens
    ensures c.NoConstraint? ==> 
      AllowedNext(c, parser, prefix, allTokens) == allTokens

  // ChooseToken: select the HIGHEST-LOGIT token from the allowed set
  // This is the key function that connects LM preferences to constrained decoding
  function {:extern} {:axiom} ChooseToken(lm: LM, c: TokenConstraint, parser: Parser, prefix: Prefix, allTokens: set<Token>): Token
    reads lm, parser
    requires lm.ValidTokensIdsLogits()
    requires |AllowedNext(c, parser, prefix, allTokens)| > 0
    // Basic: result is in the allowed set
    ensures ChooseToken(lm, c, parser, prefix, allTokens) in AllowedNext(c, parser, prefix, allTokens)
    // Result is in LM's vocabulary
    ensures ChooseToken(lm, c, parser, prefix, allTokens) in lm.Tokens
    // KEY: result has the MAXIMUM logit among all allowed tokens
    // This expresses: we pick the token the LM most prefers, subject to constraints
    ensures forall t :: t in AllowedNext(c, parser, prefix, allTokens) && t in lm.Tokens ==>
      lm.TokenToLogit(ChooseToken(lm, c, parser, prefix, allTokens)) >= lm.TokenToLogit(t)

  // ---------------------------------------------------------------------------
  // Layer 2: Sequence-Level Operations
  // ---------------------------------------------------------------------------
  // Operations on partial or complete outputs.

  // RepairRules: configuration for deterministic output repair
  datatype RepairRules =
    | BracketBalance      // Fix mismatched brackets/parens
    | QuoteFix            // Fix unclosed quotes
    | WhitespaceNormalize // Normalize whitespace
    | ComposedRepair(a: RepairRules, b: RepairRules)  // Apply multiple repairs
    | NoRepair            // Identity (no repair)

  // SemanticPredicate: abstract type for semantic validity checks (extern for compilation)
  type {:extern} SemanticPredicate(==)

  // SeqOperation: operations that transform or validate complete/partial outputs
  datatype SeqOperation =
    | Repair(rules: RepairRules)                        // Apply deterministic fixes
    | PrefixCompleteOp(constraint: TokenConstraint)     // Complete a valid prefix under constraint
    | ValidateOp(pred: SemanticPredicate)               // Check semantic validity
    | Identity                                          // No-op, return as-is

  // ApplyRepair: apply repair rules to an output
  function {:extern} {:axiom} ApplyRepair(rules: RepairRules, output: Prefix): Prefix
    // Repair doesn't increase length by more than a small constant
    ensures |ApplyRepair(rules, output)| <= |output| + 10
    // Repair preserves at least some of the original content
    ensures |ApplyRepair(rules, output)| >= 1 || |output| == 0

  // CheckSemantic: evaluate a semantic predicate on an output
  predicate {:extern} {:axiom} CheckSemantic(pred: SemanticPredicate, output: Prefix)

  // GreedyOptimal: predicate expressing that a sequence was generated greedily
  // by always picking the highest-logit token from the allowed set at each step
  predicate GreedyOptimal(lm: LM, parser: Parser, constraint: TokenConstraint, 
                          prefix: Prefix, output: Prefix, allTokens: set<Token>)
    reads lm, parser
    requires lm.ValidTokensIdsLogits()
  {
    // Output extends prefix
    |output| >= |prefix| &&
    // Each generated token was the greedy choice at that step
    forall i :: |prefix| <= i < |output| ==>
      (output[i] in lm.Tokens &&
       output[i] in AllowedNext(constraint, parser, output[..i], allTokens) &&
       // It was the max-logit choice
       forall t :: t in AllowedNext(constraint, parser, output[..i], allTokens) && t in lm.Tokens ==>
         lm.TokenToLogit(output[i]) >= lm.TokenToLogit(t))
  }

  // CompletePrefixConstrained: complete a valid prefix under a token constraint using LM
  // Uses GREEDY decoding - always picks the highest-logit token from the allowed set
  function {:extern} {:axiom} CompletePrefixConstrained(
      lm: LM, parser: Parser, prefix: Prefix, constraint: TokenConstraint, allTokens: set<Token>, maxSteps: nat): Prefix
    reads lm, parser
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix(prefix)
    // Key guarantee: output is still valid
    ensures parser.IsValidPrefix(CompletePrefixConstrained(lm, parser, prefix, constraint, allTokens, maxSteps))
    // Output extends the prefix
    ensures |CompletePrefixConstrained(lm, parser, prefix, constraint, allTokens, maxSteps)| >= |prefix|
    // Output is bounded
    ensures |CompletePrefixConstrained(lm, parser, prefix, constraint, allTokens, maxSteps)| <= |prefix| + maxSteps
    // All generated tokens are in LM vocabulary
    ensures forall i :: |prefix| <= i < |CompletePrefixConstrained(lm, parser, prefix, constraint, allTokens, maxSteps)| ==>
      CompletePrefixConstrained(lm, parser, prefix, constraint, allTokens, maxSteps)[i] in lm.Tokens
    // KEY: output was generated greedily according to LM logits
    ensures GreedyOptimal(lm, parser, constraint, prefix, 
                          CompletePrefixConstrained(lm, parser, prefix, constraint, allTokens, maxSteps), allTokens)

  // ApplySeqOp: apply a sequence operation using LM for completion
  function {:extern} {:axiom} ApplySeqOp(
      lm: LM, op: SeqOperation, parser: Parser, output: Prefix, allTokens: set<Token>, maxSteps: nat): Prefix
    reads lm, parser
    requires lm.ValidTokensIdsLogits()
    // Identity returns unchanged
    ensures op.Identity? ==> ApplySeqOp(lm, op, parser, output, allTokens, maxSteps) == output
    // Repair applies repair rules
    ensures op.Repair? ==> ApplySeqOp(lm, op, parser, output, allTokens, maxSteps) == ApplyRepair(op.rules, output)
    // PrefixComplete maintains validity if input was valid
    ensures op.PrefixCompleteOp? && parser.IsValidPrefix(output) ==> 
      parser.IsValidPrefix(ApplySeqOp(lm, op, parser, output, allTokens, maxSteps))

  // ---------------------------------------------------------------------------
  // Layer 3: Loop-Level Orchestration
  // ---------------------------------------------------------------------------
  // Strategies for orchestrating entire generation attempts.

  // CheckPredicate: what to check to determine if an output is acceptable
  datatype CheckPredicate =
    | ParseOk                                            // Output parses under grammar
    | SemanticOk(pred: SemanticPredicate)                // Custom semantic check
    | Both(a: CheckPredicate, b: CheckPredicate)         // Must pass both checks
    | Either(a: CheckPredicate, b: CheckPredicate)       // Must pass at least one

  // Attempt: a single generation attempt
  datatype Attempt =
    | Unconstrained                                      // Free LLM generation (no constraints)
    | ConstrainedAttempt(constraint: TokenConstraint)    // Constrained generation
    | WithRepair(base: Attempt, rules: RepairRules)      // Attempt + repair on output
    | WithSeqOp(base: Attempt, op: SeqOperation)         // Attempt + sequence operation

  // Strategy: loop-level orchestration of attempts
  datatype Strategy =
    // CRANE-style windowing: unconstrained outside delimiters, constrained inside
    | Window(startDelim: Token, endDelim: Token, inside: TokenConstraint, outside: TokenConstraint)
    
    // Retry: try attempt up to k times, then fallback
    | TryK(k: nat, attempt: Attempt, check: CheckPredicate, fallback: Strategy)
    
    // Cascade: try strategies in order until one succeeds
    | Cascade(strategies: seq<Strategy>, check: CheckPredicate)
    
    // Best-of-N: generate n outputs, pick first valid (or highest scoring if multiple valid)
    | BestOfN(n: nat, base: Strategy, check: CheckPredicate)
    
    // Terminal: just run constrained decode with given constraint
    | Constrained(constraint: TokenConstraint)
    
    // Unconstrained terminal: just run free generation
    | Free

  // CheckOutput: evaluate a check predicate on an output
  predicate {:extern} {:axiom} CheckOutput(check: CheckPredicate, parser: Parser, output: Prefix)
    reads parser
    // ParseOk means the parser accepts it
    ensures check.ParseOk? ==> (CheckOutput(check, parser, output) <==> parser.IsValidPrefix(output))
    // SemanticOk delegates to CheckSemantic
    ensures check.SemanticOk? ==> (CheckOutput(check, parser, output) <==> CheckSemantic(check.pred, output))
    // Both requires both to pass
    ensures check.Both? ==> (CheckOutput(check, parser, output) <==> 
      (CheckOutput(check.a, parser, output) && CheckOutput(check.b, parser, output)))
    // Either requires at least one to pass
    ensures check.Either? ==> (CheckOutput(check, parser, output) <==> 
      (CheckOutput(check.a, parser, output) || CheckOutput(check.b, parser, output)))

  // RunAttempt: execute a single attempt using LM and return the output
  function {:extern} {:axiom} RunAttempt(
      lm: LM, attempt: Attempt, parser: Parser, prompt: Prefix, allTokens: set<Token>, maxSteps: nat): Prefix
    reads lm, parser
    requires lm.ValidTokensIdsLogits()
    // Constrained attempts produce valid output
    ensures attempt.ConstrainedAttempt? ==> parser.IsValidPrefix(RunAttempt(lm, attempt, parser, prompt, allTokens, maxSteps))
    // Output is bounded in length
    ensures |RunAttempt(lm, attempt, parser, prompt, allTokens, maxSteps)| <= |prompt| + maxSteps
    // All generated tokens are from LM vocabulary
    ensures forall i :: |prompt| <= i < |RunAttempt(lm, attempt, parser, prompt, allTokens, maxSteps)| ==>
      RunAttempt(lm, attempt, parser, prompt, allTokens, maxSteps)[i] in lm.Tokens
    // KEY: Constrained attempts are GREEDY OPTIMAL - they use the LM's preferences
    ensures attempt.ConstrainedAttempt? && parser.IsValidPrefix(prompt) ==> 
      GreedyOptimal(lm, parser, attempt.constraint, prompt, 
                    RunAttempt(lm, attempt, parser, prompt, allTokens, maxSteps), allTokens)

  // RunStrategy: execute a strategy using LM and return the output
  // This is the main entry point for running a CSD
  function {:extern} {:axiom} RunStrategy(
      lm: LM, strategy: Strategy, parser: Parser, prompt: Prefix, allTokens: set<Token>, maxSteps: nat): Prefix
    reads lm, parser
    requires lm.ValidTokensIdsLogits()
    // Constrained strategy always produces valid output
    ensures strategy.Constrained? ==> parser.IsValidPrefix(RunStrategy(lm, strategy, parser, prompt, allTokens, maxSteps))
    // Window strategy produces valid output (CRANE guarantee)
    ensures strategy.Window? ==> parser.IsValidPrefix(RunStrategy(lm, strategy, parser, prompt, allTokens, maxSteps))
    // Output is bounded
    ensures |RunStrategy(lm, strategy, parser, prompt, allTokens, maxSteps)| <= |prompt| + maxSteps
    // TryK with Constrained fallback guarantees valid output
    ensures strategy.TryK? && strategy.fallback.Constrained? ==> 
      parser.IsValidPrefix(RunStrategy(lm, strategy, parser, prompt, allTokens, maxSteps))
    // Cascade ending in Constrained guarantees valid output
    ensures strategy.Cascade? && |strategy.strategies| > 0 && strategy.strategies[|strategy.strategies| - 1].Constrained? ==>
      parser.IsValidPrefix(RunStrategy(lm, strategy, parser, prompt, allTokens, maxSteps))
    // All generated tokens are from LM vocabulary
    ensures forall i :: |prompt| <= i < |RunStrategy(lm, strategy, parser, prompt, allTokens, maxSteps)| ==>
      RunStrategy(lm, strategy, parser, prompt, allTokens, maxSteps)[i] in lm.Tokens
    // KEY: Constrained strategies use GREEDY decoding according to LM preferences
    ensures strategy.Constrained? && parser.IsValidPrefix(prompt) ==>
      GreedyOptimal(lm, parser, strategy.constraint, prompt,
                    RunStrategy(lm, strategy, parser, prompt, allTokens, maxSteps), allTokens)

  // ---------------------------------------------------------------------------
  // Convenience constructors for common patterns
  // ---------------------------------------------------------------------------

  // CRANE-style: unconstrained reasoning with constrained answer windows
  function CRANEStyle(startDelim: Token, endDelim: Token): Strategy
  {
    Window(startDelim, endDelim, GrammarMask, NoConstraint)
  }

  // Retry K times unconstrained, then fall back to constrained
  function RetryThenConstrained(k: nat): Strategy
  {
    TryK(k, Unconstrained, ParseOk, Constrained(GrammarMask))
  }

  // Best-of-N with repair, falling back to constrained
  function BestOfNWithRepair(n: nat, repairRules: RepairRules): Strategy
  {
    BestOfN(n, 
      TryK(1, WithRepair(Unconstrained, repairRules), ParseOk, Constrained(GrammarMask)),
      ParseOk)
  }

  // ---------------------------------------------------------------------------
  // Strategy Validity Predicates
  // ---------------------------------------------------------------------------
  // These predicates determine whether a strategy guarantees valid output.

  // GuaranteesValidOutput: true if the strategy always produces parser-valid output
  predicate GuaranteesValidOutput(strategy: Strategy)
  {
    match strategy
    case Constrained(_) => true
    case Window(_, _, _, _) => true  // CRANE guarantees valid constrained windows
    case TryK(_, _, _, fallback) => GuaranteesValidOutput(fallback)
    case Cascade(strategies, _) => 
      |strategies| > 0 && GuaranteesValidOutput(strategies[|strategies| - 1])
    case BestOfN(_, base, _) => GuaranteesValidOutput(base)
    case Free => false
  }

  // ---------------------------------------------------------------------------
  // Main Entry Point: Run
  // ---------------------------------------------------------------------------
  // The primary function for executing a CSD strategy.

  // Run: execute a strategy that guarantees valid output using LM
  // This is the function that Qwen-generated code should ultimately call
  function {:extern} {:axiom} Run(
      lm: LM, strategy: Strategy, parser: Parser, prompt: Prefix, allTokens: set<Token>, maxSteps: nat): Prefix
    reads lm, parser
    requires lm.ValidTokensIdsLogits()
    requires GuaranteesValidOutput(strategy)
    requires |prompt| > 0
    // THE KEY GUARANTEE: output is always valid under the parser
    ensures parser.IsValidPrefix(Run(lm, strategy, parser, prompt, allTokens, maxSteps))
    // Output is bounded in length
    ensures |Run(lm, strategy, parser, prompt, allTokens, maxSteps)| <= |prompt| + maxSteps
    // Output extends the prompt
    ensures |Run(lm, strategy, parser, prompt, allTokens, maxSteps)| >= |prompt|
    // All generated tokens are from LM vocabulary
    ensures forall i :: |prompt| <= i < |Run(lm, strategy, parser, prompt, allTokens, maxSteps)| ==>
      Run(lm, strategy, parser, prompt, allTokens, maxSteps)[i] in lm.Tokens

  // ---------------------------------------------------------------------------
  // Lemmas for Strategy Composition
  // ---------------------------------------------------------------------------

  // Lemma: TryK with valid fallback guarantees valid output
  lemma TryKGuaranteesValid(k: nat, attempt: Attempt, check: CheckPredicate, fallback: Strategy)
    requires GuaranteesValidOutput(fallback)
    ensures GuaranteesValidOutput(TryK(k, attempt, check, fallback))
  {
    // Follows from definition of GuaranteesValidOutput
  }

  // Lemma: Cascade ending in valid strategy guarantees valid output
  lemma CascadeGuaranteesValid(strategies: seq<Strategy>, check: CheckPredicate)
    requires |strategies| > 0
    requires GuaranteesValidOutput(strategies[|strategies| - 1])
    ensures GuaranteesValidOutput(Cascade(strategies, check))
  {
    // Follows from definition of GuaranteesValidOutput
  }

  // Lemma: Constrained always guarantees valid output
  lemma ConstrainedGuaranteesValid(constraint: TokenConstraint)
    ensures GuaranteesValidOutput(Constrained(constraint))
  {
    // Immediate from definition
  }

  // Lemma: Window (CRANE) always guarantees valid output
  lemma WindowGuaranteesValid(startDelim: Token, endDelim: Token, inside: TokenConstraint, outside: TokenConstraint)
    ensures GuaranteesValidOutput(Window(startDelim, endDelim, inside, outside))
  {
    // CRANE's key property: constrained windows are always valid
  }

  method {:extern} {:axiom} ConstrainedDecode(lm: LM, parser: Parser, prefix: Prefix, maxSteps: nat) returns (result: Prefix)
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix(prefix)
    requires maxSteps >= 0
    requires |prefix| > 0
    ensures parser.IsValidPrefix(result)
    ensures |result| >= |prefix|
    ensures prefix == result[..|prefix|]
    ensures |result| <= |prefix| + maxSteps
    ensures parser.IsCompletePrefix(result) || |result| == |prefix| + maxSteps || (result[|result| - 1] in lm.Tokens && lm.TokenToLogit(result[|result| - 1]) != 0.0)
    ensures forall k :: |prefix| <= k < |result| ==> (parser.IsValidPrefix(result[..k])) && (result[k] in parser.ValidNextTokens(result[..k]))
    ensures lm.ValidTokensIdsLogits()
}