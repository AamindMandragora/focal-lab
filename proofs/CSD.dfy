// Compositional constrained-window decoding policies (CSDs).
//
// This module is meant to be a small Dafny “algebra” of policies that an LLM
// can compose using library combinators, rather than generating bespoke
// recursive decoding logic.

module Decoding {
  type Token = string
  type Prefix = seq<Token>
  type Logits = map<Token, real>

  // Token universe / vocabulary.
  function {:extern} {:axiom} Vocabulary(): set<Token>

  // --- Grammar-facing interface (abstract; implemented externally) ---
  function {:extern} {:axiom} Parser_ValidPrefix(prefix: Prefix): bool
  function {:extern} {:axiom} Parser_IsComplete(prefix: Prefix): bool
    ensures Parser_IsComplete(prefix) ==> Parser_ValidPrefix(prefix)

  function {:extern} {:axiom} Parser_AllowedNext(prefix: Prefix): set<Token>
    ensures forall t :: t in Parser_AllowedNext(prefix) ==> Parser_ValidPrefix(prefix + [t])
    ensures Parser_ValidPrefix(prefix) ==> (Parser_IsComplete(prefix) || |Parser_AllowedNext(prefix)| > 0)

  // --- LLM-facing interface (abstract; implemented externally) ---
  function {:extern} {:axiom} GetLogits(prefix: Prefix): Logits
    ensures forall t :: t is Token ==> t in GetLogits(prefix)

  predicate ValidLogitsForPrefix(logits: Logits, prefix: Prefix)
  {
    forall t :: t in Parser_AllowedNext(prefix) ==> t in logits
  }

  // Optional utility that represents grammar masking (e.g., SynCode-style).
  function {:extern} {:axiom} MaskLogits(prefix: Prefix, logits: Logits): Logits
    requires forall t :: t is Token ==> t in logits
    ensures forall t :: t in logits ==> t in MaskLogits(prefix, logits)
    ensures forall t :: t in Parser_AllowedNext(prefix) ==> MaskLogits(prefix, logits)[t] == logits[t]
    ensures forall t :: t !in Parser_AllowedNext(prefix) ==> MaskLogits(prefix, logits)[t] == 0.0

  // Local-step semantic acceptance predicate for rejection-style filtering.
  predicate {:extern} {:axiom} Accept(prefix: Prefix, tok: Token)

  // A generic sampler used by external implementations.
  datatype ProposalStrategy =
    | ArgMax
    | Temperature
    | TopK
    | Nucleus

  function {:extern} {:axiom} SampleWithStrategy(
    logits: Logits,
    candidates: set<Token>,
    strategy: ProposalStrategy
  ): Token
    requires |candidates| > 0
    requires forall t :: t in candidates ==> t in logits
    ensures SampleWithStrategy(logits, candidates, strategy) in candidates
}

module CSD {
  import opened Decoding

  // A constrained-decoding *strategy component*.
  //
  // Whole-loop CSD programs (below) can reference these as building blocks.
  datatype Policy =
    // Start from an unconstrained proposal strategy (external sampler).
    | Base(strategy: ProposalStrategy)
    // Apply a grammar-alignment mask (SynCode-like).
    | MaskWithSynCode(base: Policy)
    // Apply a grammar-alignment mask (CRANE-style “mask logits” view).
    | MaskWithCRANE(base: Policy)
    // Apply an abstract semantic rejection filter.
    | WithRejection(base: Policy)
    // Priority fallback: use primary if it has candidates, else backup.
    | Fallback(primary: Policy, backup: Policy)
    // Set combinators (useful for composition experiments).
    | Intersect(a: Policy, b: Policy)
    | Union(a: Policy, b: Policy)

  // Opaque “initialized” state (e.g., precomputed mask stores).
  datatype State = State

  type GrammarLike = int

  function {:extern} {:axiom} Initialize(policy: Policy, Gprime: GrammarLike): State

  // Denotational semantics for policy -> allowed next-token set.
  function AllowedNext(policy: Policy, st: State, prefix: Prefix): set<Token>
  {
    match policy
    case Base(_) =>
      Vocabulary()

    case MaskWithSynCode(base) =>
      AllowedNext(base, st, prefix) * Parser_AllowedNext(prefix)

    case MaskWithCRANE(base) =>
      AllowedNext(base, st, prefix) * Parser_AllowedNext(prefix)

    case WithRejection(base) =>
      set t: Token | t in AllowedNext(base, st, prefix) && Accept(prefix, t)

    case Fallback(primary, backup) =>
      if |AllowedNext(primary, st, prefix)| > 0 then
        AllowedNext(primary, st, prefix)
      else
        AllowedNext(backup, st, prefix)

    case Intersect(a, b) =>
      AllowedNext(a, st, prefix) * AllowedNext(b, st, prefix)

    case Union(a, b) =>
      AllowedNext(a, st, prefix) + AllowedNext(b, st, prefix)
  }

  // Policy “capability” markers used by downstream proofs.
  function IsGrammarAligned(policy: Policy): bool
  {
    match policy
    case MaskWithSynCode(_) => true
    case MaskWithCRANE(_) => true
    case WithRejection(b) => IsGrammarAligned(b)
    case Fallback(p, b) => IsGrammarAligned(p) && IsGrammarAligned(b)
    case Intersect(a, b) => IsGrammarAligned(a) && IsGrammarAligned(b)
    // Union can introduce tokens that violate grammar alignment.
    case Union(_, _) => false
    case Base(_) => false
  }

  function IsSemFiltered(policy: Policy): bool
  {
    match policy
    case WithRejection(_) => true
    case MaskWithSynCode(b) => IsSemFiltered(b)
    case MaskWithCRANE(b) => IsSemFiltered(b)
    case Fallback(p, b) => IsSemFiltered(p) && IsSemFiltered(b)
    case Intersect(a, b) => IsSemFiltered(a) || IsSemFiltered(b)
    case Union(a, b) => IsSemFiltered(a) || IsSemFiltered(b)
    case Base(_) => false
  }

  // Token choice is external/probabilistic, so we axiomatically specify it.
  function {:extern} {:axiom} ChooseToken(policy: Policy, st: State, prefix: Prefix): Token
    requires |AllowedNext(policy, st, prefix)| > 0
    ensures ChooseToken(policy, st, prefix) in AllowedNext(policy, st, prefix)
    ensures IsGrammarAligned(policy) ==> ChooseToken(policy, st, prefix) in Parser_AllowedNext(prefix)
    ensures IsSemFiltered(policy) ==> Accept(prefix, ChooseToken(policy, st, prefix))

  // -------------------------------------------------------------------------
  // Qwen “surface language” (generation restrictions)
  //
  // The intent is that the LLM generates Dafny code that only *builds values* of
  // type Policy by composing the following constructors, rather than generating
  // bespoke decoding loops/recursion:
  //   - Base(strategy)
  //   - MaskWithSynCode(p)
  //   - MaskWithCRANE(p)
  //   - WithRejection(p)
  //   - Fallback(p, q)
  //   - Intersect(p, q)
  //
  // And it should *not* use Union (it may break grammar-alignment), nor define
  // new recursive functions/methods/loops. This is a spec-level convention;
  // we encode it as a ghost predicate so proofs can require it explicitly.
  // -------------------------------------------------------------------------

  ghost predicate QwenAllowed(policy: Policy)
    decreases policy
  {
    match policy
    case Base(_) => true
    case MaskWithSynCode(b) => QwenAllowed(b)
    case MaskWithCRANE(b) => QwenAllowed(b)
    case WithRejection(b) => QwenAllowed(b)
    case Fallback(p, b) => QwenAllowed(p) && QwenAllowed(b)
    case Intersect(a, b) => QwenAllowed(a) && QwenAllowed(b)
    case Union(_, _) => false
  }

  ghost predicate QwenSafeForWindows(policy: Policy)
  {
    QwenAllowed(policy) && IsGrammarAligned(policy) && IsSemFiltered(policy)
  }

  // -------------------------------------------------------------------------
  // Whole-loop CSD programs
  //
  // This is the level that matches your pseudocode:
  //   - try unconstrained generation up to K times
  //   - if parse[G](s) succeeds, return s
  //   - else fall back to constrained decoding under G
  //
  // We intentionally keep the *parse check* abstract to accommodate either:
  //   - full acceptance (s ∈ L(G))   [often corresponds to Parser_IsComplete]
  //   - prefix acceptance (s ∈ Lp(G)) [often corresponds to Parser_ValidPrefix]
  // depending on how you decide to interpret parse[G](s) for your task.
  // -------------------------------------------------------------------------

  // Unconstrained generation of a whole candidate string/sequence.
  function {:extern} {:axiom} LLM_Generate(prompt: Prefix, maxSteps: nat, strategy: ProposalStrategy): Prefix
    ensures |LLM_Generate(prompt, maxSteps, strategy)| >= |prompt|
    ensures prompt == LLM_Generate(prompt, maxSteps, strategy)[..|prompt|]
    ensures |LLM_Generate(prompt, maxSteps, strategy)| <= |prompt| + maxSteps

  // Parse predicate for a candidate under grammar handle g.
  predicate {:extern} {:axiom} ParseOk(g: GrammarLike, s: Prefix)

  // Optional: prefix parsing notion (s ∈ L_p(G)). Used for “prefix-guided completion”.
  predicate {:extern} {:axiom} ParseOkPrefix(g: GrammarLike, s: Prefix)

  // Constrained decoding fallback (a “constrained decoding strategy” component).
  //
  // Note: This is a library call; runtime can realize it as SynCode, CRANE-style
  // masking, etc., as determined by `policy` and the grammar `g`.
  function {:extern} {:axiom} ConstrainedGenerate(g: GrammarLike, policy: Policy, prompt: Prefix, maxSteps: nat): Prefix
    ensures |ConstrainedGenerate(g, policy, prompt, maxSteps)| >= |prompt|
    ensures prompt == ConstrainedGenerate(g, policy, prompt, maxSteps)[..|prompt|]
    ensures |ConstrainedGenerate(g, policy, prompt, maxSteps)| <= |prompt| + maxSteps
    ensures ParseOk(g, ConstrainedGenerate(g, policy, prompt, maxSteps))

  // Complete a *parse-ok prefix* into a parse-ok full output via constrained decoding.
  function {:extern} {:axiom} ConstrainedCompleteFromPrefix(g: GrammarLike, policy: Policy, prefix: Prefix, maxSteps: nat): Prefix
    ensures |ConstrainedCompleteFromPrefix(g, policy, prefix, maxSteps)| >= |prefix|
    ensures prefix == ConstrainedCompleteFromPrefix(g, policy, prefix, maxSteps)[..|prefix|]
    ensures |ConstrainedCompleteFromPrefix(g, policy, prefix, maxSteps)| <= |prefix| + maxSteps
    ensures ParseOk(g, ConstrainedCompleteFromPrefix(g, policy, prefix, maxSteps))

  // A single whole-output attempt. Attempts may be unconstrained or constrained.
  // Programs will combine attempts by checking ParseOk and falling back.
  datatype Attempt =
    | Unconstrained(strategy: ProposalStrategy)
    | Constrained(policy: Policy)
    // Apply a deterministic “repair” transform after producing a candidate.
    | Repair(base: Attempt)
    // A more expensive constrained search attempt (e.g., beam/DFS). Abstracted as a library call.
    | ConstrainedSearch(policy: Policy, beamWidth: nat)

  // A repair transform (e.g., whitespace normalization, delimiter balancing).
  function {:extern} {:axiom} RepairTransform(s: Prefix): Prefix

  function RunAttempt(g: GrammarLike, attempt: Attempt, prompt: Prefix, maxSteps: nat): Prefix
    // Constrained attempts must always produce a parse-ok output.
    ensures attempt.Constrained? ==> ParseOk(g, RunAttempt(g, attempt, prompt, maxSteps))
    ensures attempt.ConstrainedSearch? ==> ParseOk(g, RunAttempt(g, attempt, prompt, maxSteps))
  {
    match attempt
    case Unconstrained(strategy) =>
      LLM_Generate(prompt, maxSteps, strategy)
    case Constrained(policy) =>
      ConstrainedGenerate(g, policy, prompt, maxSteps)
    case Repair(base) =>
      RepairTransform(RunAttempt(g, base, prompt, maxSteps))
    case ConstrainedSearch(policy, beamWidth) =>
      ConstrainedSearchGenerate(g, policy, beamWidth, prompt, maxSteps)
  }

  // Constrained “search mode” library call (beam/DFS/etc.).
  function {:extern} {:axiom} ConstrainedSearchGenerate(
    g: GrammarLike,
    policy: Policy,
    beamWidth: nat,
    prompt: Prefix,
    maxSteps: nat
  ): Prefix
    ensures |ConstrainedSearchGenerate(g, policy, beamWidth, prompt, maxSteps)| >= |prompt|
    ensures prompt == ConstrainedSearchGenerate(g, policy, beamWidth, prompt, maxSteps)[..|prompt|]
    ensures |ConstrainedSearchGenerate(g, policy, beamWidth, prompt, maxSteps)| <= |prompt| + maxSteps
    ensures ParseOk(g, ConstrainedSearchGenerate(g, policy, beamWidth, prompt, maxSteps))

  // Optional whole-string semantic validator (separate from ParseOk).
  predicate {:extern} {:axiom} SemanticOk(g: GrammarLike, s: Prefix)

  datatype Check =
    | ParseOnly
    | ParseAndSemantic

  predicate Passes(check: Check, g: GrammarLike, s: Prefix)
  {
    match check
    case ParseOnly => ParseOk(g, s)
    case ParseAndSemantic => ParseOk(g, s) && SemanticOk(g, s)
  }

  // A whole-loop CSD program.
  //
  // Key idea: Program is a *decision chain* of attempts. Each link:
  //   - runs an attempt to get a candidate s
  //   - if ParseOk(g,s), returns s
  //   - else continues with the next program
  //
  // This generalizes "retry K then fallback" but does not force a loop:
  // Qwen can generate any finite chain of attempts (including constrained ones).
  //
  // Expanded surface:
  // - TryThenElse: one attempt + check + fallback
  // - TryK: bounded repetition without unrolling
  // - BestOfNThenElse: generate N candidates, select one that passes, else fallback
  // - CompleteIfPrefixOkElse: prefix-guided completion via constrained decode
  // - ReturnParsed: final backstop producing ParseOk
  datatype Program =
    | TryThenElse(g: GrammarLike, attempt: Attempt, check: Check, onFail: Program)
    | TryK(g: GrammarLike, k: nat, attempt: Attempt, check: Check, onFail: Program)
    | BestOfNThenElse(g: GrammarLike, n: nat, strategy: ProposalStrategy, check: Check, onFail: Program)
    | CompleteIfPrefixOkElse(g: GrammarLike, strategy: ProposalStrategy, policy: Policy, onFail: Program)
    | ReturnParsed(g: GrammarLike, policy: Policy)

  function Size(program: Program): nat
    decreases program
  {
    match program
    case ReturnParsed(_, _) => 1
    case TryThenElse(_, _, _, onFail) => 1 + Size(onFail)
    case TryK(_, _, _, _, onFail) => 1 + Size(onFail)
    case BestOfNThenElse(_, _, _, _, onFail) => 1 + Size(onFail)
    case CompleteIfPrefixOkElse(_, _, _, onFail) => 1 + Size(onFail)
  }

  // Grammar-consistency invariant: all nodes in the chain must share the same g.
  ghost predicate GrammarConsistent(program: Program)
    decreases program
  {
    match program
    case ReturnParsed(_, _) => true
    case TryThenElse(g, _, _, onFail) =>
      GrammarConsistent(onFail) && onFail.g == g
    case TryK(g, _, _, _, onFail) =>
      GrammarConsistent(onFail) && onFail.g == g
    case BestOfNThenElse(g, _, _, _, onFail) =>
      GrammarConsistent(onFail) && onFail.g == g
    case CompleteIfPrefixOkElse(g, _, _, onFail) =>
      GrammarConsistent(onFail) && onFail.g == g
  }

  function Run(program: Program, prompt: Prefix, maxSteps: nat): Prefix
    decreases maxSteps, Size(program), (if program.TryK? then program.k else 0)
    ensures ParseOk(program.g, Run(program, prompt, maxSteps))
    requires GrammarConsistent(program)
  {
    match program
    case ReturnParsed(g, policy) =>
      ConstrainedGenerate(g, policy, prompt, maxSteps)

    case CompleteIfPrefixOkElse(g, strategy, policy, onFail) =>
      var s := LLM_Generate(prompt, maxSteps, strategy);
      if ParseOkPrefix(g, s) then
        ConstrainedCompleteFromPrefix(g, policy, s, maxSteps)
      else
        Run(onFail, prompt, maxSteps)

    case TryThenElse(g, attempt, check, onFail) =>
      var s := RunAttempt(g, attempt, prompt, maxSteps);
      if Passes(check, g, s) then
        s
      else
        Run(onFail, prompt, maxSteps)

    case TryK(g, k, attempt, check, onFail) =>
      if k == 0 then
        Run(onFail, prompt, maxSteps)
      else
        var s := RunAttempt(g, attempt, prompt, maxSteps);
        if Passes(check, g, s) then
          s
        else
          Run(TryK(g, k - 1, attempt, check, onFail), prompt, maxSteps)

    case BestOfNThenElse(g, n, strategy, check, onFail) =>
      var r := BestOfNSelectPassing(g, n, strategy, check, prompt, maxSteps);
      if r.found then
        r.s
      else
        Run(onFail, prompt, maxSteps)
  }

  datatype SelectResult = Select(found: bool, s: Prefix)

  // Best-of-N: generate N unconstrained candidates, select one that passes check if possible.
  function {:extern} {:axiom} BestOfNSelectPassing(
    g: GrammarLike,
    n: nat,
    strategy: ProposalStrategy,
    check: Check,
    prompt: Prefix,
    maxSteps: nat
  ): SelectResult
    ensures BestOfNSelectPassing(g, n, strategy, check, prompt, maxSteps).s == prompt
      || (|BestOfNSelectPassing(g, n, strategy, check, prompt, maxSteps).s| >= |prompt|
          && prompt == BestOfNSelectPassing(g, n, strategy, check, prompt, maxSteps).s[..|prompt|]
          && |BestOfNSelectPassing(g, n, strategy, check, prompt, maxSteps).s| <= |prompt| + maxSteps)
    ensures BestOfNSelectPassing(g, n, strategy, check, prompt, maxSteps).found
      ==> Passes(check, g, BestOfNSelectPassing(g, n, strategy, check, prompt, maxSteps).s)

  // Qwen surface for whole-loop programs: it should only build Programs/Attempts
  // from these constructors, and Policies from the policy-level surface.
  ghost predicate QwenAllowedAttempt(attempt: Attempt)
    decreases attempt
  {
    match attempt
    case Unconstrained(_) => true
    case Constrained(p) => QwenAllowed(p)
    case Repair(b) => QwenAllowedAttempt(b)
    case ConstrainedSearch(p, _) => QwenAllowed(p)
  }

  ghost predicate QwenAllowedProgram(program: Program)
    decreases program
  {
    match program
    case ReturnParsed(_, policy) => QwenAllowed(policy)
    case CompleteIfPrefixOkElse(_, _, policy, onFail) => QwenAllowed(policy) && QwenAllowedProgram(onFail)
    case TryThenElse(_, attempt, _, onFail) => QwenAllowedAttempt(attempt) && QwenAllowedProgram(onFail)
    case TryK(_, _, attempt, _, onFail) => QwenAllowedAttempt(attempt) && QwenAllowedProgram(onFail)
    case BestOfNThenElse(_, _, _, _, onFail) => QwenAllowedProgram(onFail)
  }
}


