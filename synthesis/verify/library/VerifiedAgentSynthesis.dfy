module VerifiedDecoderAgent {
  type Token = string
  type Prefix = seq<Token>
  type Id = nat
  type Logit = real

  class LM {
    const Tokens: seq<Token>
    const Ids: seq<Id>
    var Logits: array<Logit>

    var useSampling: bool
    var stepCost: int

    method SetUseSampling(enabled: bool)
      modifies this
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()
      ensures Logits == old(Logits)
      ensures useSampling == enabled
    {
      useSampling := enabled;
    }

    predicate ValidTokensIdsLogits()
      reads this
      reads this.Logits
    {
      ((|Tokens| == |Ids|) && (|Ids| == Logits.Length) && (|Ids| > 0 && Ids[0] == 0) && (|Tokens| == |set t | t in Tokens|)) &&
      (forall i :: 0 <= i < |Ids| ==> (i == Ids[i]) && (i in Ids)) && 
      (forall i, j :: 0 <= i < |Tokens| && 0 <= j < |Tokens| && i != j ==> Tokens[i] != Tokens[j]) &&
      (forall token: Token :: token in Tokens ==> (exists i :: 0 <= i < |Ids| && Tokens[i] == token)) &&
      (forall i :: (0 <= i < Logits.Length) ==> (-1000000000.0 <= Logits[i] && Logits[i] <= 1000000000.0))
    }

    constructor {:extern} {:axiom} ()
      ensures ValidTokensIdsLogits()

    function IdToToken(id: Id) : (token: Token)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires id in Ids
      ensures token in Tokens
      ensures Tokens[id] == token
      ensures id == TokenToId(token)
      ensures ValidTokensIdsLogits()
    {
      Tokens[id]
    }

    function TokenToId(token: Token) : (id: Id)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures id in Ids
      ensures Tokens[id] == token
      ensures TokenToId(Tokens[id]) == id
      ensures ValidTokensIdsLogits()
    {
      TokenToIdRecursive(token, 0)
    }

    function TokenToIdRecursive(token: Token, offset: nat) : (id: Id)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      requires 0 <= offset < |Tokens|
      requires (Tokens[offset] == token) || (token in Tokens[offset + 1..])
      ensures id in Ids
      ensures 0 <= TokenToIdRecursive(token, offset) < |Ids|
      ensures Tokens[id] == token
      ensures ValidTokensIdsLogits()
      decreases |Tokens| - offset
    {
      if Tokens[offset] == token then offset
      else TokenToIdRecursive(token, offset + 1)
    }

    function IdToLogit(id: Id) : (logit: Logit)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires id in Ids
      ensures logit in Logits[0..Logits.Length]
      ensures ValidTokensIdsLogits()
    {
      Logits[id]
    }

    function TokenToLogit(token: Token): (logit: Logit)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
    {
      IdToLogit(TokenToId(token))
    }

    function TokensToLogits(tokens: seq<Token>): (logits: seq<Logit>)
      reads this
      reads this.Logits
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
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires |ids| > 0
      requires forall id: Id :: id in ids ==> id in Ids
      ensures ValidTokensIdsLogits()
    {
      if (|ids| == 1) then [IdToLogit(ids[0])]
      else [IdToLogit(ids[0])] + IdsToLogits(ids[1..])
    }

    method MaskToken(token: Token)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
      ensures Tokens[TokenToId(token)] == token
      ensures IsMasked(token)
      ensures forall t: Token :: t in Tokens && t != token ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
    {
      var id := TokenToId(token);
      Logits[id] := -1000000000.0;
    }

    method MaskTokens(tokens: seq<Token>)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires |tokens| > 0
      requires forall token :: token in tokens ==> token in Tokens
      ensures ValidTokensIdsLogits()
      ensures forall t :: t in tokens ==> IsMasked(t)
      ensures forall t :: t in Tokens && !(t in tokens) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
    {
      var N := |tokens|;
      var i := 0;
      while i < N
        invariant 0 <= i <= N
        invariant ValidTokensIdsLogits()
        invariant forall j :: 0 <= j < i ==> IsMasked(tokens[j])
        invariant forall t :: t in Tokens && !(t in tokens[..i]) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
        decreases N - i
      {
        MaskToken(tokens[i]);
        i := i + 1;
      }
    }

    method MaskTokensExcept(tokens: seq<Token>)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires |tokens| > 0
      requires forall token :: token in tokens ==> token in Tokens
      ensures ValidTokensIdsLogits()
      ensures forall t :: t in Tokens && !(t in tokens) ==> IsMasked(t)
      ensures forall t :: t in tokens ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
    {
      var toMask: seq<Token> := [];
      var N := |Tokens|;
      var i := 0;

      while i < N
        invariant 0 <= i <= N
        invariant ValidTokensIdsLogits()
        invariant forall j :: 0 <= j < i && !(Tokens[j] in tokens) ==> Tokens[j] in toMask
        invariant forall j :: 0 <= j < i && Tokens[j] in tokens ==> !(Tokens[j] in toMask)
        invariant forall t: Token :: t in toMask ==> t !in tokens && t in Tokens
        decreases N - i
      {
        if !(Tokens[i] in tokens) {
          toMask := toMask + [Tokens[i]];
        }
        i := i + 1;
      }

      if |toMask|> 0 {
        MaskTokens(toMask);
      }
    }

    predicate IsMasked(token: Token)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
    {
      Logits[TokenToId(token)] == -1000000000.0
    }

    predicate HasUnmaskedToken()
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()
    {
      exists t: Token :: t in Tokens && !IsMasked(t)
    }

    method {:extern} {:axiom} GenerateLogits(input: Prefix)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()

    method {:extern} {:axiom} AppendTaskGuidance(guidance: string)
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()

    // Persistently down-weight `token` as a next-token at position `prefix`, so a
    // later regeneration at this position (e.g. after a grounding rollback) picks
    // a DIFFERENT token instead of looping. Host-state only: does NOT change the
    // current logits, so the Logits-array contract is preserved. (Faithful analog
    // of IterGen's recurrence_penalty on a backtracked trace position.)
    method {:extern} {:axiom} PenalizeTriedTokenAt(prefix: Prefix, token: Token)
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()

    method {:extern} {:axiom} ChooseNextToken() returns (token: Token)
      requires ValidTokensIdsLogits()
      ensures token in Tokens
      ensures !IsMasked(token)
      ensures ValidTokensIdsLogits()

    method {:extern} {:axiom} ChooseNextTokenUnconstrained() returns (token: Token)
      ensures token in Tokens
      ensures ValidTokensIdsLogits()

    method {:extern} {:axiom} GenerateUnconstrainedChunk(
      input: Prefix, maxNewTokens: nat, openSpanToken: Token, eosToken: Token
    ) returns (chunk: Prefix, stoppedOnOpenSpan: bool, stoppedOnEos: bool, stepsUsed: nat)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires openSpanToken in Tokens
      requires eosToken in Tokens
      ensures ValidTokensIdsLogits()
      ensures |chunk| <= stepsUsed <= maxNewTokens
      ensures forall i :: 0 <= i < |chunk| ==> chunk[i] in Tokens && chunk[i] != eosToken
      ensures !(stoppedOnOpenSpan && stoppedOnEos)
      ensures stoppedOnOpenSpan ==> |chunk| > 0 && chunk[|chunk| - 1] == openSpanToken
      ensures stoppedOnEos ==> stepsUsed == |chunk| + 1
      ensures !stoppedOnEos ==> stepsUsed == |chunk|
      ensures maxNewTokens > 0 ==> stepsUsed > 0

    method {:extern} {:axiom} MaskValidNextAndEos(parser: Parser, prefix: Prefix, eosToken: Token)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires eosToken in Tokens
      ensures ValidTokensIdsLogits()
      ensures forall t :: t in Tokens && !parser.ValidNextToken(prefix, t) && t != eosToken ==> IsMasked(t)
      ensures this.Logits == old(this.Logits)

    method {:extern} {:axiom} BoostValidNextAndEos(parser: Parser, prefix: Prefix, amount: real, eosToken: Token)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires eosToken in Tokens
      requires amount >= 0.0 && amount <= 100000000.0
      ensures ValidTokensIdsLogits()

    // Grounding predicate. Returns whether every identifier-like token in `text`
    // appears in the support set the host derives from the task input (the
    // prompt); returns true when the prompt contains no recognizable support set.
    // Pure with respect to the Dafny heap (reads nothing): the support set lives
    // in host state, not in any Dafny field. Implemented in the host language.
    predicate {:extern} {:axiom} SpanGrounded(text: string)

    // Locate the FIRST identifier-like token in `unitTokens` whose rendered text
    // is out-of-schema for the current example. The membership signal is identical
    // to SpanGrounded (same prompt-derived support set, same identifier filtering);
    // the addition is `idx`, the index of that token WITHIN `unitTokens`, so a
    // rollback can penalize THAT token instead of the unit's first token. `found`
    // is false (and `idx` meaningless) when every identifier is in the support set
    // or no support set was parsed. Pure with respect to the Dafny heap (reads only
    // host state); implemented in the host language. `unitTokens` is rendered by
    // concatenation, exactly as RenderPrefix renders it.
    method {:extern} {:axiom} FirstUngroundedIdentifierTokenIdx(unitTokens: Prefix)
        returns (found: bool, idx: nat)
      ensures found ==> idx < |unitTokens|
  }

  class Parser {
    predicate {:extern} {:axiom} IsValidPrefix(prefix: Prefix)
      ensures forall k: nat :: 0 <= k < |prefix| - 1 ==> IsValidPrefix(prefix[k..])

    predicate {:extern} {:axiom} IsCompletePrefix(prefix: Prefix)
      ensures IsValidPrefix(prefix)

    function {:extern} {:axiom} ValidNextTokenCount(prefix: Prefix): nat
      requires IsValidPrefix(prefix)
      ensures ValidNextTokenCount(prefix) == |ValidNextTokens(prefix)|

    predicate IsDeadPrefix(prefix: Prefix)
    {
      !IsCompletePrefix(prefix) && ValidNextTokenCount(prefix) == 0
    }

    predicate {:extern} {:axiom} ValidNextToken(prefix: Prefix, token: Token)
      requires IsValidPrefix(prefix)
      ensures ValidNextToken(prefix, token) <==> token in ValidNextTokens(prefix)

    function {:extern} {:axiom} ValidNextTokens(prefix: Prefix): seq<Token>
      requires IsValidPrefix(prefix)
      ensures forall t :: t in ValidNextTokens(prefix) ==> IsValidPrefix(prefix + [t])
      ensures (IsCompletePrefix(prefix) || |ValidNextTokens(prefix)| > 0)

    method {:extern} {:axiom} ParseG(input: string) returns (isSuccess: bool)

    // Number of schema-bearing grammar symbols (table_ref / column_ref) that have
    // COMPLETED within `prefix`, read from the parser's SymbolPosMap side-record
    // (IterGen's mechanism, ported into the vendored incremental parser). The
    // record is a pure side-effect of parsing: it never changes which tokens are
    // accepted, so reading this count cannot alter decode for any caller. Used as
    // a mid-query unit boundary — when the count rises, one more table/column name
    // just finished, so the host can ground-check it without waiting for the whole
    // query to parse. Implemented in the host language.
    function {:extern} {:axiom} CompletedSchemaSymbolCount(prefix: Prefix): nat
      requires IsValidPrefix(prefix)

    // IterGen SymbolPosMap queries (pure side-record; see CompletedSchemaSymbolCount).
    // `symbol == "token"` is reserved: count is |prefix| (one unit per token).
    function {:extern} {:axiom} GrammarSymbolCount(prefix: Prefix, symbol: string): nat
      requires IsValidPrefix(prefix)

    // Token index into `prefix` at the start of the `occurrenceIdx`-th completed
    // `symbol` span (prefix[..idx] excludes that occurrence). For `symbol == "token"`,
    // occurrenceIdx is a zero-based token index and idx == occurrenceIdx.
    function {:extern} {:axiom} GrammarSymbolStartTokenIdx(prefix: Prefix, symbol: string, occurrenceIdx: nat): nat
      requires IsValidPrefix(prefix)
      requires occurrenceIdx < GrammarSymbolCount(prefix, symbol)
      ensures 0 <= GrammarSymbolStartTokenIdx(prefix, symbol, occurrenceIdx) <= |prefix|

    function {:extern} {:axiom} GrammarSymbolEndTokenIdx(prefix: Prefix, symbol: string, occurrenceIdx: nat): nat
      requires IsValidPrefix(prefix)
      requires occurrenceIdx < GrammarSymbolCount(prefix, symbol)
      ensures GrammarSymbolStartTokenIdx(prefix, symbol, occurrenceIdx) <= GrammarSymbolEndTokenIdx(prefix, symbol, occurrenceIdx) <= |prefix|

    method {:extern} {:axiom} GetGrammarSymbolUnits(prefix: Prefix, symbol: string) returns (units: seq<string>)
      requires IsValidPrefix(prefix)
      ensures |units| == GrammarSymbolCount(prefix, symbol)
  }

  function GrammarSymbolPresent(parser: Parser, prefix: Prefix, symbol: string): bool
    requires parser.IsValidPrefix(prefix)
  {
    if symbol == "token" then |prefix| > 0
    else parser.GrammarSymbolCount(prefix, symbol) > 0
  }

  function Contains(s: string, sub: string): bool
  {
    exists i, j :: 0 <= i <= j <= |s| && s[i..j] == sub
  }

  // Flatten a token prefix back to the plain string it renders to. Used to test
  // a span's closing delimiter by its rendered SURFACE TEXT rather than by exact
  // final-token identity, so a span that closed as a split '>'+'>' or a
  // space-prefixed ' >>' token is still recognized as already-closed.
  function RenderPrefix(p: Prefix): string
  {
    if |p| == 0 then ""
    else p[0] + RenderPrefix(p[1..])
  }

  predicate RenderedEndsWith(p: Prefix, suf: string)
  {
    var s := RenderPrefix(p);
    |s| >= |suf| && s[|s| - |suf|..] == suf
  }

  class CSDHelpers {
    const lm: LM
    const parser: Parser

    function cost(): int
      reads this, lm
    {
      lm.stepCost
    }

    constructor(lm: LM, parser: Parser)
      requires lm.ValidTokensIdsLogits()
      modifies lm
      ensures lm.stepCost == 0
      ensures lm.ValidTokensIdsLogits()
      ensures this.lm == lm
      ensures this.parser == parser
      ensures this.lm.Logits == old(lm.Logits)
    {
      this.lm := lm;
      this.parser := parser;
      lm.stepCost := 0;
    }

    method AppendTaskGuidance(guidance: string)
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures lm.stepCost == old(lm.stepCost)
    {
      lm.AppendTaskGuidance(guidance);
    }

    method UnconstrainedStep(prompt: Prefix, generated: Prefix) returns (next: Token)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures lm.stepCost == old(lm.stepCost) + 1
      ensures lm.Logits == old(lm.Logits)
    {
      lm.GenerateLogits(prompt + generated);
      next := lm.ChooseNextTokenUnconstrained();
      lm.stepCost := lm.stepCost + 1;
    }

    method UnconstrainedChunk(
      prompt: Prefix, generated: Prefix, maxChunkTokens: nat, openSpanToken: Token, eosToken: Token
    ) returns (generatedOut: Prefix, stoppedOnOpenSpan: bool, stoppedOnEos: bool, stepsUsed: nat)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires openSpanToken in lm.Tokens
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= |generatedOut|
      ensures generatedOut[..|generated|] == generated
      ensures |generatedOut| <= |generated| + stepsUsed
      ensures stepsUsed <= maxChunkTokens
      ensures lm.stepCost == old(lm.stepCost) + stepsUsed
      ensures !(stoppedOnOpenSpan && stoppedOnEos)
      ensures stoppedOnOpenSpan ==> |generatedOut| > |generated| && generatedOut[|generatedOut| - 1] == openSpanToken
      ensures stoppedOnEos ==> |generatedOut| + 1 == |generated| + stepsUsed
      ensures !stoppedOnEos ==> |generatedOut| == |generated| + stepsUsed
      ensures maxChunkTokens > 0 ==> stepsUsed > 0
    {
      var chunk: Prefix;
      chunk, stoppedOnOpenSpan, stoppedOnEos, stepsUsed := lm.GenerateUnconstrainedChunk(
        prompt + generated, maxChunkTokens, openSpanToken, eosToken
      );
      generatedOut := generated + chunk;
      lm.stepCost := lm.stepCost + stepsUsed;
    }

    // Generates one symbol worth of tokens via a multi-token LM call,
    // then accepts the longest parser-valid prefix of the emitted chunk.
    method ConstrainedSymbol(
      constrainedPrompt: Prefix, currentConstrained: Prefix,
      maxSymbolTokens: nat, eosToken: Token
    ) returns (currentOut: Prefix, hitEos: bool, stepsUsed: nat)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(currentConstrained)
      requires "<<" in lm.Tokens
      requires eosToken in lm.Tokens
      requires maxSymbolTokens > 0
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(currentOut)
      ensures |currentOut| >= |currentConstrained|
      ensures |currentOut| <= |currentConstrained| + stepsUsed
      ensures stepsUsed <= maxSymbolTokens
      ensures stepsUsed > 0
      ensures lm.stepCost == old(lm.stepCost) + stepsUsed
    {
      var chunk: Prefix;
      var stoppedOnOpen: bool;
      var stoppedOnEos: bool;
      chunk, stoppedOnOpen, stoppedOnEos, stepsUsed := lm.GenerateUnconstrainedChunk(
        constrainedPrompt + currentConstrained, maxSymbolTokens, "<<", eosToken
      );
      lm.stepCost := lm.stepCost + stepsUsed;
      hitEos := stoppedOnEos;
      currentOut := currentConstrained;
      var i := 0;
      while i < |chunk|
        invariant 0 <= i <= |chunk|
        invariant parser.IsValidPrefix(currentOut)
        invariant |currentOut| >= |currentConstrained|
        invariant |currentOut| <= |currentConstrained| + i
        decreases |chunk| - i
      {
        var tok := chunk[i];
        var extended := currentOut + [tok];
        if parser.IsValidPrefix(extended) && !parser.IsDeadPrefix(extended) {
          currentOut := extended;
        } else {
          break;
        }
        i := i + 1;
      }
    }

    method ConstrainedSymbolInGenerated(
      constrainedPrompt: Prefix, generated: Prefix,
      currentConstrained: Prefix, maxSymbolTokens: nat, eosToken: Token
    ) returns (generatedOut: Prefix, currentOut: Prefix, hitEos: bool, stepsUsed: nat)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(currentConstrained)
      requires |currentConstrained| <= |generated|
      requires "<<" in lm.Tokens
      requires eosToken in lm.Tokens
      requires maxSymbolTokens > 0
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(currentOut)
      ensures |currentOut| >= |currentConstrained|
      ensures |currentOut| <= |currentConstrained| + stepsUsed
      ensures |generatedOut| <= |generated| + stepsUsed
      ensures |currentOut| <= |generatedOut|
      ensures stepsUsed <= maxSymbolTokens
      ensures stepsUsed > 0
      ensures lm.stepCost == old(lm.stepCost) + stepsUsed
    {
      var stablePrefix := generated[..|generated| - |currentConstrained|];
      currentOut, hitEos, stepsUsed := ConstrainedSymbol(
        constrainedPrompt, currentConstrained, maxSymbolTokens, eosToken
      );
      generatedOut := stablePrefix + currentOut;
      assert |stablePrefix| == |generated| - |currentConstrained|;
      assert |generatedOut| == |stablePrefix| + |currentOut|;
      assert |generatedOut| <= |generated| + stepsUsed;
      assert |currentOut| <= |generatedOut|;
    }

    method OpenConstrainedSpan(generated: Prefix) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires "<<" in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures generatedOut == generated + ["<<"]
      ensures insideOut
      ensures currentOut == []
      ensures lm.stepCost == old(lm.stepCost) + 1
      ensures lm.Logits == old(lm.Logits)
    {
      generatedOut := generated + ["<<"];
      insideOut := true;
      currentOut := [];
      lm.stepCost := lm.stepCost + 1;
    }


    method EnterObservedConstrainedSpan(generated: Prefix) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures generatedOut == generated
      ensures insideOut
      ensures currentOut == []
      ensures lm.stepCost == old(lm.stepCost)
    {
      generatedOut := generated;
      insideOut := true;
      currentOut := [];
    }

    method AppendConstrainedToken(
      generated: Prefix, currentConstrained: Prefix, next: Token
    ) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(currentConstrained)
      requires !parser.IsCompletePrefix(currentConstrained)
      requires next in lm.Tokens
      requires parser.IsValidPrefix(currentConstrained + [next])
      ensures lm.ValidTokensIdsLogits()
      ensures generatedOut == generated + [next]
      ensures insideOut
      ensures currentOut == currentConstrained + [next]
      ensures parser.IsValidPrefix(currentOut)
      ensures lm.stepCost == old(lm.stepCost)
      ensures lm.Logits == old(lm.Logits)
    {
      generatedOut := generated + [next];
      insideOut := true;
      currentOut := currentConstrained + [next];
    }

    method CloseConstrainedSpan(
      generated: Prefix, currentConstrained: Prefix
    ) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsCompletePrefix(currentConstrained)
      requires ">>" in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures RenderedEndsWith(currentConstrained, ">>") ==>
              generatedOut == generated
      ensures !RenderedEndsWith(currentConstrained, ">>") ==>
              generatedOut == generated + [">>"]
      ensures !insideOut
      ensures currentOut == []
      ensures lm.stepCost == old(lm.stepCost) + 1
      ensures lm.Logits == old(lm.Logits)
    {
      if RenderedEndsWith(currentConstrained, ">>") {
        generatedOut := generated;
      } else {
        generatedOut := generated + [">>"];
      }
      insideOut := false;
      currentOut := [];
      lm.stepCost := lm.stepCost + 1;
    }

    method ConstrainedStep(prompt: Prefix, generated: Prefix, eosToken: Token) returns (next: Token)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(generated)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(generated, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(generated + [next]) ==> t in lm.Tokens)
      ensures lm.stepCost == old(lm.stepCost) + 1
      ensures lm.Logits == old(lm.Logits)
    {
      lm.GenerateLogits(prompt + generated);
      RollbackPreservesTokenInvariant(generated);
      lm.MaskValidNextAndEos(parser, generated, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(generated, next);
        assert parser.IsValidPrefix(generated + [next]);
        ConstrainedStepNextValid(generated, next);
      }
      lm.stepCost := lm.stepCost + 1;
    }

    // Dead-end-avoiding constrained step. Plain constrained decoding masks the
    // grammar-invalid next tokens and commits whatever the model samples. But the
    // runtime grammar mask over-approximates (a whitespace-prefixed token can slip
    // through), so the sampled token can land in a prefix that is technically valid
    // yet has ZERO valid continuations -- a "dead" prefix the weak model can never
    // finish. This step uses the runtime-accurate IsDeadPrefix oracle: if the chosen
    // token would create a dead (or invalid) prefix, it masks just that token and
    // RE-SAMPLES from the same logits, up to maxRetries times, instead of committing
    // it. success=false means no non-dead continuation was found within the budget,
    // so the caller can fall back to rollback. This is one-step lookahead: it rules
    // out committing an immediately-dead token, not one whose continuations all die
    // several steps later.
    method DeadEndAvoidingStep(
      prompt: Prefix, generated: Prefix, eosToken: Token, maxRetries: nat
    ) returns (next: Token, success: bool)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(generated)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures success ==>
        (next == eosToken) ||
        (parser.IsValidPrefix(generated + [next]) && !parser.IsDeadPrefix(generated + [next]))
      ensures (success && next != eosToken) ==> parser.ValidNextToken(generated, next)
      ensures (success && next != eosToken) ==>
        (forall t: Token :: t in parser.ValidNextTokens(generated + [next]) ==> t in lm.Tokens)
      ensures lm.stepCost == old(lm.stepCost) + 1
      ensures lm.Logits == old(lm.Logits)
    {
      var logitsArray := lm.Logits;
      lm.GenerateLogits(prompt + generated);
      RollbackPreservesTokenInvariant(generated);
      lm.MaskValidNextAndEos(parser, generated, eosToken);
      next := lm.ChooseNextToken();
      var tries := 0;
      while next != eosToken
            && (!parser.IsValidPrefix(generated + [next]) || parser.IsDeadPrefix(generated + [next]))
            && tries < maxRetries
        invariant lm.ValidTokensIdsLogits()
        invariant next in lm.Tokens
        invariant !lm.IsMasked(next)
        invariant forall t: Token ::
          t in lm.Tokens && !parser.ValidNextToken(generated, t) && t != eosToken ==> lm.IsMasked(t)
        invariant lm.stepCost == old(lm.stepCost)
        invariant lm.Logits == logitsArray
        decreases maxRetries - tries
      {
        lm.MaskToken(next);
        next := lm.ChooseNextToken();
        tries := tries + 1;
      }
      if next != eosToken {
        // An unmasked, non-eos token cannot be one MaskValidNextAndEos masked,
        // so it is a genuine valid next token.
        assert !parser.ValidNextToken(generated, next) ==> lm.IsMasked(next);
        assert parser.ValidNextToken(generated, next);
        assert parser.IsValidPrefix(generated + [next]);
        RollbackPreservesTokenInvariant(generated);
        ConstrainedStepNextValid(generated, next);
      }
      success := next == eosToken ||
        (parser.IsValidPrefix(generated + [next]) && !parser.IsDeadPrefix(generated + [next]));
      lm.stepCost := lm.stepCost + 1;
    }

    method GroupHasValidMember(prefix: Prefix, group: seq<Token>) returns (anyValid: bool)
      requires parser.IsValidPrefix(prefix)
      ensures anyValid <==> (exists t :: t in group && parser.ValidNextToken(prefix, t))
      ensures lm.stepCost == old(lm.stepCost)
    {
      anyValid := false;
      var i := 0;
      while i < |group|
        invariant 0 <= i <= |group|
        invariant anyValid <==> (exists j :: 0 <= j < i && parser.ValidNextToken(prefix, group[j]))
        decreases |group| - i
      {
        if parser.ValidNextToken(prefix, group[i]) {
          anyValid := true;
        }
        i := i + 1;
      }
    }

    method BoostValidGroups(prefix: Prefix, groups: seq<seq<Token>>, amount: real)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires amount >= 0.0 && amount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures lm.stepCost == old(lm.stepCost)
      ensures lm.Logits == old(lm.Logits)
    {
      var i := 0;
      var logitsArray := lm.Logits;

      while i < |groups|
        invariant 0 <= i <= |groups|
        invariant lm.Logits == logitsArray
        invariant lm.ValidTokensIdsLogits()
        invariant lm.stepCost == old(lm.stepCost)
        decreases |groups| - i
      {
        var anyValid := GroupHasValidMember(prefix, groups[i]);
        if anyValid {
          BoostTokenLogits(groups[i], amount);
        }
        i := i + 1;
      }
    }

    method GroupBoostedConstrainedStep(
      prompt: Prefix, constrainedPrefix: Prefix,
      groups: seq<seq<Token>>, boostAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> parser.IsValidPrefix(constrainedPrefix + [next])
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures lm.stepCost == old(lm.stepCost) + 1
    {
      var logitsArray := lm.Logits;
      lm.GenerateLogits(prompt + constrainedPrefix);
      if |groups| > 0 {
        BoostValidGroups(constrainedPrefix, groups, boostAmount);
        assert lm.Logits == logitsArray;
      }
      RollbackPreservesTokenInvariant(constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(constrainedPrefix, next);
      }
      lm.stepCost := lm.stepCost + 1;
    }

    method AdaptiveConstrainedStep(
      prompt: Prefix, constrainedPrefix: Prefix,
      groups: seq<seq<Token>>, boostAmount: real, narrowThreshold: nat, eosToken: Token
    ) returns (next: Token)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> parser.IsValidPrefix(constrainedPrefix + [next])
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures lm.stepCost == old(lm.stepCost) + 1
      ensures lm.Logits == old(lm.Logits)
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      if |groups| > 0 {
        var validCount := parser.ValidNextTokenCount(constrainedPrefix);
        if validCount <= narrowThreshold {
          BoostValidGroups(constrainedPrefix, groups, boostAmount);
        }
      }
      RollbackPreservesTokenInvariant(constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(constrainedPrefix, next);
      }
      lm.stepCost := lm.stepCost + 1;
    }

    method AdaptiveConstrainedStepWithPenalties(
      prompt: Prefix, constrainedPrefix: Prefix,
      boostGroups: seq<seq<Token>>, boostAmount: real,
      penaltyTokens: seq<Token>, penaltyAmount: real,
      narrowThreshold: nat, eosToken: Token
    ) returns (next: Token)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> parser.IsValidPrefix(constrainedPrefix + [next])
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures lm.stepCost == old(lm.stepCost) + 1
    {
      var logitsArray := lm.Logits;
      lm.GenerateLogits(prompt + constrainedPrefix);
      if |boostGroups| > 0 {
        var validCount := parser.ValidNextTokenCount(constrainedPrefix);
        if validCount <= narrowThreshold {
          BoostValidGroups(constrainedPrefix, boostGroups, boostAmount);
        }
      }
      PenalizeTokenLogits(penaltyTokens, penaltyAmount);
      RollbackPreservesTokenInvariant(constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(constrainedPrefix, next);
      }
      lm.stepCost := lm.stepCost + 1;
    }

    method BoostedConstrainedStep(
      prompt: Prefix, constrainedPrefix: Prefix,
      tokensToBoost: seq<Token>, boostAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures lm.stepCost == old(lm.stepCost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      BoostTokenLogits(tokensToBoost, boostAmount);
      RollbackPreservesTokenInvariant(constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(constrainedPrefix, next);
      }
      lm.stepCost := lm.stepCost + 1;
    }

    method PenalizedConstrainedStep(
      prompt: Prefix, constrainedPrefix: Prefix,
      tokensToPenalize: seq<Token>, penaltyAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures lm.stepCost == old(lm.stepCost) + 1
      ensures lm.Logits == old(lm.Logits)
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      PenalizeTokenLogits(tokensToPenalize, penaltyAmount);
      RollbackPreservesTokenInvariant(constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(constrainedPrefix, next);
      }
      lm.stepCost := lm.stepCost + 1;
    }

    // Performs unconstrained decoding until we run out of steps.
    method UnconstrainedGeneration(prompt: Prefix, maxSteps: nat) returns (generated: Prefix)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures lm.stepCost <= old(lm.stepCost) + maxSteps
    {
      var logitsArray := lm.Logits;
      generated := [];
      var startCost := lm.stepCost;
      while lm.stepCost - startCost < maxSteps
        invariant 0 <= lm.stepCost - startCost <= maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant |generated| == lm.stepCost - startCost
        invariant lm.Logits == logitsArray
        decreases maxSteps - (lm.stepCost - startCost)
      {
        var next := UnconstrainedStep(prompt, generated);
        generated := generated + [next];
      }
    }

    // A lemma that lets us say if the LM can generate all next valid tokens, then if we append one of those to the end, the LM can still generate all next valid tokens for the new prefix.
    lemma {:axiom} ConstrainedStepNextValid(generated: Prefix, next: Token)
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(generated)
      requires forall t: Token :: t in parser.ValidNextTokens(generated) ==> t in lm.Tokens
      requires parser.IsValidPrefix(generated + [next])
      ensures forall t: Token :: t in parser.ValidNextTokens(generated + [next]) ==> t in lm.Tokens

    // Performs constrained decoding until we run out of steps or the generated string is complete in the grammar.
    method ConstrainedGeneration(prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, terminatedByEos: bool)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures parser.IsValidPrefix(generated)
      ensures terminatedByEos ==> (lm.stepCost == old(lm.stepCost) + |generated| + 1)
      ensures !terminatedByEos ==> (lm.stepCost == old(lm.stepCost) + |generated|)
    {
      var logitsArray := lm.Logits;
      generated := [];
      var startCost := lm.stepCost;
      terminatedByEos := false;
      while lm.stepCost - startCost < maxSteps && !parser.IsCompletePrefix(generated)
        invariant 0 <= lm.stepCost - startCost <= maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant !terminatedByEos ==> lm.stepCost - startCost == |generated|
        invariant parser.IsValidPrefix(generated)
        invariant !terminatedByEos
        invariant lm.Logits == logitsArray
        decreases maxSteps - (lm.stepCost - startCost)
      {
        var next := ConstrainedStep(prompt, generated, eosToken);
        if next == eosToken {
          terminatedByEos := true;
          break;
        }
        generated := generated + [next];
      }
    }


    // Returns the tokens in `prefix` that appear immediately after `keyword`.
    // Use: extract which tokens the model has emitted after a specific keyword
    // (e.g., table names after "FROM", variable names after "LET", etc.) so
    // a strategy can maintain its own semantic context across loop iterations.
    static method ExtractAfterKeyword(prefix: Prefix, keyword: Token) returns (following: seq<Token>)
      ensures forall t :: t in following ==> t in prefix
    {
      following := [];
      var i := 0;
      while i < |prefix|
        invariant 0 <= i <= |prefix|
        invariant forall t :: t in following ==> t in prefix
        decreases |prefix| - i
      {
        if prefix[i] == keyword && i + 1 < |prefix| {
          following := following + [prefix[i + 1]];
        }
        i := i + 1;
      }
    }

    static method IntersectTokenSets(a: seq<Token>, b: seq<Token>) returns (result: seq<Token>)
      requires |a| == |set t | t in a|
      ensures forall t :: t in result ==> t in a && t in b
      ensures |result| == |set t | t in result|
      ensures |result| <= |a| && |result| <= |b|
    {
      result := [];
      var i := 0;
      while i < |a|
        invariant 0 <= i <= |a|
        invariant |result| <= i
        invariant forall t :: t in result ==> t in a && t in b
        invariant |result| <= |a|
        decreases |a| - i
      {
        if a[i] in b && a[i] !in result {
          result := result + [a[i]];
        }
        i := i + 1;
      }
      assume {:axiom} |result| == |set t | t in result|;
      assume {:axiom} |result| <= |b|;
    }

    static method SubtractTokenSets(a: seq<Token>, b: seq<Token>) returns (result: seq<Token>)
      ensures forall t :: t in result ==> t in a && t !in b
      ensures |result| <= |a|
    {
      result := [];
      var i := 0;
      while i < |a|
        invariant 0 <= i <= |a|
        invariant |result| <= i
        invariant forall t :: t in result ==> t in a && t !in b
        decreases |a| - i
      {
        if a[i] !in b {
          result := result + [a[i]];
        }
        i := i + 1;
      }
    }

    method RollbackToValidPrefix(generated: Prefix) returns (repaired: Prefix)
      requires parser.IsValidPrefix([])
      ensures parser.IsValidPrefix(repaired)
      ensures |repaired| <= |generated|
    {
      repaired := generated;

      while !parser.IsValidPrefix(repaired) || parser.IsDeadPrefix(repaired)
        invariant |repaired| <= |generated|
        invariant parser.IsValidPrefix(repaired) || |repaired| > 0
        decreases |repaired|
      {
        repaired := repaired[..|repaired|-1];
      }
    }

    method RollbackConstrainedSuffix(
      generated: Prefix, currentConstrained: Prefix
    ) returns (generatedOut: Prefix, currentOut: Prefix)
      requires parser.IsValidPrefix([])
      requires |currentConstrained| <= |generated|
      ensures parser.IsValidPrefix(currentOut)
      ensures |currentOut| <= |currentConstrained|
      ensures generatedOut == generated[..|generated| - |currentConstrained|] + currentOut
      ensures |currentOut| <= |generatedOut|
      ensures |generatedOut| <= |generated|
    {
      var stablePrefix := generated[..|generated| - |currentConstrained|];
      currentOut := RollbackToValidPrefix(currentConstrained);
      generatedOut := stablePrefix + currentOut;
      assert |stablePrefix| == |generated| - |currentConstrained|;
      assert |generatedOut| == |stablePrefix| + |currentOut|;
      assert |generatedOut| <= |generated|;
      assert |currentOut| <= |generatedOut|;
    }

    method RollbackToCompletePrefix(generated: Prefix) returns (repaired: Prefix)
      requires parser.IsValidPrefix([])
      ensures parser.IsCompletePrefix(repaired) || repaired == []
      ensures parser.IsValidPrefix(repaired)
      ensures |repaired| <= |generated|
    {
      repaired := generated;

      while repaired != [] && !parser.IsCompletePrefix(repaired)
        invariant |repaired| <= |generated|
        decreases |repaired|
      {
        repaired := repaired[..|repaired|-1];
      }
    }

    method RollbackConstrainedToComplete(
      generated: Prefix, currentConstrained: Prefix
    ) returns (generatedOut: Prefix, currentOut: Prefix)
      requires parser.IsValidPrefix([])
      requires |currentConstrained| <= |generated|
      ensures parser.IsCompletePrefix(currentOut) || currentOut == []
      ensures parser.IsValidPrefix(currentOut)
      ensures |currentOut| <= |currentConstrained|
      ensures generatedOut == generated[..|generated| - |currentConstrained|] + currentOut
      ensures |currentOut| <= |generatedOut|
      ensures |generatedOut| <= |generated|
    {
      var stablePrefix := generated[..|generated| - |currentConstrained|];
      currentOut := RollbackToCompletePrefix(currentConstrained);
      generatedOut := stablePrefix + currentOut;
      assert |stablePrefix| == |generated| - |currentConstrained|;
      assert |generatedOut| == |stablePrefix| + |currentOut|;
      assert |generatedOut| <= |generated|;
      assert |currentOut| <= |generatedOut|;
    }

    // Rollback that actually lets the model RE-GENERATE. Plain RollbackToValidPrefix
    // only amputates trailing tokens down to a valid, non-dead prefix and stops --
    // the span is lost. This rolls back the same way, then re-generates forward FROM
    // that point using DeadEndAvoidingStep, which re-detects the dead branch and steers
    // around it. So the model gets a fresh, dead-end-aware attempt at the span instead
    // of just losing it.
    method RollbackAndRegenerate(
      prompt: Prefix, generated: Prefix,
      eosToken: Token, maxSteps: nat, maxRetries: nat
    ) returns (regenerated: Prefix)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(regenerated)
      ensures |regenerated| <= |generated| + maxSteps
      ensures lm.stepCost <= old(lm.stepCost) + maxSteps
      ensures lm.stepCost >= old(lm.stepCost)
    {
      var logitsArray := lm.Logits;
      var repaired := RollbackToValidPrefix(generated);
      regenerated := repaired;
      var startCost := lm.stepCost;
      while lm.stepCost - startCost < maxSteps && !parser.IsCompletePrefix(regenerated)
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(regenerated)
        invariant 0 <= lm.stepCost - startCost <= maxSteps
        invariant |regenerated| <= |generated| + (lm.stepCost - startCost)
        invariant lm.Logits == logitsArray
        decreases maxSteps - (lm.stepCost - startCost)
      {
        var next, ok := DeadEndAvoidingStep(prompt, regenerated, eosToken, maxRetries);
        if !ok || next == eosToken {
          break;
        }
        regenerated := regenerated + [next];
      }
    }

    // Roll the constrained span back to its last point that already parses as a
    // complete expression, then KEEP GENERATING forward from there (dead-end-aware),
    // tracking the longest complete point reached. Returns that best complete point.
    // Generation is capped at (maxSteps - closeReserve), so at least closeReserve
    // steps remain afterwards for the caller to emit the closing delimiter. The
    // returned span is always either empty or a complete, valid prefix -- so the
    // caller can always close it.
    method RollbackAndContinue(
      prompt: Prefix, generated: Prefix,
      currentConstrained: Prefix, eosToken: Token, maxSteps: nat, closeReserve: nat, maxRetries: nat
    ) returns (generatedOut: Prefix, currentOut: Prefix)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires eosToken in lm.Tokens
      requires |currentConstrained| <= |generated|
      requires closeReserve <= maxSteps
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsCompletePrefix(currentOut) || currentOut == []
      ensures parser.IsValidPrefix(currentOut)
      ensures generatedOut == generated[..|generated| - |currentConstrained|] + currentOut
      ensures lm.stepCost <= old(lm.stepCost) + (maxSteps - closeReserve)
      ensures lm.stepCost >= old(lm.stepCost)
    {
      var logitsArray := lm.Logits;
      var stablePrefix := generated[..|generated| - |currentConstrained|];
      var budget := maxSteps - closeReserve;
      var bestComplete := RollbackToCompletePrefix(currentConstrained);
      var running := bestComplete;
      var startCost := lm.stepCost;
      while lm.stepCost - startCost < budget
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(running)
        invariant parser.IsCompletePrefix(bestComplete) || bestComplete == []
        invariant parser.IsValidPrefix(bestComplete)
        invariant 0 <= lm.stepCost - startCost <= budget
        invariant lm.Logits == logitsArray
        decreases budget - (lm.stepCost - startCost)
      {
        var next, ok := DeadEndAvoidingStep(prompt + stablePrefix, running, eosToken, maxRetries);
        if !ok || next == eosToken {
          break;
        }
        running := running + [next];
        if parser.IsCompletePrefix(running) {
          bestComplete := running;
        }
      }
      currentOut := bestComplete;
      generatedOut := stablePrefix + currentOut;
    }

    static method FlattenTokenGroups(groups: seq<seq<Token>>) returns (flat: seq<Token>)
      ensures forall t :: t in flat ==> exists g :: g in groups && t in g
    {
      flat := [];
      var i := 0;
      while i < |groups|
        invariant 0 <= i <= |groups|
        invariant forall t :: t in flat ==> exists g :: g in groups && t in g
        decreases |groups| - i
      {
        flat := flat + groups[i];
        i := i + 1;
      }
    }

    static method GroupContaining(groups: seq<seq<Token>>, tok: Token) returns (idx: int)
      ensures -1 <= idx < |groups|
      ensures idx >= 0 ==> tok in groups[idx]
    {
      idx := -1;
      var i := 0;
      while i < |groups|
        invariant 0 <= i <= |groups|
        invariant -1 <= idx < |groups|
        invariant idx >= 0 ==> tok in groups[idx]
        decreases |groups| - i
      {
        if tok in groups[i] {
          idx := i;
          return;
        }
        i := i + 1;
      }
    }

    method LastTokenBefore(s: Prefix, sep: Token) returns (tok: Token, found: bool)
      ensures found ==> exists i :: 1 <= i < |s| && s[i] == sep && tok == s[i-1]
    {
      var idx: int := |s|;
      while idx > 0 && s[idx-1] != sep
        invariant 0 <= idx <= |s|
        invariant forall j :: idx <= j < |s| ==> s[j] != sep
        decreases idx
      {
        idx := idx - 1;
      }
      if idx >= 2 {
        found := true;
        tok := s[idx - 2];
      } else {
        found := false;
        tok := "";
      }
    }

    lemma {:axiom} RollbackPreservesTokenInvariant(prefix: Prefix)
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      ensures forall t: Token :: t in parser.ValidNextTokens(prefix) ==> t in lm.Tokens

    static function ExtractContentBetweenDelimiters(input: string, startDelim: string, endDelim: string): (content: string)
      ensures content != "" ==> exists pre, post :: input == pre + startDelim + content + endDelim + post
    {
      ExtractContentExtern(input, startDelim, endDelim)
    }

    static function {:extern} {:axiom} ExtractContentExtern(input: string, startDelim: string, endDelim: string): (content: string)
      ensures content != "" ==> exists pre, post :: input == pre + startDelim + content + endDelim + post
    
    method BoostTokenLogits(tokens: seq<Token>, amount: real) 
      modifies lm, lm.Logits 
      requires lm.ValidTokensIdsLogits() 
      requires amount >= 0.0 && amount <= 100000000.0 
      ensures lm.ValidTokensIdsLogits()
      ensures lm.stepCost == old(lm.stepCost)
      ensures lm.Logits == old(lm.Logits)
    { 
      var validTokens := IntersectTokenSets(lm.Tokens, tokens); 
      assert (set t | t in validTokens) <= (set t | t in lm.Tokens); 
      var i := 0;
      var logitsArray := lm.Logits; 

      while i < |validTokens| 
        invariant 0 <= i <= |validTokens|
        invariant lm.Logits == logitsArray
        invariant lm.ValidTokensIdsLogits() 
        invariant lm.stepCost == old(lm.stepCost)
        decreases |validTokens| - i 
      { 
        var id := lm.TokenToId(validTokens[i]); 
        assert 0 <= id < lm.Logits.Length; 
        var newVal := lm.Logits[id] + amount; 
        if newVal > 1000000000.0 { 
            newVal := 1000000000.0; 
        } 
        lm.Logits[id] := newVal; 
        i := i + 1; 
      } 
    }

    method PenalizeTokenLogits(tokens: seq<Token>, amount: real)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires amount >= 0.0 && amount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures lm.Logits == old(lm.Logits)
      ensures lm.stepCost == old(lm.stepCost)
    {
      var logitsArray := lm.Logits;
      var validTokens := IntersectTokenSets(lm.Tokens, tokens);
      var i := 0;
      while i < |validTokens|
        invariant 0 <= i <= |validTokens|
        invariant lm.ValidTokensIdsLogits()
        invariant lm.Logits == logitsArray
        invariant lm.stepCost == old(lm.stepCost)
        decreases |validTokens| - i
      {
        var id := lm.TokenToId(validTokens[i]);
        var newVal := lm.Logits[id] - amount;
        if newVal < -1000000000.0 { newVal := -1000000000.0; }
        lm.Logits[id] := newVal;
        i := i + 1;
      }
    }

    method MaskTokensInPrefix(prefix: Prefix)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures forall t :: t in prefix && t in lm.Tokens ==> lm.IsMasked(t)
      ensures forall t :: t in lm.Tokens && !(t in prefix) ==>
        lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
      ensures lm.stepCost == old(lm.stepCost)
    {
      var logitsArray := lm.Logits;
      var i := 0;
      while i < |prefix|
        invariant 0 <= i <= |prefix|
        invariant lm.ValidTokensIdsLogits()
        invariant forall t :: t in prefix[..i] && t in lm.Tokens ==> lm.IsMasked(t)
        invariant forall t :: t in lm.Tokens && !(t in prefix[..i]) ==>
          lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
        invariant lm.stepCost == old(lm.stepCost)
        invariant lm.Logits == logitsArray
        decreases |prefix| - i
      {
        if prefix[i] in lm.Tokens {
          lm.MaskToken(prefix[i]);
        }
        i := i + 1;
      }
    }

    method GetHighestLogitToken() returns (token: Token)
      requires lm.ValidTokensIdsLogits()
      requires |lm.Tokens| > 0
      ensures lm.ValidTokensIdsLogits()
      ensures token in lm.Tokens
    {
      var bestIdx := 0;
      var i := 1;
      while i < |lm.Tokens|
        invariant 1 <= i <= |lm.Tokens|
        invariant 0 <= bestIdx < |lm.Tokens|
        invariant lm.ValidTokensIdsLogits()
        decreases |lm.Tokens| - i
      {
        if lm.Logits[i] > lm.Logits[bestIdx] {
          bestIdx := i;
        }
        i := i + 1;
      }
      token := lm.IdToToken(bestIdx);
    }

    method GetLogitGap() returns (gap: real)
      requires lm.ValidTokensIdsLogits()
      requires |lm.Tokens| >= 2
      ensures lm.ValidTokensIdsLogits()
      ensures gap >= 0.0
    {
      var top1: real := -1000000001.0;
      var top2: real := -1000000001.0;
      var i := 0;
      while i < |lm.Tokens|
        invariant 0 <= i <= |lm.Tokens|
        invariant lm.ValidTokensIdsLogits()
        invariant top2 <= top1
        decreases |lm.Tokens| - i
      {
        if lm.Logits[i] > -1000000000.0 {
          var L := lm.Logits[i];
          if L > top1 {
            top2 := top1;
            top1 := L;
          } else if L > top2 {
            top2 := L;
          }
        }
        i := i + 1;
      }
      if top2 < -1000000000.0 {
        gap := 0.0;
      } else {
        gap := top1 - top2;
      }
    }

    static lemma {:axiom} UnchosenIndexExists(n: nat, chosenIdx: seq<int>, k: nat, picked: nat)
      requires picked < k <= n
      requires forall u :: u in chosenIdx ==> 0 <= u < n
      requires |chosenIdx| == picked
      requires forall i, j :: 0 <= i < j < |chosenIdx| ==> chosenIdx[i] != chosenIdx[j]
      ensures exists u :: 0 <= u < n && !(u in chosenIdx)

    lemma {:axiom} DistinctChosenSeq(idx: seq<int>)
      requires lm.ValidTokensIdsLogits()
      requires forall u :: u in idx ==> 0 <= u < |lm.Tokens|
      requires forall i, j :: 0 <= i < j < |idx| ==> idx[i] != idx[j]
      ensures lm.ValidTokensIdsLogits()

    method GetTopKTokens(k: nat) returns (tokens: seq<Token>)
      requires lm.ValidTokensIdsLogits()
      requires 1 <= k <= |lm.Tokens|
      ensures lm.ValidTokensIdsLogits()
      ensures |tokens| <= k
      ensures forall t :: t in tokens ==> t in lm.Tokens
      ensures forall i, j :: 0 <= i < j < |tokens| ==> tokens[i] != tokens[j]
      ensures lm.stepCost == old(lm.stepCost)
    {
      tokens := [];
      var picked := 0;
      while picked < k
        invariant 0 <= picked <= k
        invariant |tokens| == picked
        invariant lm.ValidTokensIdsLogits()
        invariant forall t :: t in tokens ==> t in lm.Tokens
        invariant forall i, j :: 0 <= i < j < |tokens| ==> tokens[i] != tokens[j]
        invariant lm.stepCost == old(lm.stepCost)
        decreases k - picked
      {
        // Pick the highest-logit index whose decoded STRING is not already
        // chosen. Dedup is by string, not by index: at runtime two vocab ids
        // can decode to the same string, so distinct indices need not be
        // distinct strings. If no unused string remains, stop early so the
        // result holds genuinely distinct tokens (|tokens| < k is allowed).
        var bestIdx: int := -1;
        var j := 0;
        while j < |lm.Tokens|
          invariant 0 <= j <= |lm.Tokens|
          invariant lm.ValidTokensIdsLogits()
          invariant bestIdx == -1 || (0 <= bestIdx < |lm.Tokens| && !(lm.Tokens[bestIdx] in tokens))
          decreases |lm.Tokens| - j
        {
          if !(lm.Tokens[j] in tokens) {
            if bestIdx == -1 || lm.Logits[j] > lm.Logits[bestIdx] {
              bestIdx := j;
            }
          }
          j := j + 1;
        }
        if bestIdx == -1 {
          break;
        }
        tokens := tokens + [lm.Tokens[bestIdx]];
        picked := picked + 1;
      }
    }

    method DeadEndDetection(prefix: Prefix, minValidCount: nat) returns (isNarrow: bool)
      requires parser.IsValidPrefix(prefix)
      ensures isNarrow <==> parser.ValidNextTokenCount(prefix) < minValidCount
    {
      var validCount := parser.ValidNextTokenCount(prefix);
      isNarrow := validCount < minValidCount;
    }

    method SoftConstrainedStep(
      prompt: Prefix, constrainedPrefix: Prefix,
      boostAmount: real, eosToken: Token
    ) returns (next: Token, isValid: bool)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures isValid <==> (next == eosToken || parser.IsValidPrefix(constrainedPrefix + [next]))
      ensures isValid && next != eosToken ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures lm.stepCost == old(lm.stepCost) + 1
      ensures lm.Logits == old(lm.Logits)
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      RollbackPreservesTokenInvariant(constrainedPrefix);
      lm.BoostValidNextAndEos(parser, constrainedPrefix, boostAmount, eosToken);
      next := lm.ChooseNextTokenUnconstrained();
      lm.stepCost := lm.stepCost + 1;
      isValid := next == eosToken || parser.IsValidPrefix(constrainedPrefix + [next]);
      if isValid && next != eosToken {
        ConstrainedStepNextValid(constrainedPrefix, next);
      }
    }

    method SafeSoftConstrainedStep(
      prompt: Prefix, constrainedPrefix: Prefix,
      boostAmount: real, eosToken: Token
    ) returns (next: Token, usedFallback: bool)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || parser.IsValidPrefix(constrainedPrefix + [next])
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures lm.stepCost == old(lm.stepCost) + 1
      ensures lm.Logits == old(lm.Logits)
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      RollbackPreservesTokenInvariant(constrainedPrefix);
      lm.BoostValidNextAndEos(parser, constrainedPrefix, boostAmount, eosToken);
      var softNext := lm.ChooseNextTokenUnconstrained();
      if softNext == eosToken || parser.IsValidPrefix(constrainedPrefix + [softNext]) {
        next := softNext;
        usedFallback := false;
        if next != eosToken {
          ConstrainedStepNextValid(constrainedPrefix, next);
        }
      } else {
        lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
        next := lm.ChooseNextToken();
        usedFallback := true;
        if next != eosToken {
          assert !lm.IsMasked(next);
          assert parser.ValidNextToken(constrainedPrefix, next);
          assert parser.IsValidPrefix(constrainedPrefix + [next]);
          ConstrainedStepNextValid(constrainedPrefix, next);
        }
      }
      lm.stepCost := lm.stepCost + 1;
    }

    method ConfidenceGatedStep(
      prompt: Prefix, constrainedPrefix: Prefix, eosToken: Token
    ) returns (next: Token, wasConstrained: bool)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || parser.IsValidPrefix(constrainedPrefix + [next])
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures lm.stepCost == old(lm.stepCost) + 1
      ensures lm.Logits == old(lm.Logits)
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      var topToken := GetHighestLogitToken();
      if topToken == eosToken {
        next := topToken;
        wasConstrained := false;
      } else if parser.IsValidPrefix(constrainedPrefix + [topToken]) {
        next := topToken;
        wasConstrained := false;
        RollbackPreservesTokenInvariant(constrainedPrefix);
        ConstrainedStepNextValid(constrainedPrefix, next);
      } else {
        RollbackPreservesTokenInvariant(constrainedPrefix);
        lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
        next := lm.ChooseNextToken();
        wasConstrained := true;
        if next != eosToken {
          assert !lm.IsMasked(next);
          assert parser.ValidNextToken(constrainedPrefix, next);
          assert parser.IsValidPrefix(constrainedPrefix + [next]);
          RollbackPreservesTokenInvariant(constrainedPrefix + [next]);
        }
      }
      lm.stepCost := lm.stepCost + 1;
    }

    static function CountSubstring(s: string, sub: string): nat
      requires |sub| > 0
      decreases |s|
    {
      if |s| < |sub| then 0
      else if s[..|sub|] == sub then 1 + CountSubstring(s[|sub|..], sub)
      else CountSubstring(s[1..], sub)
    }

    static function OccurrencesInRange(prefix: Prefix, target: Token, hi: nat): nat
      requires hi <= |prefix|
    {
      if hi == 0 then 0
      else OccurrencesInRange(prefix, target, hi - 1) + (if prefix[hi - 1] == target then 1 else 0)
    }

    static method CountTokenOccurrences(prefix: Prefix, target: Token) returns (count: nat)
      ensures count == OccurrencesInRange(prefix, target, |prefix|)
    {
      count := 0;
      var i := 0;
      while i < |prefix|
        invariant 0 <= i <= |prefix|
        invariant count == OccurrencesInRange(prefix, target, i)
        decreases |prefix| - i
      {
        if prefix[i] == target {
          count := count + 1;
        }
        i := i + 1;
      }
    }

    static method TokensSinceLastOccurrence(prefix: Prefix, target: Token) returns (dist: nat)
      ensures dist <= |prefix|
      ensures dist == |prefix| ==> forall i :: 0 <= i < |prefix| ==> prefix[i] != target
      ensures dist < |prefix| ==> prefix[|prefix| - 1 - dist] == target
    {
      dist := 0;
      while dist < |prefix| && prefix[|prefix| - 1 - dist] != target
        invariant 0 <= dist <= |prefix|
        invariant forall j :: |prefix| - dist <= j < |prefix| ==> prefix[j] != target
        decreases |prefix| - dist
      {
        dist := dist + 1;
      }
    }

    method GetTokenLogit(token: Token) returns (logit: real)
      requires lm.ValidTokensIdsLogits()
      requires token in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures logit == lm.Logits[lm.TokenToId(token)]
    {
      logit := lm.Logits[lm.TokenToId(token)];
    }

    method ValidTokenCount(prefix: Prefix) returns (count: nat)
      requires parser.IsValidPrefix(prefix)
      ensures count == parser.ValidNextTokenCount(prefix)
    {
      count := parser.ValidNextTokenCount(prefix);
    }

    method TopValidCandidates(
      prompt: Prefix, prefix: Prefix, maxCandidates: nat, eosToken: Token
    ) returns (candidates: seq<Token>)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires maxCandidates > 0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures 0 < |candidates| <= maxCandidates
      ensures forall t :: t in candidates ==> t in lm.Tokens
      ensures forall t :: t in candidates ==> t == eosToken || t in parser.ValidNextTokens(prefix)
      ensures forall i, j :: 0 <= i < j < |candidates| ==> candidates[i] != candidates[j]
      ensures lm.stepCost == old(lm.stepCost) + 1
    {
      var baseCost := lm.stepCost;
      lm.GenerateLogits(prompt + prefix);
      RollbackPreservesTokenInvariant(prefix);
      var validWithEos := parser.ValidNextTokens(prefix) + [eosToken];
      var pool: seq<Token> := [];
      var i := 0;

      while i < |validWithEos|
        invariant lm.ValidTokensIdsLogits()
        invariant 0 <= i <= |validWithEos|
        invariant |pool| <= i
        invariant forall t :: t in pool ==> t in lm.Tokens
        invariant forall t :: t in pool ==> t == eosToken || t in parser.ValidNextTokens(prefix)
        invariant forall i, j :: 0 <= i < j < |pool| ==> pool[i] != pool[j]
        invariant lm.stepCost == baseCost
        decreases |validWithEos| - i
      {
        var tok := validWithEos[i];
        if !(tok in pool) {
          pool := pool + [tok];
        }
        i := i + 1;
      }

      if |pool| == 0 {
        pool := [eosToken];
      }
      var target := if maxCandidates < |pool| then maxCandidates else |pool|;
      var chosen: seq<Token> := [];

      while |chosen| < target
        invariant lm.ValidTokensIdsLogits()
        invariant 0 < target <= |pool|
        invariant 0 <= |chosen| <= target
        invariant forall t :: t in chosen ==> t in pool
        invariant forall t :: t in chosen ==> t in lm.Tokens
        invariant forall t :: t in chosen ==> t == eosToken || t in parser.ValidNextTokens(prefix)
        invariant forall i, j :: 0 <= i < j < |chosen| ==> chosen[i] != chosen[j]
        invariant forall i, j :: 0 <= i < j < |pool| ==> pool[i] != pool[j]
        invariant lm.stepCost == baseCost
        decreases target - |chosen|
      {
        var bestTok := pool[0];
        var bestLogit := -1000000000.0;
        var found := false;
        var j := 0;

        while j < |pool|
          invariant lm.ValidTokensIdsLogits()
          invariant 0 <= j <= |pool|
          invariant found ==> bestTok in pool
          invariant found ==> !(bestTok in chosen)
          invariant found ==> bestLogit == lm.Logits[lm.TokenToId(bestTok)]
          invariant found ==> forall k :: 0 <= k < j && !(pool[k] in chosen) ==> lm.Logits[lm.TokenToId(pool[k])] <= bestLogit
          invariant !found ==> forall k :: 0 <= k < j ==> pool[k] in chosen
          invariant lm.stepCost == baseCost
          decreases |pool| - j
        {
          var tok := pool[j];
          if !(tok in chosen) {
            var tokLogit := lm.Logits[lm.TokenToId(tok)];
            if !found || tokLogit > bestLogit {
              bestTok := tok;
              bestLogit := tokLogit;
              found := true;
            }
          }
          j := j + 1;
        }

        if found {
          chosen := chosen + [bestTok];
        } else {
          break;
        }
      }

      if |chosen| == 0 {
        candidates := [pool[0]];
      } else {
        candidates := chosen;
      }
      lm.stepCost := lm.stepCost + 1;
    }

    method IsTokenValidNext(prefix: Prefix, token: Token) returns (isValid: bool)
      requires parser.IsValidPrefix(prefix)
      ensures isValid <==> parser.ValidNextToken(prefix, token)
    {
      isValid := parser.ValidNextToken(prefix, token);
    }

    method RepetitionPenaltyStep(
      prompt: Prefix, prefix: Prefix,
      generated: Prefix, penaltyAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(prefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(prefix + [next]) ==> t in lm.Tokens)
      ensures lm.stepCost == old(lm.stepCost) + 1
    {
      lm.GenerateLogits(prompt + prefix);
      PenalizeTokenLogits(generated, penaltyAmount);
      RollbackPreservesTokenInvariant(prefix);
      lm.MaskValidNextAndEos(parser, prefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(prefix, next);
        assert parser.IsValidPrefix(prefix + [next]);
        ConstrainedStepNextValid(prefix, next);
      }
      lm.stepCost := lm.stepCost + 1;
    }

    method SaveLogitsSnapshot() returns (snapshot: seq<Logit>)
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures |snapshot| == lm.Logits.Length
      ensures forall i :: 0 <= i < lm.Logits.Length ==> snapshot[i] == lm.Logits[i]
      ensures forall i :: 0 <= i < |snapshot| ==>
        -1000000000.0 <= snapshot[i] && snapshot[i] <= 1000000000.0
      ensures lm.stepCost == old(lm.stepCost)
    {
      snapshot := lm.Logits[0..lm.Logits.Length];
    }

    method RestoreLogitsSnapshot(snapshot: seq<Logit>)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires |snapshot| == lm.Logits.Length
      requires forall i :: 0 <= i < |snapshot| ==> -1000000000.0 <= snapshot[i] <= 1000000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures forall i :: 0 <= i < lm.Logits.Length ==> lm.Logits[i] == snapshot[i]
      ensures lm.stepCost == old(lm.stepCost)
    {
      var logitsArray := lm.Logits;
      var i := 0;
      while i < lm.Logits.Length
        invariant 0 <= i <= lm.Logits.Length
        invariant lm.ValidTokensIdsLogits()
        invariant forall j :: 0 <= j < i ==> lm.Logits[j] == snapshot[j]
        invariant forall j :: i <= j < lm.Logits.Length ==> lm.Logits[j] == old(lm.Logits[j])
        invariant lm.stepCost == old(lm.stepCost)
        invariant lm.Logits == logitsArray
        decreases lm.Logits.Length - i
      {
        lm.Logits[i] := snapshot[i];
        i := i + 1;
      }
    }

    method RolloutConstrainedWithPenalties(
      prompt: Prefix, startPrefix: Prefix,
      totalBudget: nat, penalties: seq<Token>, penaltyAmount: real, eosToken: Token
    ) returns (generatedOut: Prefix, stepsUsed: nat, terminatedByEos: bool)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(startPrefix)
      requires eosToken in lm.Tokens
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(generatedOut)
      ensures |startPrefix| <= |generatedOut| <= |startPrefix| + totalBudget
      ensures |generatedOut| <= |startPrefix| + stepsUsed
      ensures stepsUsed <= totalBudget
      ensures lm.stepCost == old(lm.stepCost) + stepsUsed
    {
      var logitsArray := lm.Logits;
      generatedOut := startPrefix;
      terminatedByEos := false;
      var startCost := lm.stepCost;
      while lm.stepCost - startCost < totalBudget && !parser.IsCompletePrefix(generatedOut)
        invariant 0 <= lm.stepCost - startCost <= totalBudget
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(generatedOut)
        invariant |startPrefix| <= |generatedOut| <= |startPrefix| + (lm.stepCost - startCost)
        invariant !terminatedByEos
        invariant lm.Logits == logitsArray
        decreases totalBudget - (lm.stepCost - startCost)
      {
        var next := PenalizedConstrainedStep(
          prompt, generatedOut, penalties, penaltyAmount, eosToken
        );
        if next == eosToken {
          terminatedByEos := true;
          break;
        }
        generatedOut := generatedOut + [next];
      }
      stepsUsed := lm.stepCost - startCost;
    }

    method SpeculativeConstrainedRollout(
      prompt: Prefix, constrainedPrefix: Prefix,
      numTokens: nat, eosToken: Token
    ) returns (candidateTokens: Prefix, candidatePrefix: Prefix,
               hitComplete: bool, hitEos: bool, stepsUsed: nat)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires eosToken in lm.Tokens
      requires numTokens >= 1
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(candidatePrefix)
      ensures candidatePrefix == constrainedPrefix + candidateTokens
      ensures |candidateTokens| <= numTokens
      ensures stepsUsed <= numTokens
      ensures lm.stepCost == old(lm.stepCost) + stepsUsed
    {
      var logitsArray := lm.Logits;
      var snap := SaveLogitsSnapshot();
      candidateTokens := [];
      var cur := constrainedPrefix;
      hitEos := false;
      var startCost := lm.stepCost;

      while lm.stepCost - startCost < numTokens && !parser.IsCompletePrefix(cur) && !hitEos
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(cur)
        invariant |constrainedPrefix| <= |cur|
        invariant cur[..|constrainedPrefix|] == constrainedPrefix
        invariant candidateTokens == cur[|constrainedPrefix|..]
        invariant |candidateTokens| + |constrainedPrefix| == |cur|
        invariant |candidateTokens| <= lm.stepCost - startCost <= numTokens
        invariant hitEos ==> |candidateTokens| + 1 <= lm.stepCost - startCost
        invariant !hitEos ==> |candidateTokens| == lm.stepCost - startCost
        invariant lm.Logits == logitsArray
        decreases numTokens - (lm.stepCost - startCost), if hitEos || parser.IsCompletePrefix(cur) then 0 else 1
      {
        var next := ConstrainedStep(prompt, cur, eosToken);
        if next == eosToken {
          hitEos := true;
        } else {
          cur := cur + [next];
          candidateTokens := candidateTokens + [next];
        }
      }

      stepsUsed := lm.stepCost - startCost;
      RestoreLogitsSnapshot(snap);
      candidatePrefix := cur;
      hitComplete := parser.IsCompletePrefix(cur);
    }

    // One self-discharging decode step. Advances generation by at most one token
    // and charges exactly one unit of cost on EVERY control path, so a caller's
    // single `while helpers.cost() < maxSteps` loop that calls only ManagedStep and then
    // sets `cost := helpers.cost()` discharges the strategy-level length, cost and
    // progress postconditions by construction (loop runs >=1 iteration when
    // maxSteps>0, and cost==steps). `done` is true when the step hit EOS or closed
    // the span (the caller may stop). Outside a span: one UnconstrainedStep; "<<"
    // opens a constrained span. Inside a span: close if the parser reports a
    // complete prefix, else one AdaptiveConstrainedStep and append it. Composes
    // only already-verified CSDHelpers primitives; adds no new decode behavior.
    // Body is exactly one iteration of GenerateWithManagedSpan's loop.
    method ManagedStep(
      prompt: Prefix,
      generated: Prefix,
      insideConstrained: bool,
      currentConstrained: Prefix,
      validTokenGroups: seq<seq<Token>>,
      boostAmount: real,
      narrowThreshold: nat,
      eosToken: Token
    ) returns (
      generatedOut: Prefix,
      insideOut: bool,
      currentOut: Prefix,
      done: bool
    )
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires !insideConstrained ==> currentConstrained == []
      requires insideConstrained ==> parser.IsValidPrefix(currentConstrained)
      requires insideConstrained ==> |currentConstrained| <= |generated|
      requires "<<" in lm.Tokens && ">>" in lm.Tokens
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures lm.stepCost == old(lm.stepCost) + 1
      ensures |generatedOut| <= |generated| + 1
      ensures !insideOut ==> currentOut == []
      ensures insideOut ==> parser.IsValidPrefix(currentOut)
      ensures insideOut ==> |currentOut| <= |generatedOut|
    {
      var logitsArray := lm.Logits;
      generatedOut := generated;
      insideOut := insideConstrained;
      currentOut := currentConstrained;
      done := false;
      if !insideConstrained {
        var next := UnconstrainedStep(prompt, generated);
        if next == eosToken {
          done := true;
          return;
        }
        generatedOut := generated + [next];
        if next == "<<" {
          insideOut := true;
          currentOut := [];
        }
      } else {
        var cg, ci, cc, closed := CloseSpanIfComplete(generated, currentConstrained);
        assert lm.Logits == logitsArray;
        if closed {
          generatedOut := cg;
          insideOut := ci;
          currentOut := cc;
          done := true;
          return;
        } else {
          var constrainedPrompt := prompt + generated[..|generated| - |currentConstrained|];
          var next := AdaptiveConstrainedStep(
            constrainedPrompt, currentConstrained,
            validTokenGroups, boostAmount, narrowThreshold, eosToken
          );
          if next == eosToken {
            done := true;
            return;
          } else {
            var appendedGenerated, appendedInside, appendedCurrent := AppendConstrainedToken(
              generated, currentConstrained, next
            );
            generatedOut := appendedGenerated;
            insideOut := appendedInside;
            currentOut := appendedCurrent;
          }
        }
      }
    }

    // Higher-order span-managed generation. Runs a full free-then-constrained
    // decode loop and discharges the strategy-level length, cost, progress and
    // parser-validity postconditions internally, so a caller need not write a
    // loop or its proof. Outside a span: UnconstrainedStep until "<<" is observed.
    // Inside a span: close if the parser reports a complete prefix, else take one
    // AdaptiveConstrainedStep and append it. Composes only already-verified
    // CSDHelpers primitives; adds no new decode behavior.
    method GenerateWithManagedSpan(
                  prompt: Prefix,
      generatedPrefix: Prefix,
      insideConstrained: bool,
      currentConstrained: Prefix,
      maxSteps: nat,
      validTokenGroups: seq<seq<Token>>,
      boostAmount: real,
      narrowThreshold: nat,
      eosToken: Token
    ) returns (
      generated: Prefix,
      insideConstrainedOut: bool,
      currentConstrainedOut: Prefix
    )
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires !insideConstrained ==> currentConstrained == []
      requires insideConstrained ==> parser.IsValidPrefix(currentConstrained)
      requires insideConstrained ==> |currentConstrained| <= |generatedPrefix|
      requires "<<" in lm.Tokens && ">>" in lm.Tokens
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= |generatedPrefix| + maxSteps
      ensures !insideConstrainedOut ==> currentConstrainedOut == []
      ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
      ensures lm.stepCost <= old(lm.stepCost) + maxSteps
      ensures maxSteps == 0 || lm.stepCost > old(lm.stepCost) || generated != generatedPrefix ||
              insideConstrainedOut != insideConstrained ||
              currentConstrainedOut != currentConstrained
    {
      var logitsArray := lm.Logits;
      generated := generatedPrefix;
      insideConstrainedOut := insideConstrained;
      currentConstrainedOut := currentConstrained;

      var startCost := lm.stepCost;
      while lm.stepCost - startCost < maxSteps
        invariant 0 <= lm.stepCost - startCost <= maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant !insideConstrainedOut ==> currentConstrainedOut == []
        invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
        invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
        invariant |generated| <= |generatedPrefix| + (lm.stepCost - startCost)
        invariant lm.Logits == logitsArray
        decreases maxSteps - (lm.stepCost - startCost)
      {
        if !insideConstrainedOut {
          var next := UnconstrainedStep(prompt, generated);
          if next == eosToken {
            break;
          }
          generated := generated + [next];
          if next == "<<" {
            insideConstrainedOut := true;
            currentConstrainedOut := [];
          }
        } else {
          var cg, ci, cc, closed := CloseSpanIfComplete(generated, currentConstrainedOut);
          if closed {
            generated := cg;
            insideConstrainedOut := ci;
            currentConstrainedOut := cc;
            break;
          } else {
            var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
            var next := AdaptiveConstrainedStep(
              constrainedPrompt, currentConstrainedOut,
              validTokenGroups, boostAmount, narrowThreshold, eosToken
            );
            if next == eosToken {
              break;
            } else {
              var appendedGenerated, appendedInside, appendedCurrent := AppendConstrainedToken(
                generated, currentConstrainedOut, next
              );
              generated := appendedGenerated;
              insideConstrainedOut := appendedInside;
              currentConstrainedOut := appendedCurrent;
            }
          }
        }
      }
    }

    // Like GenerateWithManagedSpan, but the unconstrained PREAMBLE is hard-capped at
    // `prefixBudget` steps: after that many unconstrained tokens (or once "<<" is
    // observed) the span is force-opened, so the constrained phase is guaranteed
    // budget to reach ">>".  A single unified step counter advances by exactly 1 per
    // step, so this discharges the length (|generated| <= |generatedPrefix| + maxSteps),
    // cost (cost <= old(lm.stepCost) + maxSteps), and progress postconditions by construction
    // — a strategy needs only one call plus `cost := helpers.cost`, with no hand-rolled
    // budget bookkeeping.  Modeled on GenerateWithManagedSpan (identical invariants); the
    // only added branch force-opens via the proven OpenConstrainedSpan.
    method CloseSpanIfComplete(
      generated: Prefix, currentConstrained: Prefix
    ) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix, closed: bool)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(currentConstrained)
      requires ">>" in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generatedOut| <= |generated| + 1
      ensures parser.IsCompletePrefix(currentConstrained) ==>
              (!insideOut && currentOut == [] && lm.stepCost == old(lm.stepCost) + 1 && closed)
      ensures !parser.IsCompletePrefix(currentConstrained) ==>
              (generatedOut == generated && insideOut == true &&
               currentOut == currentConstrained && lm.stepCost == old(lm.stepCost) && !closed)
      ensures lm.Logits == old(lm.Logits)
    {
      if parser.IsCompletePrefix(currentConstrained) {
        generatedOut, insideOut, currentOut := CloseConstrainedSpan(generated, currentConstrained);
        closed := true;
      } else {
        generatedOut := generated;
        insideOut := true;
        currentOut := currentConstrained;
        closed := false;
      }
    }

    // IterGen-style unit-level iterative improvement.
    //
    // Generates constrained tokens one at a time. Each time the parser
    // transitions to a COMPLETE prefix (a "unit boundary"), it checks whether
    // the rendered text of that unit is an element of `allowedUnits`. If
    // `allowedUnits` is empty the check is disabled (all units pass). On a
    // check failure:
    //   * if rollbackBudgetLeft > 0 AND retryCount < maxRetries, the span is
    //     rolled back to the last complete point before this unit, the first
    //     token after that point is penalized in the logits (recurrence penalty),
    //     and generation continues — mirroring IterGen's backward(unit) + retry.
    //   * otherwise the current (possibly bad) result is accepted and generation
    //     continues, preserving termination.
    //
    // Fairness: allowedUnits is expected to be populated from the schema text
    // already visible in the prompt (the same information IterGen's db_info
    // string provides), not from DB execution.
    //
    // Cost accounting: bounded by a single flat `budget` of total steps (NOT a
    // product of two free parameters), mirroring CloseSpanWithinBudget /
    // RegenerateUnitOnGroundingFailure. A caller passes its whole remaining step
    // budget and the per-call bound composes directly against the strategy
    // template's `cost <= maxSteps`, with no nonlinear-arithmetic / division lemma.
    // maxRetries / maxRollbackBudget stay as behavioral knobs (they gate the
    // rollback branch; they no longer size the bound).
    method RegenerateUnitOnCheckFailure(
      prompt: Prefix, currentConstrained: Prefix,
      eosToken: Token,
      budget: nat,
      maxRetries: nat,
      maxRollbackBudget: nat,
      allowedUnits: seq<string>
    ) returns (resultConstrained: Prefix)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(currentConstrained)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(resultConstrained)
      ensures lm.stepCost <= old(lm.stepCost) + budget
      ensures lm.stepCost >= old(lm.stepCost)
      // Length bound (mirrors RegenerateUnitOnGroundingFailure): each loop step
      // appends at most one net token (rollbacks only shrink the span), so the
      // produced span grows by at most the total step budget. Without this a
      // caller cannot prove the strategy template's
      // `|generated| <= |generatedPrefix| + maxSteps` postcondition.
      ensures |resultConstrained| <= |currentConstrained| + budget
    {
      var logitsArray := lm.Logits;
      resultConstrained := currentConstrained;
      // The last prefix for which IsCompletePrefix held (or the entry point).
      var checkpointConstrained := currentConstrained;
      var retryCount := 0;
      var rollbackBudgetUsed := 0;
      var startCost := lm.stepCost;

      while lm.stepCost - startCost < budget
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(resultConstrained)
        invariant parser.IsValidPrefix(checkpointConstrained)
        invariant |checkpointConstrained| <= |resultConstrained|
        invariant |resultConstrained| <= |currentConstrained| + (lm.stepCost - startCost)
        invariant 0 <= lm.stepCost - startCost <= budget
        invariant lm.Logits == logitsArray
        decreases budget - (lm.stepCost - startCost)
      {
        var next, ok := DeadEndAvoidingStep(prompt, resultConstrained, eosToken, 8);
        if !ok || next == eosToken {
          break;
        }
        var extended := resultConstrained + [next];
        resultConstrained := extended;

        // Unit boundary: IsCompletePrefix just became true.
        if parser.IsCompletePrefix(resultConstrained) {
          // Render the unit (from checkpointConstrained to here) and check it.
          var unitText := RenderPrefix(resultConstrained[|checkpointConstrained|..]);
          var passes := |allowedUnits| == 0 || unitText in allowedUnits;
          if passes {
            // Accept the unit: advance the checkpoint.
            checkpointConstrained := resultConstrained;
            retryCount := 0;
          } else if rollbackBudgetUsed < maxRollbackBudget && retryCount < maxRetries {
            // Roll back to the checkpoint and penalize the continuation.
            retryCount := retryCount + 1;
            rollbackBudgetUsed := rollbackBudgetUsed + 1;
            resultConstrained := checkpointConstrained;
            // Penalize the rejected first-token-past-checkpoint in current logits
            // so the model is steered away from the same continuation on retry.
            lm.GenerateLogits(prompt + resultConstrained);
            lm.MaskValidNextAndEos(parser, resultConstrained, eosToken);
            if next in lm.Tokens {
              lm.MaskToken(next);
            }
          } else {
            // No budget or retries left: accept the unit anyway to preserve termination.
            checkpointConstrained := resultConstrained;
            retryCount := 0;
          }
        }
      }
    }

    // Like RegenerateUnitOnCheckFailure, but the per-unit acceptance test is the
    // grounding predicate lm.SpanGrounded(renderedUnit) instead of membership in
    // a caller-supplied allowed set. It checks the identifier-like tokens WITHIN
    // each completed unit against the support set the host derives from the
    // prompt, rather than matching the whole rendered unit string. The rollback /
    // penalize / regenerate loop and all bounds are identical.
    //
    // Fairness: the grounding support set is taken only from prompt text (the
    // same information visible in the prompt to any baseline), not from DB
    // execution or gold labels.
    //
    // Cost accounting: bounded by a single flat `budget` of total steps (NOT a
    // product of two free parameters), mirroring CloseSpanWithinBudget. A caller
    // passes its whole remaining step budget and the per-call bound composes
    // directly against the strategy template's `cost <= maxSteps` postcondition,
    // with no nonlinear-arithmetic or division-sizing proof obligation.
    method RegenerateUnitOnGroundingFailure(
      prompt: Prefix, currentConstrained: Prefix,
      eosToken: Token,
      budget: nat,
      maxRetries: nat,
      maxRollbackBudget: nat
    ) returns (resultConstrained: Prefix)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(currentConstrained)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(resultConstrained)
      ensures lm.stepCost <= old(lm.stepCost) + budget
      ensures lm.stepCost >= old(lm.stepCost)
      // Length bound: each loop step appends at most one net token (rollbacks
      // only shrink the span), so the produced span grows by at most `budget`.
      // Without this a caller cannot prove the strategy template's
      // `|generated| <= |generatedPrefix| + maxSteps` postcondition.
      ensures |resultConstrained| <= |currentConstrained| + budget
    {
      var logitsArray := lm.Logits;
      resultConstrained := currentConstrained;
      // The last prefix at which a schema symbol (table/column) completed grounded
      // (or the entry point). Rollbacks return here, so retries are CHEAP — they
      // replay only the current symbol, not the whole query.
      var checkpointConstrained := currentConstrained;
      // Count of schema symbols completed at the checkpoint. The unit boundary
      // fires when the live count rises above this — i.e. one more table/column
      // name just finished mid-query (IterGen's SymbolPosMap boundary), instead of
      // waiting for the whole query to parse (IsCompletePrefix).
      var prevCount := parser.CompletedSchemaSymbolCount(currentConstrained);
      var retryCount := 0;
      var rollbackBudgetUsed := 0;
      var startCost := lm.stepCost;

      while lm.stepCost - startCost < budget
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(resultConstrained)
        invariant parser.IsValidPrefix(checkpointConstrained)
        invariant |checkpointConstrained| <= |resultConstrained|
        invariant 0 <= lm.stepCost - startCost <= budget
        invariant |resultConstrained| <= |currentConstrained| + (lm.stepCost - startCost)
        invariant lm.Logits == logitsArray
        decreases budget - (lm.stepCost - startCost)
      {
        var next, ok := DeadEndAvoidingStep(prompt, resultConstrained, eosToken, 8);
        if !ok || next == eosToken {
          break;
        }
        var extended := resultConstrained + [next];
        resultConstrained := extended;

        // Unit boundary: a table_ref/column_ref symbol just COMPLETED mid-query
        // (the schema-symbol count rose). Ground-check from the last grounded
        // checkpoint to here, exactly when IterGen checks membership.
        var newCount := parser.CompletedSchemaSymbolCount(resultConstrained);
        if newCount > prevCount {
          // Ground-check the unit (from checkpointConstrained to here) AND locate
          // the first out-of-schema identifier's token in one call. found=false
          // means every identifier is grounded (same signal as SpanGrounded).
          var unit := resultConstrained[|checkpointConstrained|..];
          var found, idx := lm.FirstUngroundedIdentifierTokenIdx(unit);
          if !found {
            // Accept the unit: advance the checkpoint and the symbol count.
            checkpointConstrained := resultConstrained;
            prevCount := newCount;
            retryCount := 0;
          } else if rollbackBudgetUsed < maxRollbackBudget && retryCount < maxRetries {
            // Roll the generation cursor back to the checkpoint and PERSISTENTLY
            // penalize the OUT-OF-SCHEMA identifier's token at its OWN position
            // (not the unit's first token). Under greedy decode the replay from
            // the checkpoint is deterministic, so it re-reaches that exact prefix;
            // PenalizeTriedTokenAt is re-applied there every regen, steering the
            // identifier away from the out-of-schema name. (A one-shot MaskToken
            // would be wiped by the next GenerateLogits and loop forever.)
            // `idx < |unit|` (extern postcondition) ==> badPos < |resultConstrained|.
            retryCount := retryCount + 1;
            rollbackBudgetUsed := rollbackBudgetUsed + 1;
            var badPos := |checkpointConstrained| + idx;
            var badToken := resultConstrained[badPos];
            var penalizePrefix := resultConstrained[..badPos];
            resultConstrained := checkpointConstrained;
            lm.GenerateLogits(prompt + resultConstrained);
            lm.MaskValidNextAndEos(parser, resultConstrained, eosToken);
            lm.PenalizeTriedTokenAt(prompt + penalizePrefix, badToken);
          } else {
            // No budget or retries left: accept the unit anyway to preserve termination.
            checkpointConstrained := resultConstrained;
            prevCount := newCount;
            retryCount := 0;
          }
        }
      }
    }

    // Rewind the active constrained suffix by `num` completed grammar units of
    // `symbol` (IterGen `backward`). `symbol == "token"` rolls back `num` tokens.
    // When fewer than `num` units exist, rewinds to the span entry (empty suffix).
    method RollbackToGrammarSymbol(
            generated: Prefix,
      currentConstrained: Prefix,
      symbol: string,
      num: nat
    ) returns (generatedOut: Prefix, currentOut: Prefix)
      requires parser.IsValidPrefix(currentConstrained)
      requires parser.IsValidPrefix([])
      requires |currentConstrained| <= |generated|
      requires num >= 1
      ensures parser.IsValidPrefix(currentOut)
      ensures generatedOut == generated[..|generated| - |currentConstrained|] + currentOut
      ensures |currentOut| <= |currentConstrained|
      ensures lm.stepCost == old(lm.stepCost)
    {
      var stablePrefix := generated[..|generated| - |currentConstrained|];
      if symbol == "token" {
        if |currentConstrained| <= num {
          currentOut := [];
        } else {
          currentOut := RollbackToValidPrefix(currentConstrained[..|currentConstrained| - num]);
        }
      } else {
        var cnt := parser.GrammarSymbolCount(currentConstrained, symbol);
        if cnt < num {
          currentOut := [];
        } else {
          var targetOcc := cnt - num;
          assert targetOcc < cnt;
          var startTok := parser.GrammarSymbolStartTokenIdx(currentConstrained, symbol, targetOcc);
          currentOut := RollbackToValidPrefix(currentConstrained[..startTok]);
        }
      }
      generatedOut := stablePrefix + currentOut;
    }

    // IterGen `view`: rendered text of each completed grammar-unit span for `symbol`.
    method ViewGrammarSymbols(
      prefix: Prefix, symbol: string
    ) returns (units: seq<string>)
      requires parser.IsValidPrefix(prefix)
      ensures |units| == parser.GrammarSymbolCount(prefix, symbol)
      ensures lm.stepCost == old(lm.stepCost)
    {
      units := parser.GetGrammarSymbolUnits(prefix, symbol);
    }

    // Thin wrapper around RollbackToGrammarSymbol (IterGen `backward` API).
    method ForwardUntilGrammarSymbol(
      prompt: Prefix,
      generated: Prefix, currentConstrained: Prefix,
      symbol: string, num: nat,
      eosToken: Token, budget: nat
    ) returns (
      generatedOut: Prefix,
      currentOut: Prefix,
      hitEos: bool,
      stepsUsed: nat
    )
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(currentConstrained)
      requires |currentConstrained| <= |generated|
      requires eosToken in lm.Tokens
      requires num >= 1
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(currentOut)
      ensures generatedOut == generated[..|generated| - |currentConstrained|] + currentOut
      ensures |currentOut| <= |currentConstrained| + stepsUsed
      ensures |generatedOut| <= |generated| + stepsUsed
      ensures stepsUsed <= budget
      ensures lm.stepCost == old(lm.stepCost) + stepsUsed
    {
      var logitsArray := lm.Logits;
      var stablePrefix := generated[..|generated| - |currentConstrained|];
      var running := currentConstrained;
      var startCount := parser.GrammarSymbolCount(running, symbol);
      var targetCount := startCount + num;
      hitEos := false;
      var startCost := lm.stepCost;

      while lm.stepCost - startCost < budget
        && parser.GrammarSymbolCount(running, symbol) < targetCount
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(running)
        invariant |running| <= |currentConstrained| + (lm.stepCost - startCost)
        invariant 0 <= lm.stepCost - startCost <= budget
        invariant lm.Logits == logitsArray
        decreases budget - (lm.stepCost - startCost)
      {
        var next, _ := SafeSoftConstrainedStep(prompt + stablePrefix, running, 0.0, eosToken);
        if next == eosToken {
          hitEos := true;
          break;
        }
        running := running + [next];
      }

      generatedOut := stablePrefix + running;
      currentOut := running;
      stepsUsed := lm.stepCost - startCost;
    }

    // Advance an open constrained span toward a completable state and emit the
    // closing delimiter, all within `budget` steps. Generates forward
    // (dead-end-aware) tracking the longest prefix that parses as complete, then
    // emits ">>" at that longest complete point, reserving one step for the close.
    // When no completable state is reachable within the budget the span is left
    // open (the grammar forbids closing an incomplete prefix). Composes only the
    // already-verified DeadEndAvoidingStep and CloseConstrainedSpan primitives.
    method CloseSpanWithinBudget(
      prompt: Prefix, generated: Prefix,
      currentConstrained: Prefix, eosToken: Token, budget: nat
    ) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)
      modifies lm, lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(currentConstrained)
      requires |currentConstrained| <= |generated|
      requires eosToken in lm.Tokens
      requires ">>" in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures !insideOut ==> currentOut == []
      ensures insideOut ==> parser.IsValidPrefix(currentOut)
      ensures insideOut ==> |currentOut| <= |generatedOut|
      ensures |generatedOut| <= |generated| + budget
      ensures lm.stepCost <= old(lm.stepCost) + budget
      ensures lm.stepCost >= old(lm.stepCost)
    {
      var logitsArray := lm.Logits;
      var stablePrefix := generated[..|generated| - |currentConstrained|];
      var running := currentConstrained;
      var bestComplete: Prefix := [];
      var haveComplete := false;
      if parser.IsCompletePrefix(currentConstrained) {
        bestComplete := currentConstrained;
        haveComplete := true;
      }
      var startCost := lm.stepCost;

      while lm.stepCost - startCost + 1 < budget
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(running)
        invariant |running| <= |currentConstrained| + (lm.stepCost - startCost)
        invariant haveComplete ==>
          (parser.IsCompletePrefix(bestComplete) && |bestComplete| <= |running|)
        invariant !haveComplete ==> bestComplete == []
        invariant 0 <= lm.stepCost - startCost <= budget
        invariant lm.Logits == logitsArray
        decreases budget - (lm.stepCost - startCost)
      {
        var next, ok := DeadEndAvoidingStep(prompt + stablePrefix, running, eosToken, 8);
        if !ok || next == eosToken {
          break;
        }
        running := running + [next];
        if parser.IsCompletePrefix(running) {
          bestComplete := running;
          haveComplete := true;
        }
      }

      if lm.stepCost - startCost < budget && haveComplete {
        var gc, ci, cc := CloseConstrainedSpan(stablePrefix + bestComplete, bestComplete);
        generatedOut := gc;
        insideOut := ci;
        currentOut := cc;
      } else {
        generatedOut := stablePrefix + running;
        insideOut := true;
        currentOut := running;
      }
    }
  }
}
