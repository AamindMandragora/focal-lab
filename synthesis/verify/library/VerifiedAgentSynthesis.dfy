module VerifiedDecoderAgent {
  type Token = string
  type Prefix = seq<Token>
  type Id = nat
  type Logit = real

  class LM {
    // Library functions to be implemented in Python using TensorFlow.

    const Tokens: seq<Token>
    const Ids: seq<Id>
    var Logits: array<Logit>

    // This predicate ensures that there's a bijection from Ids to Tokens, that the set of Ids are just a subsequence of the natural numbers (like indices in an array), and that each token has a corresponding logit between some upper and lower bound (pre-softmax).
    predicate ValidTokensIdsLogits()
      reads this
      reads this.Logits
    {
      ((|Tokens| == |Ids|) && (|Ids| == Logits.Length) && (|Ids| > 0 && Ids[0] == 0)) &&
      (forall i :: 0 <= i < |Ids| ==> (i == Ids[i]) && (i in Ids)) && 
      (forall i, j :: 0 <= i < |Tokens| && 0 <= j < |Tokens| && i != j ==> Tokens[i] != Tokens[j]) &&
      (forall token: Token :: token in Tokens ==> (exists i :: 0 <= i < |Ids| && Tokens[i] == token)) &&
      (forall i :: (0 <= i < Logits.Length) ==> (-1000000000.0 <= Logits[i] && Logits[i] <= 1000000000.0))
    }

    // The constructor for this LM wrapper class will create lists of Tokens, Ids, and Logits according to the above standards.
    constructor {:extern} {:axiom} ()
      ensures ValidTokensIdsLogits()

    // Function for getting an id's corresponding token.
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

    // Function for getting a token's corresponding id.
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

    // Helper function for the above method, actual implementation will not be recursive.
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

    // Function for getting an id's corresponding logit. 
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

    // Function for getting a token's corresponding logit.
    function TokenToLogit(token: Token): (logit: Logit)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
    {
      IdToLogit(TokenToId(token))
    }

    // Function for getting the corresponding logits for a list of tokens.
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

    // Function for getting the corresponding logits for a list of ids.
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

    // Method that sets a token's logit to -1000000000.0, ensuring it is never chosen.
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

    // Method that masks a list of tokens, ensuring none of them are chosen.
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

    // Method that masks every token except for a list of tokens, ensuring only one of them is chosen.
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

    // Function that checks if a specific token is masked.
    predicate IsMasked(token: Token)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
    {
      Logits[TokenToId(token)] == -1000000000.0
    }

    // Function that checks if an unmasked token exists to choose from.
    predicate HasUnmaskedToken()
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()
    {
      exists t: Token :: t in Tokens && !IsMasked(t)
    }

    // Extern method that calculates the logits for next possible tokens given an input string.
    method {:extern} {:axiom} GenerateLogits(input: Prefix)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()

    // Extern method choosing the next token using the calculated logits.
    method {:extern} {:axiom} ChooseNextToken() returns (token: Token)
      requires ValidTokensIdsLogits()
      ensures token in Tokens
      ensures !IsMasked(token)
      ensures ValidTokensIdsLogits()

    // Extern method choosing the next token from the FULL vocabulary.
    method {:extern} {:axiom} ChooseNextTokenUnconstrained() returns (token: Token)
      ensures token in Tokens
      ensures ValidTokensIdsLogits()

    // Extern method generating a short unconstrained continuation in one runtime call.
    // The returned chunk excludes EOS, but includes openSpanToken when that token ended the chunk.
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

    // Bulk hard-mask using the parser's valid-next set plus EOS.
    // Runtime implementations may use DFA masks directly instead of materializing tokens.
    method {:extern} {:axiom} MaskValidNextAndEos(parser: Parser, prefix: Prefix, eosToken: Token)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires eosToken in Tokens
      ensures ValidTokensIdsLogits()
      ensures forall t :: t in Tokens && !parser.ValidNextToken(prefix, t) && t != eosToken ==> IsMasked(t)

    // Bulk soft-boost using the parser's valid-next set plus EOS.
    // Runtime implementations may apply the boost from a DFA mask directly.
    method {:extern} {:axiom} BoostValidNextAndEos(parser: Parser, prefix: Prefix, amount: real, eosToken: Token)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires eosToken in Tokens
      requires amount >= 0.0 && amount <= 100000000.0
      ensures ValidTokensIdsLogits()
  }

  class Parser {
    // Library functions to be implemented in Python using Lark.

    // Extern function checking if the given prefix is valid under the grammar.
    predicate {:extern} {:axiom} IsValidPrefix(prefix: Prefix)
      ensures forall k: nat :: 0 <= k < |prefix| - 1 ==> IsValidPrefix(prefix[k..])

    // Extern function checking if the given prefix is complete under the grammar.
    predicate {:extern} {:axiom} IsCompletePrefix(prefix: Prefix)
      ensures IsValidPrefix(prefix)

    // Extern function returning the exact number of valid continuations without
    // materializing the entire continuation set.
    function {:extern} {:axiom} ValidNextTokenCount(prefix: Prefix): nat
      requires IsValidPrefix(prefix)
      ensures ValidNextTokenCount(prefix) == |ValidNextTokens(prefix)|

    // Function checking if the prefix isn't complete and cannot be completed.
    predicate IsDeadPrefix(prefix: Prefix)
    {
      !IsCompletePrefix(prefix) && ValidNextTokenCount(prefix) == 0
    }

    // Function checking if the given token is a valid continuation of the prefix.
    predicate {:extern} {:axiom} ValidNextToken(prefix: Prefix, token: Token)
      requires IsValidPrefix(prefix)
      ensures ValidNextToken(prefix, token) <==> token in ValidNextTokens(prefix)

    // Extern function returning the set of next tokens valid under the grammar.
    function {:extern} {:axiom} ValidNextTokens(prefix: Prefix): seq<Token>
      requires IsValidPrefix(prefix)
      ensures forall t :: t in ValidNextTokens(prefix) ==> IsValidPrefix(prefix + [t])
      ensures (IsCompletePrefix(prefix) || |ValidNextTokens(prefix)| > 0)

    // Parses a raw string using the grammar.
    // isSuccess: true if the string is fully valid under the grammar, false otherwise.
    method {:extern} {:axiom} ParseG(input: string) returns (isSuccess: bool)
  }

  function Contains(s: string, sub: string): bool
  {
    exists i, j :: 0 <= i <= j <= |s| && s[i..j] == sub
  }

  class CSDHelpers {
    // Library functions that QWEN must directly use to synthesize the constrained decoding agent.

    var cost: int

    constructor()
      ensures cost == 0
    {
      cost := 0;
    }

    // Performs a single unconstrained decoding step and returns the next token.
    method UnconstrainedStep(lm: LM, prompt: Prefix, generated: Prefix) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + generated);
      next := lm.ChooseNextTokenUnconstrained();
      cost := cost + 1;
    }

    // Generates a short unconstrained continuation in one runtime call.
    // The emitted chunk excludes EOS; if stoppedOnOpenSpan is true, the returned
    // generatedOut already ends with openSpanToken.
    method UnconstrainedChunk(
      lm: LM, prompt: Prefix, generated: Prefix, maxChunkTokens: nat, openSpanToken: Token, eosToken: Token
    ) returns (generatedOut: Prefix, stoppedOnOpenSpan: bool, stoppedOnEos: bool, stepsUsed: nat)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires openSpanToken in lm.Tokens
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= |generatedOut|
      ensures generatedOut[..|generated|] == generated
      ensures |generatedOut| <= |generated| + stepsUsed
      ensures stepsUsed <= maxChunkTokens
      ensures cost == old(cost) + stepsUsed
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
      cost := cost + stepsUsed;
    }

    // Generates one "symbol" worth of tokens by issuing one multi-token LM call,
    // then accepting the longest valid-parser-prefix of that chunk.
    //
    // Unlike ConstrainedStep (one token, hard-masked), this lets the LM generate
    // naturally for up to maxSymbolTokens tokens in a single forward pass.
    // Only the longest valid prefix is kept, so multi-subword identifiers like
    // "highschooler" can be emitted as a natural unit and validated holistically.
    //
    // Cost: bumps helpers.cost by stepsUsed (tokens generated, which may exceed
    // the number of tokens accepted into currentOut).
    //
    // Use: replace ConstrainedStep when token-level hard-masking breaks the
    // model's natural multi-token identifiers or clause boundaries.
    method ConstrainedSymbol(
      lm: LM, parser: Parser, constrainedPrompt: Prefix, currentConstrained: Prefix,
      maxSymbolTokens: nat, eosToken: Token
    ) returns (currentOut: Prefix, hitEos: bool, stepsUsed: nat)
      modifies lm.Logits, this
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
      ensures cost == old(cost) + stepsUsed
    {
      var chunk: Prefix;
      var stoppedOnOpen: bool;
      var stoppedOnEos: bool;
      // Use "<<" as the open-span sentinel — it should not appear in valid SQL
      // or arithmetic content, so generation runs until EOS or budget exhausted.
      chunk, stoppedOnOpen, stoppedOnEos, stepsUsed := lm.GenerateUnconstrainedChunk(
        constrainedPrompt + currentConstrained, maxSymbolTokens, "<<", eosToken
      );
      cost := cost + stepsUsed;
      hitEos := stoppedOnEos;
      // Walk the chunk accepting tokens that extend a valid parser prefix.
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
        if parser.IsValidPrefix(extended) {
          currentOut := extended;
        } else {
          break;
        }
        i := i + 1;
      }
    }

    // ConstrainedSymbol wrapper that updates the full generated output.
    // This packages the stable-prefix slice used by many generated strategies
    // so callers only need the length bound, not suffix-equality invariants.
    method ConstrainedSymbolInGenerated(
      lm: LM, parser: Parser, constrainedPrompt: Prefix, generated: Prefix,
      currentConstrained: Prefix, maxSymbolTokens: nat, eosToken: Token
    ) returns (generatedOut: Prefix, currentOut: Prefix, hitEos: bool, stepsUsed: nat)
      modifies lm.Logits, this
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
      ensures cost == old(cost) + stepsUsed
    {
      var stablePrefix := generated[..|generated| - |currentConstrained|];
      currentOut, hitEos, stepsUsed := ConstrainedSymbol(
        lm, parser, constrainedPrompt, currentConstrained, maxSymbolTokens, eosToken
      );
      generatedOut := stablePrefix + currentOut;
      assert |stablePrefix| == |generated| - |currentConstrained|;
      assert |generatedOut| == |stablePrefix| + |currentOut|;
      assert |generatedOut| <= |generated| + stepsUsed;
      assert |currentOut| <= |generatedOut|;
    }

    method OpenConstrainedSpan(lm: LM, generated: Prefix) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)
      modifies this
      requires lm.ValidTokensIdsLogits()
      requires "<<" in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures generatedOut == generated + ["<<"]
      ensures insideOut
      ensures currentOut == []
      ensures cost == old(cost) + 1
    {
      generatedOut := generated + ["<<"];
      insideOut := true;
      currentOut := [];
      cost := cost + 1;
    }


    method EnterObservedConstrainedSpan(lm: LM, generated: Prefix) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)
      modifies this
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures generatedOut == generated
      ensures insideOut
      ensures currentOut == []
      ensures cost == old(cost)
    {
      generatedOut := generated;
      insideOut := true;
      currentOut := [];
    }

    method AppendConstrainedToken(
      lm: LM, parser: Parser, generated: Prefix, currentConstrained: Prefix, next: Token
    ) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)
      modifies this
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
      ensures cost == old(cost)
    {
      generatedOut := generated + [next];
      insideOut := true;
      currentOut := currentConstrained + [next];
    }

    method CloseConstrainedSpan(
      lm: LM, parser: Parser, generated: Prefix, currentConstrained: Prefix
    ) returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)
      modifies this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsCompletePrefix(currentConstrained)
      requires ">>" in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |currentConstrained| > 0 && currentConstrained[|currentConstrained|-1] == ">>" ==>
              generatedOut == generated
      ensures !(|currentConstrained| > 0 && currentConstrained[|currentConstrained|-1] == ">>") ==>
              generatedOut == generated + [">>"]
      ensures !insideOut
      ensures currentOut == []
      ensures cost == old(cost) + 1
    {
      if |currentConstrained| > 0 && currentConstrained[|currentConstrained|-1] == ">>" {
        generatedOut := generated;  // >> already appended by constrained step (csd_start grammar)
      } else {
        generatedOut := generated + [">>"];
      }
      insideOut := false;
      currentOut := [];
      cost := cost + 1;
    }

    // Performs a single constrained decoding step and returns the next token.
    // EOS token is always allowed as a valid choice, enabling early termination.
    method ConstrainedStep(lm: LM, parser: Parser, prompt: Prefix, generated: Prefix, eosToken: Token) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(generated)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(generated, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(generated + [next]) ==> t in lm.Tokens)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + generated);
      RollbackPreservesTokenInvariant(lm, parser, generated);
      lm.MaskValidNextAndEos(parser, generated, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(generated, next);
        assert parser.IsValidPrefix(generated + [next]);
        ConstrainedStepNextValid(lm, parser, generated, next);
      }
      cost := cost + 1;
    }

    // Returns true iff at least one token in group is parser-valid at prefix.
    // Cost: 0 (pure parser queries, no LM call).
    method GroupHasValidMember(parser: Parser, prefix: Prefix, group: seq<Token>) returns (anyValid: bool)
      requires parser.IsValidPrefix(prefix)
      ensures anyValid <==> (exists t :: t in group && parser.ValidNextToken(prefix, t))
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

    // Boosts any externally supplied token group that has a valid member at prefix.
    // Tokens outside lm.Tokens are ignored by SafeBoostTokenLogits.
    method BoostValidGroups(lm: LM, parser: Parser, prefix: Prefix, groups: seq<seq<Token>>, amount: real)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires amount >= 0.0 && amount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures cost == old(cost)
    {
      var i := 0;
      while i < |groups|
        invariant 0 <= i <= |groups|
        invariant lm.ValidTokensIdsLogits()
        invariant cost == old(cost)
        decreases |groups| - i
      {
        var anyValid := GroupHasValidMember(parser, prefix, groups[i]);
        if anyValid {
          SafeBoostTokenLogits(lm, groups[i], amount);
        }
        i := i + 1;
      }
    }

    // One constrained token step with soft boosts from caller-supplied token groups
    // before hard grammar masking.
    method GroupBoostedConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      groups: seq<seq<Token>>, boostAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> parser.IsValidPrefix(constrainedPrefix + [next])
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      if |groups| > 0 {
        BoostValidGroups(lm, parser, constrainedPrefix, groups, boostAmount);
      }
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      }
      cost := cost + 1;
    }

    // Group-boosted constrained step that only applies boosts when the grammar
    // is narrow enough that valid-token groups are likely to be informative.
    method AdaptiveConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      groups: seq<seq<Token>>, boostAmount: real, narrowThreshold: nat, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> parser.IsValidPrefix(constrainedPrefix + [next])
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      if |groups| > 0 {
        var validCount := parser.ValidNextTokenCount(constrainedPrefix);
        if validCount <= narrowThreshold {
          BoostValidGroups(lm, parser, constrainedPrefix, groups, boostAmount);
        }
      }
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      }
      cost := cost + 1;
    }

    // ConstrainedStep variant that applies logit penalties before grammar masking.
    // Penalizes specified tokens (e.g. [">>"] to discourage premature closing),
    // then hard-masks invalid tokens and selects the best grammar-valid token.
    // EOS token is always allowed as a valid choice, enabling early termination.
    // Has identical postconditions to ConstrainedStep — maintains the full loop invariant.
    method PenalizedConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      tokensToPenalize: seq<Token>, penaltyAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires forall t :: t in tokensToPenalize ==> t in lm.Tokens
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      PenalizeTokenLogits(lm, tokensToPenalize, penaltyAmount);
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      }
      cost := cost + 1;
    }

    // ConstrainedStep variant that boosts specified tokens before grammar masking.
    // Boost tokens (e.g. [">>"] to force closing after long expressions),
    // then hard-masks invalid tokens and selects the best grammar-valid token.
    // EOS token is always allowed as a valid choice, enabling early termination.
    // Has identical postconditions to ConstrainedStep — maintains the full loop invariant.
    method BoostedConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      tokensToBoost: seq<Token>, boostAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires forall t :: t in tokensToBoost ==> t in lm.Tokens
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      BoostTokenLogits(lm, tokensToBoost, boostAmount);
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      }
      cost := cost + 1;
    }

    // Safe public wrapper for synthesized strategies. Tokens outside the LM
    // vocabulary are ignored before boosting, so callers do not need to prove
    // token membership for literal or context-derived token lists.
    method SafeBoostedConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      tokensToBoost: seq<Token>, boostAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      SafeBoostTokenLogits(lm, tokensToBoost, boostAmount);
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      }
      cost := cost + 1;
    }

    // Safe public wrapper for synthesized strategies. Tokens outside the LM
    // vocabulary are ignored before penalizing, so callers do not need to prove
    // token membership for literal or context-derived token lists.
    method SafePenalizedConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      tokensToPenalize: seq<Token>, penaltyAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(constrainedPrefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      SafePenalizeTokenLogits(lm, tokensToPenalize, penaltyAmount);
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(constrainedPrefix, next);
        assert parser.IsValidPrefix(constrainedPrefix + [next]);
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      }
      cost := cost + 1;
    }

    // Performs unconstrained decoding until we run out of steps.
    method UnconstrainedGeneration(lm: LM, prompt: Prefix, maxSteps: nat) returns (generated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| == maxSteps
      ensures cost == old(cost) + |generated|
    {
      generated := [];
      var steps := 0;
      while steps < maxSteps
        invariant 0 <= steps <= maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant steps == |generated|
        invariant cost == old(cost) + steps
        decreases maxSteps - steps
      {
        var next := UnconstrainedStep(lm, prompt, generated);
        generated := generated + [next];
        steps := steps + 1;
      }
    }

    // A lemma that lets us say if the LM can generate all next valid tokens, then if we append one of those to the end, the LM can still generate all next valid tokens for the new prefix.
    static lemma {:axiom} ConstrainedStepNextValid(lm: LM, parser: Parser, generated: Prefix, next: Token)
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(generated)
      requires forall t: Token :: t in parser.ValidNextTokens(generated) ==> t in lm.Tokens
      requires parser.IsValidPrefix(generated + [next])
      ensures forall t: Token :: t in parser.ValidNextTokens(generated + [next]) ==> t in lm.Tokens

    // Performs constrained decoding until we run out of steps or the generated string is complete in the grammar.
    method ConstrainedGeneration(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, terminatedByEos: bool)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures parser.IsValidPrefix(generated)
      ensures terminatedByEos ==> (cost == old(cost) + |generated| + 1)
      ensures !terminatedByEos ==> (cost == old(cost) + |generated|)
    {
      generated := [];
      var steps := 0;
      terminatedByEos := false;
      while steps < maxSteps && !parser.IsCompletePrefix(generated)
        invariant 0 <= steps <= maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant steps == |generated|
        invariant parser.IsValidPrefix(generated)
        invariant cost == old(cost) + steps
        invariant !terminatedByEos
        decreases maxSteps - steps
      {
        var next := ConstrainedStep(lm, parser, prompt, generated, eosToken);
        if next == eosToken {
          steps := steps + 1;
          terminatedByEos := true;
          break;
        }
        generated := generated + [next];
        steps := steps + 1;
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

    // Returns tokens present in both a and b (set intersection over seqs).
    // Use: narrow a parser-valid candidate set to tokens also in a
    // strategy-maintained semantic context (e.g., in-scope table columns).
    static method IntersectTokenSets(a: seq<Token>, b: seq<Token>) returns (result: seq<Token>)
      ensures forall t :: t in result ==> t in a && t in b
      ensures |result| <= |a|
    {
      result := [];
      var i := 0;
      while i < |a|
        invariant 0 <= i <= |a|
        invariant |result| <= i
        invariant forall t :: t in result ==> t in a && t in b
        decreases |a| - i
      {
        if a[i] in b {
          result := result + [a[i]];
        }
        i := i + 1;
      }
    }

    // Returns tokens in a that are not in b (set difference over seqs).
    // Use: after boosting context-relevant tokens, penalize the rest.
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

    // Deletes invalid tokens from the end of the generated prefix until it becomes valid, then returns.
    static method RollbackToValidPrefix(parser: Parser, generated: Prefix) returns (repaired: Prefix)
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

    method RollbackConstrainedSpan(
      parser: Parser, stablePrefix: Prefix, generated: Prefix, currentConstrained: Prefix
    ) returns (generatedOut: Prefix, currentOut: Prefix)
      requires parser.IsValidPrefix([])
      requires generated == stablePrefix + currentConstrained
      ensures parser.IsValidPrefix(currentOut)
      ensures |currentOut| <= |currentConstrained|
      ensures generatedOut == stablePrefix + currentOut
    {
      currentOut := RollbackToValidPrefix(parser, currentConstrained);
      generatedOut := stablePrefix + currentOut;
    }

    // Roll back only the active constrained suffix, computing the stable prefix
    // internally from the maintained length invariant.
    method RollbackConstrainedSuffix(
      parser: Parser, generated: Prefix, currentConstrained: Prefix
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
      currentOut := RollbackToValidPrefix(parser, currentConstrained);
      generatedOut := stablePrefix + currentOut;
      assert |stablePrefix| == |generated| - |currentConstrained|;
      assert |generatedOut| == |stablePrefix| + |currentOut|;
      assert |generatedOut| <= |generated|;
      assert |currentOut| <= |generatedOut|;
    }

    // Flattens a sequence of token groups into a single token bag.
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

    // Returns the index of a group containing tok, or -1 if no group contains it.
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

    // Returns the token immediately preceding the last occurrence of sep in s.
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

    // Lemma: After rollback, valid next tokens are still in LM vocabulary
    static lemma {:axiom} RollbackPreservesTokenInvariant(lm: LM, parser: Parser, prefix: Prefix)
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      ensures forall t: Token :: t in parser.ValidNextTokens(prefix) ==> t in lm.Tokens

    // Helper: Convert Prefix (seq<Token>) to a single string
    static function PrefixToString(p: Prefix): string
    {
      if |p| == 0 then ""
      else p[0] + PrefixToString(p[1..])
    }

    // New library function for extracting content between delimiters
    // Defined as a function to allow reasoning in specifications
    static function ExtractContentBetweenDelimiters(input: string, startDelim: string, endDelim: string): (content: string)
      ensures content != "" ==> exists pre, post :: input == pre + startDelim + content + endDelim + post
    {
      ExtractContentExtern(input, startDelim, endDelim)
    }

    static function {:extern} {:axiom} ExtractContentExtern(input: string, startDelim: string, endDelim: string): (content: string)
      ensures content != "" ==> exists pre, post :: input == pre + startDelim + content + endDelim + post
    // =========================================================================
    // PER-TOKEN CSD BUILDING BLOCKS
    // These operate at the single-token level and can be composed into
    // novel constrained decoding strategies different from CRANE.
    // =========================================================================

    // Adds `amount` to logits of specified tokens. Clamped to 1000000000.0 upper bound.
    // Cost: 0 (no LM call, just logit array modification).
    // Use: boost grammar-valid tokens for soft constraining, or boost << to encourage expressions.
    method BoostTokenLogits(lm: LM, tokens: seq<Token>, amount: real)
      modifies lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires forall t :: t in tokens ==> t in lm.Tokens
      requires amount >= 0.0 && amount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures forall t :: t in lm.Tokens && !(t in tokens) ==>
        lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
    {
      var i := 0;
      while i < |tokens|
        invariant 0 <= i <= |tokens|
        invariant lm.ValidTokensIdsLogits()
        invariant forall t :: t in lm.Tokens && !(t in tokens[..i]) ==>
          lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
        decreases |tokens| - i
      {
        var id := lm.TokenToId(tokens[i]);
        var newVal := lm.Logits[id] + amount;
        if newVal > 1000000000.0 { newVal := 1000000000.0; }
        lm.Logits[id] := newVal;
        i := i + 1;
      }
    }

    // Safe public wrapper for synthesized strategies. Tokens outside the LM
    // vocabulary are ignored instead of becoming a proof obligation for callers.
    method SafeBoostTokenLogits(lm: LM, tokens: seq<Token>, amount: real)
      modifies lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires amount >= 0.0 && amount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
    {
      var validTokens := IntersectTokenSets(lm.Tokens, tokens);
      BoostTokenLogits(lm, validTokens, amount);
    }

    // Subtracts `amount` from logits of specified tokens. Clamped to -1000000000.0 lower bound.
    // Cost: 0 (no LM call).
    // Use: penalize >> to prevent premature expression closing, penalize << to force reasoning.
    method PenalizeTokenLogits(lm: LM, tokens: seq<Token>, amount: real)
      modifies lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires forall t :: t in tokens ==> t in lm.Tokens
      requires amount >= 0.0 && amount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures forall t :: t in lm.Tokens && !(t in tokens) ==>
        lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
    {
      var i := 0;
      while i < |tokens|
        invariant 0 <= i <= |tokens|
        invariant lm.ValidTokensIdsLogits()
        invariant forall t :: t in lm.Tokens && !(t in tokens[..i]) ==>
          lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
        decreases |tokens| - i
      {
        var id := lm.TokenToId(tokens[i]);
        var newVal := lm.Logits[id] - amount;
        if newVal < -1000000000.0 { newVal := -1000000000.0; }
        lm.Logits[id] := newVal;
        i := i + 1;
      }
    }

    // Safe public wrapper for synthesized strategies. Tokens outside the LM
    // vocabulary are ignored instead of becoming a proof obligation for callers.
    method SafePenalizeTokenLogits(lm: LM, tokens: seq<Token>, amount: real)
      modifies lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires amount >= 0.0 && amount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
    {
      var validTokens := IntersectTokenSets(lm.Tokens, tokens);
      PenalizeTokenLogits(lm, validTokens, amount);
    }

    // Returns the token with the highest logit from current logits (argmax).
    // Cost: 0 (no LM call, just reads the logit array).
    // Assumes GenerateLogits was already called by the caller.
    // Use: inspect model's preference before deciding whether to constrain.
    method GetHighestLogitToken(lm: LM) returns (token: Token)
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

    // Returns true if the number of valid grammar continuations is below threshold.
    // Cost: 0 (pure parser query, no LM call).
    // Use: detect when grammar is forcing output (model has no real choice), bail out.
    method DeadEndDetection(parser: Parser, prefix: Prefix, minValidCount: nat) returns (isNarrow: bool)
      requires parser.IsValidPrefix(prefix)
      ensures isNarrow <==> parser.ValidNextTokenCount(prefix) < minValidCount
    {
      var validCount := parser.ValidNextTokenCount(prefix);
      isNarrow := validCount < minValidCount;
    }

    // One token generation step with soft grammar constraint.
    // Instead of hard-masking invalid tokens to -inf, boosts valid tokens' logits.
    // EOS token is always boosted to ensure it can be selected for early termination.
    // The model can still pick invalid tokens (very unlikely with large boost).
    // Cost: 1 (one GenerateLogits + one ChooseNextToken).
    // Returns (token, isValid) so caller can decide what to do if invalid.
    method SoftConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      boostAmount: real, eosToken: Token
    ) returns (next: Token, isValid: bool)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures isValid <==> (next == eosToken || parser.IsValidPrefix(constrainedPrefix + [next]))
      ensures isValid && next != eosToken ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.BoostValidNextAndEos(parser, constrainedPrefix, boostAmount, eosToken);
      next := lm.ChooseNextTokenUnconstrained();
      cost := cost + 1;
      isValid := next == eosToken || parser.IsValidPrefix(constrainedPrefix + [next]);
      if isValid && next != eosToken {
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      }
    }

    // Soft grammar preference with a hard fallback. This gives the LM one
    // softly biased choice, but if that choice would not preserve parser
    // validity, it falls back to the same hard mask used by ConstrainedStep.
    method SafeSoftConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
      boostAmount: real, eosToken: Token
    ) returns (next: Token, usedFallback: bool)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires boostAmount >= 0.0 && boostAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || parser.IsValidPrefix(constrainedPrefix + [next])
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
      lm.BoostValidNextAndEos(parser, constrainedPrefix, boostAmount, eosToken);
      var softNext := lm.ChooseNextTokenUnconstrained();
      if softNext == eosToken || parser.IsValidPrefix(constrainedPrefix + [softNext]) {
        next := softNext;
        usedFallback := false;
        if next != eosToken {
          ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
        }
      } else {
        lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
        next := lm.ChooseNextToken();
        usedFallback := true;
        if next != eosToken {
          assert !lm.IsMasked(next);
          assert parser.ValidNextToken(constrainedPrefix, next);
          assert parser.IsValidPrefix(constrainedPrefix + [next]);
          ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
        }
      }
      cost := cost + 1;
    }

    // One token generation step with confidence-gated constraining.
    // Generates logits, checks if model's top choice is grammar-valid.
    // If valid: uses model's choice directly (preserves model quality).
    // If invalid: applies hard grammar constraint (like CRANE).
    // EOS token is always allowed as a valid choice, enabling early termination.
    // Cost: 1 (one GenerateLogits + one ChooseNextToken).
    method ConfidenceGatedStep(
      lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix, eosToken: Token
    ) returns (next: Token, wasConstrained: bool)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(constrainedPrefix)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || parser.IsValidPrefix(constrainedPrefix + [next])
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(constrainedPrefix + [next]) ==> t in lm.Tokens)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + constrainedPrefix);
      var topToken := GetHighestLogitToken(lm);
      if topToken == eosToken {
        next := topToken;
        wasConstrained := false;
      } else if parser.IsValidPrefix(constrainedPrefix + [topToken]) {
        next := topToken;
        wasConstrained := false;
        RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
        ConstrainedStepNextValid(lm, parser, constrainedPrefix, next);
      } else {
        RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix);
        lm.MaskValidNextAndEos(parser, constrainedPrefix, eosToken);
        next := lm.ChooseNextToken();
        wasConstrained := true;
        if next != eosToken {
          assert !lm.IsMasked(next);
          assert parser.ValidNextToken(constrainedPrefix, next);
          assert parser.IsValidPrefix(constrainedPrefix + [next]);
          RollbackPreservesTokenInvariant(lm, parser, constrainedPrefix + [next]);
        }
      }
      cost := cost + 1;
    }

    // Counts non-overlapping occurrences of `sub` in `s`.
    // Cost: 0 (pure function, no LM call).
    // Use: count >> occurrences to track expression count, enable adaptive behavior.
    static function CountSubstring(s: string, sub: string): nat
      requires |sub| > 0
      decreases |s|
    {
      if |s| < |sub| then 0
      else if s[..|sub|] == sub then 1 + CountSubstring(s[|sub|..], sub)
      else CountSubstring(s[1..], sub)
    }

    // Returns the current logit value for a specific token.
    // Cost: 0 (reads logit array, no forward pass).
    // Requires GenerateLogits was already called. Does not modify state.
    // Use: inspect confidence for one specific token (e.g. ">>" or "<<") before deciding to constrain.
    method GetTokenLogit(lm: LM, token: Token) returns (logit: real)
      requires lm.ValidTokensIdsLogits()
      requires token in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures logit == lm.Logits[lm.TokenToId(token)]
    {
      logit := lm.Logits[lm.TokenToId(token)];
    }

    // Multiplies every logit by scalar, clamped to [-1000000000.0, 1000000000.0].
    // Cost: 0 (no forward pass, just array modification).
    // scalar > 0: temperature scaling. scalar < 1 sharpens distribution, scalar > 1 flattens it.
    // Use: apply temperature before sampling without re-running the model.
    method ScaleAllLogits(lm: LM, scalar: real)
      modifies lm.Logits
      requires lm.ValidTokensIdsLogits()
      requires scalar > 0.0 && scalar <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures forall t :: t in lm.Tokens && !(t in lm.Tokens) ==>
        lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
    {
      var i := 0;
      while i < |lm.Tokens|
        invariant 0 <= i <= |lm.Tokens|
        invariant lm.ValidTokensIdsLogits()
        invariant forall t :: t in lm.Tokens && !(t in lm.Tokens[..i]) ==>
          lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
        decreases |lm.Tokens| - i
      {
        var id := lm.TokenToId(lm.Tokens[i]);
        var newVal := lm.Logits[id] * scalar;
        if newVal > 1000000000.0 { newVal := 1000000000.0; }
        if newVal < -1000000000.0 { newVal := -1000000000.0; }
        lm.Logits[id] := newVal;
        i := i + 1;
      }
    }

    // Returns the exact number of grammar-valid continuations from prefix.
    // Cost: 0 (pure parser query, no LM call).
    // More precise than DeadEndDetection — returns a count, not a boolean.
    // Use: implement continuous constraint policies (e.g. hard-constrain when count < 5, soft when count < 50).
    method ValidTokenCount(parser: Parser, prefix: Prefix) returns (count: nat)
      requires parser.IsValidPrefix(prefix)
      ensures count == parser.ValidNextTokenCount(prefix)
    {
      count := parser.ValidNextTokenCount(prefix);
    }

    // Returns up to maxCandidates highest-logit legal continuations from prefix.
    // Generates logits once, then repeatedly scans the valid set to collect distinct top-scoring tokens.
    // EOS token is included as an admissible candidate.
    // Cost: 1 (one GenerateLogits; selection only reads current logits).
    method TopValidCandidates(
      lm: LM, parser: Parser, prompt: Prefix, prefix: Prefix, maxCandidates: nat, eosToken: Token
    ) returns (candidates: seq<Token>)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires maxCandidates > 0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures 0 < |candidates| <= maxCandidates
      ensures forall t :: t in candidates ==> t in lm.Tokens
      ensures forall t :: t in candidates ==> t == eosToken || t in parser.ValidNextTokens(prefix)
      ensures forall i, j :: 0 <= i < j < |candidates| ==> candidates[i] != candidates[j]
      ensures cost == old(cost) + 1
    {
      var baseCost := cost;
      lm.GenerateLogits(prompt + prefix);
      RollbackPreservesTokenInvariant(lm, parser, prefix);
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
        invariant cost == baseCost
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
        invariant cost == baseCost
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
          invariant cost == baseCost
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
      cost := cost + 1;
    }

    // Returns whether a specific token is a valid grammar continuation of prefix.
    // Cost: 0 (pure parser query, no LM call).
    // Cheaper than ValidNextTokens when you only care about one token.
    // Use: check if ">>" or "<<" is currently valid before deciding whether to boost or penalize it.
    method IsTokenValidNext(parser: Parser, prefix: Prefix, token: Token) returns (isValid: bool)
      requires parser.IsValidPrefix(prefix)
      ensures isValid <==> parser.ValidNextToken(prefix, token)
    {
      isValid := parser.ValidNextToken(prefix, token);
    }

    // Constrained step with penalty on tokens already present in generated.
    // Penalizes repeated tokens before hard-masking, then samples grammar-valid token.
    // EOS token is always allowed as a valid choice, enabling early termination.
    // Cost: 1 (one GenerateLogits + one ChooseNextToken).
    // Grammar validity guaranteed (same postconditions as ConstrainedStep).
    // Use: prevent the model from regenerating the same subexpression (e.g. <<a*b>><<a*b>>...).
    method RepetitionPenaltyStep(
      lm: LM, parser: Parser, prompt: Prefix, prefix: Prefix,
      generated: Prefix, penaltyAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires forall t :: t in generated ==> t in lm.Tokens
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(prefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(prefix + [next]) ==> t in lm.Tokens)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + prefix);
      PenalizeTokenLogits(lm, generated, penaltyAmount);
      RollbackPreservesTokenInvariant(lm, parser, prefix);
      lm.MaskValidNextAndEos(parser, prefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(prefix, next);
        assert parser.IsValidPrefix(prefix + [next]);
        ConstrainedStepNextValid(lm, parser, prefix, next);
      }
      cost := cost + 1;
    }

    // Safe public wrapper for synthesized strategies. Repeated-token penalties
    // are filtered through the LM vocabulary, so callers do not need to prove
    // every visible token is in lm.Tokens.
    method SafeRepetitionPenaltyStep(
      lm: LM, parser: Parser, prompt: Prefix, prefix: Prefix,
      generated: Prefix, penaltyAmount: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(prefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(prefix + [next]) ==> t in lm.Tokens)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + prefix);
      SafePenalizeTokenLogits(lm, generated, penaltyAmount);
      RollbackPreservesTokenInvariant(lm, parser, prefix);
      lm.MaskValidNextAndEos(parser, prefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(prefix, next);
        assert parser.IsValidPrefix(prefix + [next]);
        ConstrainedStepNextValid(lm, parser, prefix, next);
      }
      cost := cost + 1;
    }

    // Constrained step with temperature scaling applied before hard-masking.
    // Divides all logits by temperature (i.e. scales by 1/temperature), then hard-masks invalid tokens.
    // EOS token is always allowed as a valid choice, enabling early termination.
    // Cost: 1 (one GenerateLogits + one ChooseNextToken).
    // Grammar validity guaranteed (same postconditions as ConstrainedStep).
    // temperature < 1: sharpens distribution (more deterministic).
    // temperature > 1: flattens distribution (more exploratory).
    // Use: tune sampling sharpness independently per phase without re-running the model.
    method TemperatureConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, prefix: Prefix, temperature: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires temperature >= 0.00000001 && temperature <= 100000000.0
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(prefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(prefix + [next]) ==> t in lm.Tokens)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + prefix);
      var scalar := 1.0 / temperature;
      if scalar > 100000000.0 { scalar := 100000000.0; }
      ScaleAllLogits(lm, scalar);
      RollbackPreservesTokenInvariant(lm, parser, prefix);
      lm.MaskValidNextAndEos(parser, prefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(prefix, next);
        assert parser.IsValidPrefix(prefix + [next]);
        ConstrainedStepNextValid(lm, parser, prefix, next);
      }
      cost := cost + 1;
    }

    // Safe public wrapper for synthesized strategies. Temperature is clamped to
    // the range accepted by ScaleAllLogits before hard parser masking.
    method SafeTemperatureConstrainedStep(
      lm: LM, parser: Parser, prompt: Prefix, prefix: Prefix, temperature: real, eosToken: Token
    ) returns (next: Token)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(prefix)
      requires eosToken in lm.Tokens
      ensures lm.ValidTokensIdsLogits()
      ensures next in lm.Tokens
      ensures (next == eosToken) || (parser.ValidNextToken(prefix, next))
      ensures (next != eosToken) ==> (forall t: Token :: t in parser.ValidNextTokens(prefix + [next]) ==> t in lm.Tokens)
      ensures cost == old(cost) + 1
    {
      lm.GenerateLogits(prompt + prefix);
      var safeTemperature := temperature;
      if safeTemperature < 0.00000001 {
        safeTemperature := 0.00000001;
      }
      if safeTemperature > 100000000.0 {
        safeTemperature := 100000000.0;
      }
      var scalar := 1.0 / safeTemperature;
      if scalar > 100000000.0 {
        scalar := 100000000.0;
      }
      if scalar <= 0.0 {
        scalar := 1.0;
      }
      ScaleAllLogits(lm, scalar);
      RollbackPreservesTokenInvariant(lm, parser, prefix);
      lm.MaskValidNextAndEos(parser, prefix, eosToken);
      next := lm.ChooseNextToken();
      if next != eosToken {
        assert !lm.IsMasked(next);
        assert parser.ValidNextToken(prefix, next);
        assert parser.IsValidPrefix(prefix + [next]);
        ConstrainedStepNextValid(lm, parser, prefix, next);
      }
      cost := cost + 1;
    }

    // Snapshot current LM logits so search can branch/retry without losing state.
    method SaveLogitsSnapshot(lm: LM) returns (snapshot: seq<Logit>)
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures |snapshot| == lm.Logits.Length
      ensures forall i :: 0 <= i < lm.Logits.Length ==> snapshot[i] == lm.Logits[i]
      ensures cost == old(cost)
    {
      snapshot := lm.Logits[0..lm.Logits.Length];
    }

    // Restore LM logits from a prior snapshot.
    method RestoreLogitsSnapshot(lm: LM, snapshot: seq<Logit>)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires |snapshot| == lm.Logits.Length
      ensures lm.ValidTokensIdsLogits()
      ensures forall i :: 0 <= i < lm.Logits.Length ==> lm.Logits[i] == snapshot[i]
      ensures cost == old(cost)
    {
      var i := 0;
      while i < lm.Logits.Length
        invariant 0 <= i <= lm.Logits.Length
        invariant lm.ValidTokensIdsLogits()
        invariant forall j :: 0 <= j < i ==> lm.Logits[j] == snapshot[j]
        decreases lm.Logits.Length - i
      {
        lm.Logits[i] := snapshot[i];
        i := i + 1;
      }
    }

    // Rollout helper for search-style strategies: constrained decoding from a
    // valid start prefix under a single local budget.
    method RolloutConstrainedWithPenalties(
      lm: LM, parser: Parser, prompt: Prefix, startPrefix: Prefix,
      totalBudget: nat, penalties: seq<Token>, penaltyAmount: real, eosToken: Token
    ) returns (generatedOut: Prefix, stepsUsed: nat, terminatedByEos: bool)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(startPrefix)
      requires eosToken in lm.Tokens
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(generatedOut)
      ensures |startPrefix| <= |generatedOut| <= |startPrefix| + totalBudget
      ensures stepsUsed <= totalBudget
      ensures cost == old(cost) + stepsUsed
    {
      generatedOut := startPrefix;
      stepsUsed := 0;
      terminatedByEos := false;
      while stepsUsed < totalBudget && !parser.IsCompletePrefix(generatedOut)
        invariant 0 <= stepsUsed <= totalBudget
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(generatedOut)
        invariant |startPrefix| <= |generatedOut| <= |startPrefix| + stepsUsed
        invariant !terminatedByEos
        invariant cost == old(cost) + stepsUsed
        decreases totalBudget - stepsUsed
      {
        var next := SafePenalizedConstrainedStep(
          lm, parser, prompt, generatedOut, penalties, penaltyAmount, eosToken
        );
        if next == eosToken {
          terminatedByEos := true;
          stepsUsed := stepsUsed + 1;
          break;
        }
        generatedOut := generatedOut + [next];
        stepsUsed := stepsUsed + 1;
      }
    }

    // CARS-style retry search:
    // - attempt constrained rollout
    // - if unfinished and retries remain, penalize the first newly emitted token
    //   on the next try
    // - operate under one global token budget so total cost stays bounded
    method CarsRetryConstrainedGeneration(
      lm: LM, parser: Parser, prompt: Prefix, startPrefix: Prefix,
      totalBudget: nat, maxRetries: nat, penaltyAmount: real, eosToken: Token
    ) returns (best: Prefix, retriesUsed: nat, terminatedByEos: bool)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(startPrefix)
      requires eosToken in lm.Tokens
      requires penaltyAmount >= 0.0 && penaltyAmount <= 100000000.0
      ensures lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(best)
      ensures |startPrefix| <= |best| <= |startPrefix| + totalBudget
      ensures retriesUsed <= maxRetries
      ensures cost <= old(cost) + totalBudget
    {
      best := startPrefix;
      retriesUsed := 0;
      terminatedByEos := false;
      var penalties: seq<Token> := [];
      var budgetUsed: nat := 0;
      var done := false;

      while !done && budgetUsed < totalBudget
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(best)
        invariant |startPrefix| <= |best| <= |startPrefix| + budgetUsed
        invariant retriesUsed <= maxRetries
        invariant budgetUsed <= totalBudget
        invariant cost == old(cost) + budgetUsed
        decreases totalBudget - budgetUsed, (if done then 0 else 1)
      {
        var remaining := totalBudget - budgetUsed;
        var trial: Prefix;
        var used: nat;
        var hitEos: bool;
        trial, used, hitEos := RolloutConstrainedWithPenalties(
          lm, parser, prompt, startPrefix, remaining, penalties, penaltyAmount, eosToken
        );
        best := trial;
        budgetUsed := budgetUsed + used;
        terminatedByEos := hitEos;

        if hitEos || parser.IsCompletePrefix(trial) {
          done := true;
        } else if retriesUsed < maxRetries && |trial| > |startPrefix| {
          penalties := penalties + [trial[|startPrefix|]];
          retriesUsed := retriesUsed + 1;
        } else {
          done := true;
        }
      }
    }

    // Strategy 7: CRANE-style generation (Reasoning-Math-Reasoning).
    // Starts unconstrained. When "<<" is seen, switches to constrained.
    // When ">>" is seen, switches back to unconstrained.
    method CraneGeneration(
      lm: LM,
      parser: Parser,
      prompt: Prefix,
      maxSteps: nat,
      minReasoningSteps: nat,
      eosToken: Token
    ) returns (generated: Prefix)
      modifies lm.Logits, this
      requires lm.ValidTokensIdsLogits()
      requires "<<" in lm.Tokens && ">>" in lm.Tokens
      requires eosToken in lm.Tokens
      requires parser.IsValidPrefix([])
      ensures lm.ValidTokensIdsLogits()
      ensures |generated| <= maxSteps
      ensures cost <= old(cost) + maxSteps
    {
      generated := [];
      var steps := 0;
      var insideConstrained := false;
      var currentConstrained: Prefix := [];

      while steps < maxSteps
        invariant 0 <= steps <= maxSteps
        invariant steps == |generated|
        invariant |currentConstrained| <= |generated|
        invariant lm.ValidTokensIdsLogits()
        invariant !insideConstrained ==> currentConstrained == []
        invariant insideConstrained ==> parser.IsValidPrefix(currentConstrained)
        invariant cost == old(cost) + steps
        decreases maxSteps - steps, (if insideConstrained then 1 else 0)
      {
        if !insideConstrained {
          var next := UnconstrainedStep(lm, prompt, generated);
          if next == eosToken {
            break;
          }
          generated := generated + [next];
          steps := steps + 1;
          if Contains(next, "<<") {
            insideConstrained := true;
            currentConstrained := [];
          }
        } else {
          if parser.IsCompletePrefix(currentConstrained) {
            insideConstrained := false;
            currentConstrained := [];
          } else {
            var constrainedPrompt := prompt + generated[..|generated| - |currentConstrained|];
            var next, wasConstrained := ConfidenceGatedStep(
              lm, parser, constrainedPrompt, currentConstrained, eosToken
            );
            if next == eosToken {
              break;
            }
            generated := generated + [next];
            steps := steps + 1;

            if Contains(next, ">>") {
              insideConstrained := false;
              currentConstrained := [];
            } else {
              currentConstrained := currentConstrained + [next];
            }
          }
        }
      }
    }
  }
}
