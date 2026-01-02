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

    constructor {:axiom} ()
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