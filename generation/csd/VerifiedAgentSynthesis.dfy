module VerifiedDecoderAgent {
  type Token = string
  type Prefix = seq<Token>
  type Id = nat
  type Logit = real

  const LeftDelimiter: Token := "<<"
  const RightDelimiter: Token := ">>"
  const SpacedLeftDelimiter: Token := " <<"
  const SpacedRightDelimiter: Token := " >>"

  predicate Contains(s: string, sub: string)
  {
    exists i, j :: 0 <= i <= j <= |s| && s[i..j] == sub
  }

  predicate PrefixContains(p: Prefix, t: Token)
  {
    exists i :: 0 <= i < |p| && p[i] == t
  }

  predicate DelimitedAnswerValidForParser(parser: Parser, prefix: Prefix)
  {
    PrefixContains(prefix, LeftDelimiter) &&
    PrefixContains(prefix, RightDelimiter)
  }

  class LM {
    const Tokens: seq<Token>
    const Ids: seq<Id>
    var Logits: array<Logit>

    constructor {:extern} {:axiom} ()
      ensures ValidTokensIdsLogits()

    predicate ValidTokensIdsLogits()
      reads this
      reads this.Logits
    {
      ((|Tokens| == |Ids|) && (|Ids| == Logits.Length) && (|Ids| > 0 && Ids[0] == 0)) &&
      (forall i :: 0 <= i < |Ids| ==> (i == Ids[i]) && (i in Ids)) &&
      (forall i, j :: 0 <= i < |Tokens| && 0 <= j < |Tokens| && i != j ==> Tokens[i] != Tokens[j]) &&
      (forall token: Token :: token in Tokens ==> (exists i :: 0 <= i < |Ids| && Tokens[i] == token)) &&
      (forall i :: 0 <= i < Logits.Length ==> Logits[i] <= 1000000000.0 && Logits[i] >= -1000000000.0)
    }

    lemma {:axiom} ValidTokensIdsLogitsAlways()
      ensures ValidTokensIdsLogits()

    function IdToToken(id: Id): (token: Token)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires id in Ids
      ensures token in Tokens
      ensures Tokens[id] == token
      ensures id == TokenToId(token)
      ensures ValidTokensIdsLogits()
    {
      this.Tokens[id]
    }

    function TokenToId(token: Token): (id: Id)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures id in Ids
      ensures Tokens[id] == token
      ensures TokenToId(Tokens[id]) == id
      ensures ValidTokensIdsLogits()
    {
      this.TokenToIdRecursive(token, 0)
    }

    function TokenToIdRecursive(token: Token, offset: nat): (id: Id)
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
      if this.Tokens[offset] == token then offset
      else this.TokenToIdRecursive(token, offset + 1)
    }

    function IdToLogit(id: Id): (logit: Logit)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires id in Ids
      ensures logit in Logits[0..Logits.Length]
      ensures ValidTokensIdsLogits()
    {
      this.Logits[id]
    }

    function TokenToLogit(token: Token): (logit: Logit)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
    {
      this.IdToLogit(this.TokenToId(token))
    }

    function TokensToLogits(tokens: seq<Token>): (logits: seq<Logit>)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires |tokens| > 0
      requires forall token: Token :: token in tokens ==> token in Tokens
      ensures ValidTokensIdsLogits()
    {
      if |tokens| == 1 then [this.TokenToLogit(tokens[0])]
      else [this.TokenToLogit(tokens[0])] + this.TokensToLogits(tokens[1..])
    }

    function IdsToLogits(ids: seq<Id>): (logits: seq<Logit>)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires |ids| > 0
      requires forall id: Id :: id in ids ==> id in Ids
      ensures ValidTokensIdsLogits()
    {
      if |ids| == 1 then [this.IdToLogit(ids[0])]
      else [this.IdToLogit(ids[0])] + this.IdsToLogits(ids[1..])
    }

    method MaskToken(token: Token)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
      ensures IsMasked(token)
      ensures forall t: Token :: t in Tokens && t != token ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
    {
      var id := this.TokenToId(token);
      this.Logits[id] := -1000000000.0;
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
      {
        this.MaskToken(tokens[i]);
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
      var toMask: Prefix := [];
      var N := |this.Tokens|;
      var i := 0;
      while i < N
        invariant 0 <= i <= N
        invariant ValidTokensIdsLogits()
        invariant forall j :: 0 <= j < i && !(Tokens[j] in tokens) ==> Tokens[j] in toMask
        invariant forall j :: 0 <= j < i && Tokens[j] in tokens ==> !(Tokens[j] in toMask)
        invariant forall t: Token :: t in toMask ==> t !in tokens && t in Tokens
      {
        if this.Tokens[i] !in tokens {
          toMask := (toMask + [this.Tokens[i]]);
        }
        i := i + 1;
      }
      if |toMask| > 0 {
        this.MaskTokens(toMask);
      }
    }

    predicate IsMasked(token: Token)
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
    {
      this.Logits[this.TokenToId(token)] == -1000000000.0
    }

    predicate HasUnmaskedToken()
      reads this
      reads this.Logits
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()
    {
      exists t: Token :: t in Tokens && !IsMasked(t)
    }

    method BiasToken(token: Token, delta: Logit)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      ensures ValidTokensIdsLogits()
      ensures -1000000000.0 <= Logits[TokenToId(token)] <= 1000000000.0
      ensures Logits[TokenToId(token)] == if old(Logits[TokenToId(token)]) + delta > 1000000000.0 then 1000000000.0 else if old(Logits[TokenToId(token)]) + delta < -1000000000.0 then -1000000000.0 else old(Logits[TokenToId(token)]) + delta
      ensures forall t: Token :: t in Tokens && t != token ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
    {
      var token_id := this.TokenToId(token);
      var raw := (this.Logits[token_id] + delta);
      if raw > 1000000000.0 {
        raw := 1000000000.0;
      }
      if raw < -1000000000.0 {
        raw := -1000000000.0;
      }
      this.Logits[token_id] := raw;
    }

    method BiasTokens(tokens: Prefix, delta: Logit)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires |tokens| > 0
      requires forall token :: token in tokens ==> token in Tokens
      ensures ValidTokensIdsLogits()
      ensures forall t :: t in Tokens && !(t in tokens) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
    {
      var n := |tokens|;
      var i := 0;
      while i < n
        invariant 0 <= i <= n
        invariant ValidTokensIdsLogits()
        invariant forall t :: t in Tokens && !(t in tokens[..i]) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
      {
        this.BiasToken(tokens[i], delta);
        i := i + 1;
      }
    }

    method ScaleToken(token: Token, factor: Logit)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires token in Tokens
      requires factor != 0.0
      ensures ValidTokensIdsLogits()
      ensures -1000000000.0 <= Logits[TokenToId(token)] <= 1000000000.0
      ensures forall t: Token :: t in Tokens && t != token ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
    {
      var token_id := this.TokenToId(token);
      var raw := (this.Logits[token_id] * factor);
      if raw > 1000000000.0 {
        raw := 1000000000.0;
      }
      if raw < -1000000000.0 {
        raw := -1000000000.0;
      }
      this.Logits[token_id] := raw;
    }

    method ScaleTokens(tokens: Prefix, factor: Logit)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires |tokens| > 0
      requires forall token :: token in tokens ==> token in Tokens
      requires factor != 0.0
      ensures ValidTokensIdsLogits()
      ensures forall t :: t in Tokens && !(t in tokens) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
    {
      var n := |tokens|;
      var i := 0;
      while i < n
        invariant 0 <= i <= n
        invariant ValidTokensIdsLogits()
        invariant forall t :: t in Tokens && !(t in tokens[..i]) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
      {
        this.ScaleToken(tokens[i], factor);
        i := i + 1;
      }
    }

    method {:extern} {:axiom} ClampLogits(low: Logit, high: Logit)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires -1000000000.0 <= low
      requires low <= high
      requires high <= 1000000000.0
      ensures ValidTokensIdsLogits()
      ensures forall id :: 0 <= id < Logits.Length ==> low <= Logits[id] <= high
      ensures forall id :: 0 <= id < Logits.Length ==> (old(Logits[id]) >= low && old(Logits[id]) <= high ==> Logits[id] == old(Logits[id]))

    method {:extern} {:axiom} TopKFilter(k: int)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires 1 <= k <= |Tokens|
      ensures ValidTokensIdsLogits()
      ensures HasUnmaskedToken()
      ensures forall t :: t in Tokens && !IsMasked(t) ==> !old(IsMasked(t))

    method {:extern} {:axiom} GenerateLogits(input: Prefix)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()

    method {:extern} {:axiom} ChooseNextToken() returns (token: Token)
      requires ValidTokensIdsLogits()
      ensures token in Tokens
      ensures !IsMasked(token)
      ensures ValidTokensIdsLogits()

  }

  class Parser {
    predicate {:extern} {:axiom} IsValidPrefix(prefix: Prefix)
      ensures forall k :: 0 <= k < |prefix| ==> IsValidPrefix(prefix[..k])

    lemma {:axiom} EmptyPrefixIsValid()
      ensures IsValidPrefix([])

    predicate {:extern} {:axiom} IsCompletePrefix(prefix: Prefix)
      ensures IsValidPrefix(prefix)

    predicate IsDeadPrefix(prefix: Prefix)
    {
      !IsCompletePrefix(prefix) && |ValidNextTokens(prefix)| == 0
    }

    predicate ValidNextToken(prefix: Prefix, token: Token)
      requires IsValidPrefix(prefix)
    {
      token in ValidNextTokens(prefix)
    }

    function {:extern} {:axiom} ValidNextTokens(prefix: Prefix): (result: seq<Token>)
      requires IsValidPrefix(prefix)
      ensures forall t :: t in ValidNextTokens(prefix) ==> IsValidPrefix(prefix + [t])
      ensures (IsCompletePrefix(prefix) || |ValidNextTokens(prefix)| > 0)

    function ValidContinuationCount(prefix: Prefix): (result: int)
      requires IsValidPrefix(prefix)
      ensures result >= 0
      ensures result == |ValidNextTokens(prefix)|
      ensures result == 0 ==> (IsCompletePrefix(prefix) || IsDeadPrefix(prefix))
    {
      |this.ValidNextTokens(prefix)|
    }

    function {:extern} {:axiom} ParserDistanceToComplete(prefix: Prefix): (result: int)
      requires IsValidPrefix(prefix)
      ensures result >= 0
      ensures IsCompletePrefix(prefix) ==> result == 0
      ensures !IsCompletePrefix(prefix) ==> result >= 1

  }

  class Delimiter {
    const Left: Token
    const Right: Token

    constructor (left: Token, right: Token)
      requires left != right
      ensures this.Left == left && this.Right == right
      ensures this.Left != this.Right
    {
      this.Left := left;
      this.Right := right;
    }

    function LastLeftDelimiterIndex(prefix: Prefix): (result: nat)
      ensures result <= |prefix|
      ensures result < |prefix| ==> prefix[result] == this.Left
      ensures result == |prefix| ==> forall i :: 0 <= i < |prefix| ==> prefix[i] != this.Left
      ensures result < |prefix| ==> forall i :: result < i < |prefix| ==> prefix[i] != this.Left
      decreases |prefix|
    {
      if |prefix| == 0 then 0
      else
      if prefix[|prefix|-1] == this.Left then |prefix|-1
      else
        var lastInRest := LastLeftDelimiterIndex(prefix[..|prefix|-1]);
        if lastInRest < |prefix|-1 then lastInRest else |prefix|
    }

    function FirstRightDelimiterIndex(content: Prefix): (result: nat)
      ensures result <= |content|
      ensures result < |content| ==> content[result] == this.Right
      ensures forall i :: 0 <= i < result ==> content[i] != this.Right
      decreases |content|
    {
      if |content| == 0 then 0
      else if content[0] == this.Right then 0
      else 1 + FirstRightDelimiterIndex(content[1..])
    }

    lemma NoFirstRightDelimiterIndexMeansNoRight(content: Prefix)
      requires FirstRightDelimiterIndex(content) == |content|
      ensures !PrefixContains(content, this.Right)
    {
      assert this.FirstRightDelimiterIndex(content) == |content|;
      assert !PrefixContains(content, this.Right);
    }

    function GetDelimitedContent(prefix: Prefix): (result: Prefix)
      ensures |GetDelimitedContent(prefix)| <= |prefix|
      ensures forall t: Token :: t in GetDelimitedContent(prefix) ==> t in prefix
    {
      var start := this.LastLeftDelimiterIndex(prefix) + 1;
      if start > |prefix| then []
      else
      var afterLeft := prefix[start..|prefix|];
      var endIdx := this.FirstRightDelimiterIndex(afterLeft);
      afterLeft[..endIdx]
    }

    predicate InsideDelimitedWindow(prefix: Prefix)
    {
      var start := LastLeftDelimiterIndex(prefix) + 1;
      start <= |prefix| && FirstRightDelimiterIndex(prefix[start..|prefix|]) == |prefix[start..|prefix|]|
    }

    lemma InsideDelimitedWindowNoRight(prefix: Prefix)
      requires InsideDelimitedWindow(prefix)
      ensures !PrefixContains(GetDelimitedContent(prefix), this.Right)
    {
      var start := (this.LastLeftDelimiterIndex(prefix) + 1);
      var after_left := prefix[start..];
      this.NoFirstRightDelimiterIndexMeansNoRight(after_left);
    }

    lemma {:axiom} GetDelimitedContentAppend(prefix: Prefix, next: Token)
      requires InsideDelimitedWindow(prefix)
      requires next != Right
      requires next != Left
      ensures GetDelimitedContent(prefix + [next]) == GetDelimitedContent(prefix) + [next]
      ensures next != Right ==> InsideDelimitedWindow(prefix + [next])

    lemma AppendLeftEntersWindow(prefix: Prefix)
      ensures InsideDelimitedWindow(prefix + [this.Left])
      ensures GetDelimitedContent(prefix + [this.Left]) == []
    {
      assert this.InsideDelimitedWindow((prefix + [this.Left]));
      assert this.GetDelimitedContent((prefix + [this.Left])) == [];
    }

    lemma FirstRightDelimiterAppendRight(content: Prefix)
      requires FirstRightDelimiterIndex(content) == |content|
      ensures FirstRightDelimiterIndex(content + [this.Right]) == |content|
    {
      if |content| == 0 {
      } else {
        this.FirstRightDelimiterAppendRight(content[1..]);
        assert (content + [this.Right])[1..] == content[1..] + [this.Right];
      }
    }

    lemma LastLeftDelimiterAppendNonLeft(prefix: Prefix, tok: Token)
      requires tok != this.Left
      ensures var oldIdx := LastLeftDelimiterIndex(prefix); var newIdx := LastLeftDelimiterIndex(prefix + [tok]); if oldIdx < |prefix| then newIdx == oldIdx else newIdx == |prefix + [tok]|
    {
      assert tok != this.Left;
      var old_idx := this.LastLeftDelimiterIndex(prefix);
      var new_idx := this.LastLeftDelimiterIndex((prefix + [tok]));
      if old_idx < |prefix| {
        assert new_idx == old_idx;
      } else {
        assert new_idx == |(prefix + [tok])|;
      }
    }

    lemma AppendRightExitsWindow(prefix: Prefix)
      requires InsideDelimitedWindow(prefix)
      requires this.Left != this.Right
      ensures !InsideDelimitedWindow(prefix + [this.Right])
    {
      assert this.InsideDelimitedWindow(prefix);
      assert this.Left != this.Right;
      assert !this.InsideDelimitedWindow((prefix + [this.Right]));
    }

  }

  class CSDHelpers {
    const lm: LM
    const parser: Parser

    constructor (lm: LM, parser: Parser)
      requires lm.ValidTokensIdsLogits()
      ensures this.lm == lm && this.parser == parser
      ensures lm.ValidTokensIdsLogits()
      ensures this.lm.ValidTokensIdsLogits()
    {
      this.lm := lm;
      this.parser := parser;
    }

    lemma {:axiom} AllValidNextTokensInLM(content: Prefix)
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(content)
      ensures lm.ValidTokensIdsLogits()
      ensures forall t: Token :: t in parser.ValidNextTokens(content) ==> t in lm.Tokens

    lemma {:axiom} ValidNextTokensInLMAfterStep(content: Prefix, next: Token)
      requires lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix(content)
      requires !parser.IsCompletePrefix(content)
      requires forall t: Token :: t in parser.ValidNextTokens(content) ==> t in lm.Tokens
      requires parser.IsValidPrefix(content + [next])
      ensures forall t: Token :: t in parser.ValidNextTokens(content + [next]) ==> t in lm.Tokens

    function LongestValidSuffix(prefix: Prefix): (result: Prefix)
      reads this
      reads this.parser
      requires parser.IsValidPrefix([])
      ensures parser.IsValidPrefix(result)
      ensures |result| <= |prefix|
      ensures |prefix| > 0 && parser.IsValidPrefix(prefix) ==> result == prefix
      ensures |prefix| == 0 ==> result == []
      ensures forall i :: 0 <= i < |result| ==> result[i] == prefix[|prefix| - |result| + i]
      decreases |prefix|
    {
      (if |prefix| == 0 then [] else (if this.parser.IsValidPrefix(prefix) then prefix else this.LongestValidSuffix(prefix[1..])))
    }

    lemma {:axiom} LongestValidSuffixAppend(prefix: Prefix, next: Token)
      requires parser.IsValidPrefix([])
      requires parser.IsValidPrefix(LongestValidSuffix(prefix))
      requires parser.ValidNextToken(LongestValidSuffix(prefix), next)
      ensures parser.IsValidPrefix(LongestValidSuffix(prefix) + [next])
      ensures |LongestValidSuffix(prefix + [next])| >= |LongestValidSuffix(prefix)| + 1

    lemma LongestValidSuffixIsValid(prefix: Prefix)
      requires parser.IsValidPrefix([])
      ensures parser.IsValidPrefix(LongestValidSuffix(prefix))
    {
      assert this.parser.IsValidPrefix(this.LongestValidSuffix(prefix));
    }

    lemma LongestValidSuffixNotDead(prefix: Prefix)
      requires parser.IsValidPrefix([])
      requires parser.IsValidPrefix(LongestValidSuffix(prefix))
      ensures parser.IsCompletePrefix(LongestValidSuffix(prefix)) || |parser.ValidNextTokens(LongestValidSuffix(prefix))| > 0
    {
      var suffix := this.LongestValidSuffix(prefix);
      assert ((this.parser.IsCompletePrefix(suffix)) || (|this.parser.ValidNextTokens(suffix)| > 0));
    }

    predicate CanConstrain(prefix: Prefix)
      reads this
      reads this.parser
      requires parser.IsValidPrefix([])
    {
      !this.parser.IsCompletePrefix(this.LongestValidSuffix(prefix))
    }

    predicate IsComplete(prefix: Prefix)
      reads this
      reads this.parser
      requires parser.IsValidPrefix([])
    {
      this.parser.IsCompletePrefix(this.LongestValidSuffix(prefix))
    }

    predicate IsDead(prefix: Prefix)
      reads this
      reads this.parser
      requires parser.IsValidPrefix([])
    {
      this.parser.IsDeadPrefix(this.LongestValidSuffix(prefix))
    }

    function ValidContinuationCount(prefix: Prefix): (result: int)
      reads this
      reads this.parser
      requires parser.IsValidPrefix([])
      ensures result >= 0
    {
      this.parser.ValidContinuationCount(this.LongestValidSuffix(prefix))
    }

    function ParserDistanceToComplete(prefix: Prefix): (result: int)
      reads this
      reads this.parser
      requires parser.IsValidPrefix([])
      ensures result >= 0
      ensures IsComplete(prefix) ==> result == 0
      ensures !IsComplete(prefix) ==> result >= 1
    {
      this.parser.ParserDistanceToComplete(this.LongestValidSuffix(prefix))
    }

    predicate IsLeftDelimiterToken(token: Token)
    {
      ((token == LeftDelimiter) || (token == SpacedLeftDelimiter))
    }

    predicate IsRightDelimiterToken(token: Token)
    {
      ((token == RightDelimiter) || (token == SpacedRightDelimiter))
    }

    predicate EndsWithLeftDelimiter(prefix: Prefix)
    {
      ((|prefix| > 0) && (this.IsLeftDelimiterToken(prefix[(|prefix| - 1)])))
    }

    predicate EndsWithRightDelimiter(prefix: Prefix)
    {
      ((|prefix| > 0) && (this.IsRightDelimiterToken(prefix[(|prefix| - 1)])))
    }

    predicate ContainsLeftDelimiter(prefix: Prefix)
    {
      ((PrefixContains(prefix, LeftDelimiter)) || (PrefixContains(prefix, SpacedLeftDelimiter)))
    }

    predicate ContainsRightDelimiter(prefix: Prefix)
    {
      ((PrefixContains(prefix, RightDelimiter)) || (PrefixContains(prefix, SpacedRightDelimiter)))
    }

    method LastTokenBefore(generated: Prefix, target: Token) returns (token: Token, found: bool)
      ensures found ==> token in generated
      ensures !found ==> token == ""
      decreases |generated|
    {
      var i := (|generated| - 1);
      while i >= 1
        invariant -1 <= i < |generated|
        decreases i + 1
      {
        if generated[i] == target {
          token := generated[(i - 1)];
          found := true;
          return;
        }
        i := i - 1;
      }
      token := "";
      found := false;
      return;
    }

    function CountOccurrences(generated: Prefix, target: Token): (result: int)
      ensures result >= 0
      decreases |generated|
    {
      (if |generated| == 0 then 0 else ((if generated[0] == target then 1 else 0) + this.CountOccurrences(generated[1..], target)))
    }

    function TokensSinceLastDelimiter(generated: Prefix): (result: int)
      ensures 0 <= result <= |generated|
      decreases |generated|
    {
      (if |generated| == 0 then 0 else (if ((generated[(|generated| - 1)] == LeftDelimiter) || (generated[(|generated| - 1)] == RightDelimiter) || (generated[(|generated| - 1)] == SpacedLeftDelimiter) || (generated[(|generated| - 1)] == SpacedRightDelimiter)) then 0 else (1 + this.TokensSinceLastDelimiter(generated[..|generated|-1]))))
    }

    method UnconstrainedStep(prompt: Prefix, generated: Prefix, stepsLeft: nat) returns (nextToken: Token, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures nextToken in lm.Tokens
      ensures !lm.IsMasked(nextToken)
    {
      this.lm.ValidTokensIdsLogitsAlways();
      this.lm.GenerateLogits((prompt + generated));
      if |this.lm.Tokens| > 4 {
        this.MaskAllDelimiters(generated);
      }
      var next_token := this.lm.ChooseNextToken();
      nextToken := next_token;
      remainingSteps := (stepsLeft - 1);
      return;
    }

    method UnconstrainedAllowLeftDelimiterStep(prompt: Prefix, generated: Prefix, stepsLeft: nat) returns (nextToken: Token, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures nextToken in lm.Tokens
      ensures !lm.IsMasked(nextToken)
    {
      this.lm.ValidTokensIdsLogitsAlways();
      this.lm.GenerateLogits((prompt + generated));
      if |this.lm.Tokens| > 4 {
        this.MaskRightDelimiters(generated);
      }
      var next_token := this.lm.ChooseNextToken();
      nextToken := next_token;
      remainingSteps := (stepsLeft - 1);
      return;
    }

    method {:extern} {:axiom} UnconstrainedBiasLeftDelimiterStep(prompt: Prefix, generated: Prefix, bias: Logit, stepsLeft: nat) returns (nextToken: Token, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires stepsLeft >= 1
      requires bias > 0.0
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures nextToken in lm.Tokens
      ensures !lm.IsMasked(nextToken)

    method UnconstrainedNudgeLeftDelimiterStep(prompt: Prefix, generated: Prefix, stepsLeft: nat) returns (nextToken: Token, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures nextToken in lm.Tokens
      ensures !lm.IsMasked(nextToken)
    {
      this.lm.ValidTokensIdsLogitsAlways();
      this.lm.GenerateLogits((prompt + generated));
      if |this.lm.Tokens| > 4 {
        this.MaskRightDelimiters(generated);
        this.BiasLeftDelimiters(5.0);
      }
      var next_token := this.lm.ChooseNextToken();
      nextToken := next_token;
      remainingSteps := (stepsLeft - 1);
      return;
    }

    method ConstrainedStep(prompt: Prefix, generated: Prefix, stepsLeft: nat) returns (nextToken: Token, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires !parser.IsCompletePrefix(LongestValidSuffix(generated))
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures nextToken in lm.Tokens
      ensures !lm.IsMasked(nextToken)
      ensures parser.ValidNextToken(LongestValidSuffix(generated), nextToken)
      ensures parser.IsValidPrefix(LongestValidSuffix(generated) + [nextToken])
      ensures |LongestValidSuffix(generated + [nextToken])| >= |LongestValidSuffix(generated)| + 1
    {
      this.LongestValidSuffixIsValid(generated);
      var suffix := this.LongestValidSuffix(generated);
      this.AllValidNextTokensInLM(suffix);
      this.lm.GenerateLogits((prompt + generated));
      this.lm.MaskTokensExcept(this.parser.ValidNextTokens(suffix));
      var next_token := this.lm.ChooseNextToken();
      this.LongestValidSuffixAppend(generated, next_token);
      nextToken := next_token;
      remainingSteps := (stepsLeft - 1);
      return;
    }

    method ConstrainedOrRightDelimiterStep(prompt: Prefix, generated: Prefix, stepsLeft: nat) returns (nextToken: Token, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires RightDelimiter in lm.Tokens
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures nextToken in lm.Tokens
      ensures !lm.IsMasked(nextToken)
      ensures (nextToken == RightDelimiter || nextToken == SpacedRightDelimiter) ==> parser.IsCompletePrefix(LongestValidSuffix(generated))
      ensures (nextToken != RightDelimiter && nextToken != SpacedRightDelimiter) ==> parser.ValidNextToken(LongestValidSuffix(generated), nextToken)
    {
      this.LongestValidSuffixIsValid(generated);
      var suffix := this.LongestValidSuffix(generated);
      this.AllValidNextTokensInLM(suffix);
      this.lm.GenerateLogits((prompt + generated));
      var valid_tokens := this.parser.ValidNextTokens(suffix);
      if this.parser.IsCompletePrefix(suffix) {
        if SpacedRightDelimiter in this.lm.Tokens {
          this.lm.MaskTokensExcept((valid_tokens + [RightDelimiter, SpacedRightDelimiter]));
        } else {
          this.lm.MaskTokensExcept((valid_tokens + [RightDelimiter]));
        }
      } else {
        this.lm.MaskTokensExcept(valid_tokens);
      }
      var next_token := this.lm.ChooseNextToken();
      assume {:axiom} next_token in lm.Tokens;
      assume {:axiom} !lm.IsMasked(next_token);
      if next_token == RightDelimiter || next_token == SpacedRightDelimiter {
        assume {:axiom} parser.IsCompletePrefix(LongestValidSuffix(generated));
      } else {
        assume {:axiom} parser.ValidNextToken(LongestValidSuffix(generated), next_token);
      }
      nextToken := next_token;
      remainingSteps := (stepsLeft - 1);
      return;
    }

    method {:extern} {:axiom} SoftConstrainedStep(prompt: Prefix, generated: Prefix, penalty: Logit, stepsLeft: int) returns (nextToken: Token, remainingSteps: int)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires stepsLeft >= 1
      requires penalty > 0.0
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures nextToken in lm.Tokens
      ensures !lm.IsMasked(nextToken)

    method {:extern} {:axiom} TopKConstrainedStep(prompt: Prefix, generated: Prefix, k: int, stepsLeft: int) returns (nextToken: Token, remainingSteps: int)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires stepsLeft >= 1
      requires 1 <= k <= |lm.Tokens|
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures nextToken in lm.Tokens
      ensures !lm.IsMasked(nextToken)
      ensures parser.ValidNextToken(LongestValidSuffix(generated), nextToken)
      ensures parser.IsValidPrefix(LongestValidSuffix(generated) + [nextToken])

    method ForcedTokenStep(prompt: Prefix, generated: Prefix, token: Token, stepsLeft: nat) returns (nextToken: Token, remainingSteps: nat)
      requires this.lm.ValidTokensIdsLogits()
      requires token in lm.Tokens
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures nextToken == token
      ensures nextToken in lm.Tokens
    {
      this.lm.ValidTokensIdsLogitsAlways();
      nextToken := token;
      remainingSteps := (stepsLeft - 1);
      return;
    }

    method {:extern} {:axiom} SoftConstrainToGrammar(prefix: Prefix, penalty: Logit)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires penalty > 0.0
      ensures this.lm.ValidTokensIdsLogits()
      ensures forall t :: t in parser.ValidNextTokens(LongestValidSuffix(prefix)) && t in lm.Tokens ==> lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])

    method {:extern} {:axiom} IntersectWithGrammar(prefix: Prefix)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      ensures this.lm.ValidTokensIdsLogits()
      ensures forall t :: t in lm.Tokens && !(t in parser.ValidNextTokens(LongestValidSuffix(prefix))) ==> lm.IsMasked(t)
      ensures forall t :: t in parser.ValidNextTokens(LongestValidSuffix(prefix)) && t in lm.Tokens ==> lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])

    method BiasForCompletion(prefix: Prefix, bonus: Logit)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires bonus > 0.0
      ensures this.lm.ValidTokensIdsLogits()
    {
      this.LongestValidSuffixIsValid(prefix);
      var suffix := this.LongestValidSuffix(prefix);
      if this.parser.IsCompletePrefix(suffix) {
        return;
      }
      var valid_next := this.parser.ValidNextTokens(suffix);
      this.AllValidNextTokensInLM(suffix);
      var n := |valid_next|;
      var i := 0;
      while i < n
        invariant 0 <= i <= n
        invariant lm.ValidTokensIdsLogits()
      {
        if this.parser.IsCompletePrefix((suffix + [valid_next[i]])) {
          this.lm.BiasToken(valid_next[i], bonus);
        }
        i := i + 1;
      }
    }

    method MaskAllDelimiters(generated: Prefix)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      ensures this.lm.ValidTokensIdsLogits()
    {
      if LeftDelimiter in this.lm.Tokens {
        this.lm.MaskToken(LeftDelimiter);
      }
      if RightDelimiter in this.lm.Tokens {
        this.lm.MaskToken(RightDelimiter);
      }
      if SpacedLeftDelimiter in this.lm.Tokens {
        this.lm.MaskToken(SpacedLeftDelimiter);
      }
      if SpacedRightDelimiter in this.lm.Tokens {
        this.lm.MaskToken(SpacedRightDelimiter);
      }
    }

    method MaskRightDelimiters(generated: Prefix)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      ensures this.lm.ValidTokensIdsLogits()
    {
      if RightDelimiter in this.lm.Tokens {
        this.lm.MaskToken(RightDelimiter);
      }
      if SpacedRightDelimiter in this.lm.Tokens {
        this.lm.MaskToken(SpacedRightDelimiter);
      }
    }

    method BiasLeftDelimiters(bias: Logit)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires bias > 0.0
      ensures this.lm.ValidTokensIdsLogits()
    {
      if LeftDelimiter in this.lm.Tokens {
        this.lm.BiasToken(LeftDelimiter, bias);
      }
      if SpacedLeftDelimiter in this.lm.Tokens {
        this.lm.BiasToken(SpacedLeftDelimiter, bias);
      }
    }

    method BiasRightDelimiters(bias: Logit)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires bias > 0.0
      ensures this.lm.ValidTokensIdsLogits()
    {
      if RightDelimiter in this.lm.Tokens {
        this.lm.BiasToken(RightDelimiter, bias);
      }
      if SpacedRightDelimiter in this.lm.Tokens {
        this.lm.BiasToken(SpacedRightDelimiter, bias);
      }
    }

    method MaskLeftDelimiters(generated: Prefix)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      ensures this.lm.ValidTokensIdsLogits()
    {
      if LeftDelimiter in this.lm.Tokens {
        this.lm.MaskToken(LeftDelimiter);
      }
      if SpacedLeftDelimiter in this.lm.Tokens {
        this.lm.MaskToken(SpacedLeftDelimiter);
      }
    }

    method AppendUnconstrainedStep(prompt: Prefix, prefix: Prefix, stepsLeft: nat) returns (updated: Prefix, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures |updated| == |prefix| + 1
      ensures |updated| + remainingSteps == |prefix| + stepsLeft
      ensures updated[|prefix|] in lm.Tokens
    {
      var next_token, remaining := this.UnconstrainedStep(prompt, prefix, stepsLeft);
      updated := (prefix + [next_token]);
      remainingSteps := remaining;
      return;
    }

    method AppendUnconstrainedAllowLeftDelimiterStep(prompt: Prefix, prefix: Prefix, stepsLeft: nat) returns (updated: Prefix, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures |updated| == |prefix| + 1
      ensures |updated| + remainingSteps == |prefix| + stepsLeft
      ensures updated[|prefix|] in lm.Tokens
    {
      var next_token, remaining := this.UnconstrainedAllowLeftDelimiterStep(prompt, prefix, stepsLeft);
      updated := (prefix + [next_token]);
      remainingSteps := remaining;
      return;
    }

    method AppendUnconstrainedNudgeLeftDelimiterStep(prompt: Prefix, prefix: Prefix, stepsLeft: nat) returns (updated: Prefix, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures |updated| == |prefix| + 1
      ensures |updated| + remainingSteps == |prefix| + stepsLeft
      ensures updated[|prefix|] in lm.Tokens
    {
      var next_token, remaining := this.UnconstrainedNudgeLeftDelimiterStep(prompt, prefix, stepsLeft);
      updated := (prefix + [next_token]);
      remainingSteps := remaining;
      return;
    }

    method AppendConstrainedStep(prompt: Prefix, prefix: Prefix, stepsLeft: nat) returns (updated: Prefix, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires !parser.IsCompletePrefix(LongestValidSuffix(prefix))
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures |updated| == |prefix| + 1
      ensures |updated| + remainingSteps == |prefix| + stepsLeft
      ensures updated[|prefix|] in lm.Tokens
      ensures parser.ValidNextToken(LongestValidSuffix(prefix), updated[|prefix|])
      ensures parser.IsValidPrefix(LongestValidSuffix(prefix) + [updated[|prefix|]])
    {
      var next_token, remaining := this.ConstrainedStep(prompt, prefix, stepsLeft);
      updated := (prefix + [next_token]);
      remainingSteps := remaining;
      return;
    }

    method AppendConstrainedOrRightDelimiterStep(prompt: Prefix, prefix: Prefix, stepsLeft: nat) returns (updated: Prefix, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires RightDelimiter in lm.Tokens
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures |updated| == |prefix| + 1
      ensures |updated| + remainingSteps == |prefix| + stepsLeft
      ensures updated[|prefix|] in lm.Tokens
      ensures (updated[|prefix|] == RightDelimiter || updated[|prefix|] == SpacedRightDelimiter) ==> parser.IsCompletePrefix(LongestValidSuffix(prefix))
      ensures (updated[|prefix|] != RightDelimiter && updated[|prefix|] != SpacedRightDelimiter) ==> parser.ValidNextToken(LongestValidSuffix(prefix), updated[|prefix|])
    {
      var next_token, remaining := this.ConstrainedOrRightDelimiterStep(prompt, prefix, stepsLeft);
      updated := (prefix + [next_token]);
      remainingSteps := remaining;
      return;
    }

    method {:extern} {:axiom} AppendSoftConstrainedStep(prompt: Prefix, prefix: Prefix, penalty: Logit, stepsLeft: int) returns (updated: Prefix, remainingSteps: int)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires stepsLeft >= 1
      requires penalty > 0.0
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures |updated| == |prefix| + 1
      ensures |updated| + remainingSteps == |prefix| + stepsLeft
      ensures updated[|prefix|] in lm.Tokens

    method {:extern} {:axiom} AppendTopKConstrainedStep(prompt: Prefix, prefix: Prefix, k: int, stepsLeft: int) returns (updated: Prefix, remainingSteps: int)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires stepsLeft >= 1
      requires 1 <= k <= |lm.Tokens|
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures |updated| == |prefix| + 1
      ensures |updated| + remainingSteps == |prefix| + stepsLeft
      ensures updated[|prefix|] in lm.Tokens
      ensures parser.ValidNextToken(LongestValidSuffix(prefix), updated[|prefix|])

    method AppendForcedToken(prefix: Prefix, token: Token, stepsLeft: nat) returns (updated: Prefix, remainingSteps: nat)
      requires this.lm.ValidTokensIdsLogits()
      requires token in lm.Tokens
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures updated == prefix + [token]
      ensures |updated| + remainingSteps == |prefix| + stepsLeft
    {
      var next_token, remaining_steps := this.ForcedTokenStep([], prefix, token, stepsLeft);
      updated := (prefix + [next_token]);
      remainingSteps := remaining_steps;
      return;
    }

    method AppendLeftDelimiter(prefix: Prefix, stepsLeft: nat) returns (updated: Prefix, remainingSteps: nat)
      requires this.lm.ValidTokensIdsLogits()
      requires LeftDelimiter in lm.Tokens
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures updated == prefix + [LeftDelimiter]
      ensures |updated| + remainingSteps == |prefix| + stepsLeft
    {
      updated, remainingSteps := this.AppendForcedToken(prefix, LeftDelimiter, stepsLeft);
    }

    method AppendRightDelimiter(prefix: Prefix, stepsLeft: nat) returns (updated: Prefix, remainingSteps: nat)
      requires this.lm.ValidTokensIdsLogits()
      requires RightDelimiter in lm.Tokens
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures updated == prefix + [RightDelimiter]
      ensures |updated| + remainingSteps == |prefix| + stepsLeft
    {
      updated, remainingSteps := this.AppendForcedToken(prefix, RightDelimiter, stepsLeft);
    }

    function ValidTokenCount(prefix: Prefix): (result: int)
      reads this
      reads this.parser
      requires parser.IsValidPrefix([])
      ensures result >= 0
    {
      this.ValidContinuationCount(prefix)
    }

    method OpenConstrainedSpan(prefix: Prefix, stepsLeft: int) returns (updated: Prefix, insideSpan: bool, currentConstrained: Prefix, remainingSteps: int)
      requires this.lm.ValidTokensIdsLogits()
      requires LeftDelimiter in lm.Tokens
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
    {
      updated, remainingSteps := this.AppendLeftDelimiter(prefix, stepsLeft);
      updated := updated;
      insideSpan := true;
      currentConstrained := [];
      remainingSteps := remainingSteps;
      return;
    }

    method CloseConstrainedSpan(prefix: Prefix, currentConstrained: Prefix, stepsLeft: int) returns (updated: Prefix, insideSpan: bool, updatedConstrained: Prefix, remainingSteps: int)
      requires this.lm.ValidTokensIdsLogits()
      requires RightDelimiter in lm.Tokens
      requires stepsLeft >= 1
      requires parser.IsValidPrefix([])
      requires parser.IsValidPrefix(currentConstrained)
      requires parser.IsCompletePrefix(currentConstrained)
      requires |currentConstrained| <= |prefix|
      requires prefix[|prefix| - |currentConstrained|..] == currentConstrained
      ensures this.lm.ValidTokensIdsLogits()
    {
      updated, remainingSteps := this.AppendRightDelimiter(prefix, stepsLeft);
      updated := updated;
      insideSpan := false;
      updatedConstrained := [];
      remainingSteps := remainingSteps;
      return;
    }

    method AppendConstrainedToken(prefix: Prefix, currentConstrained: Prefix, token: Token) returns (updated: Prefix, insideSpan: bool, updatedConstrained: Prefix)
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires parser.IsValidPrefix(currentConstrained)
      requires !parser.IsCompletePrefix(currentConstrained)
      requires token in lm.Tokens
      requires parser.ValidNextToken(currentConstrained, token)
      ensures this.lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(currentConstrained + [token])
    {
      updated := (prefix + [token]);
      var updated_constrained := (currentConstrained + [token]);
      updated := updated;
      insideSpan := true;
      updatedConstrained := updated_constrained;
      return;
    }

    method AdaptiveConstrainedStep(prompt: Prefix, stablePrefix: Prefix, currentConstrained: Prefix, validTokenGroups: seq<seq<Token>>, bonus: Logit, narrowThreshold: int, eosToken: Token, stepsLeft: int) returns (nextToken: Token, remainingSteps: int)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires parser.IsValidPrefix(currentConstrained)
      requires !parser.IsCompletePrefix(currentConstrained)
      requires stepsLeft >= 1
      requires bonus > 0.0
      requires narrowThreshold >= 0
      requires eosToken in lm.Tokens
      requires |validTokenGroups| >= 0
      requires forall g: seq<Token> :: g in validTokenGroups ==> forall t: Token :: t in g ==> t in lm.Tokens
      requires forall t: Token :: t in parser.ValidNextTokens(currentConstrained) ==> t in lm.Tokens
      ensures this.lm.ValidTokensIdsLogits()
    {
      this.lm.GenerateLogits(((prompt + stablePrefix) + currentConstrained));
      var valid_tokens := this.parser.ValidNextTokens(currentConstrained);
      this.lm.MaskTokensExcept(valid_tokens);
      if this.parser.ValidContinuationCount(currentConstrained) > narrowThreshold {
        var group_index := 0;
        while group_index < |validTokenGroups|
          invariant 0 <= group_index <= |validTokenGroups|
          invariant lm.ValidTokensIdsLogits()
        {
          var group := validTokenGroups[group_index];
          var token_index := 0;
          while token_index < |group|
            invariant 0 <= token_index <= |group|
            invariant lm.ValidTokensIdsLogits()
          {
            var token := group[token_index];
            if token in valid_tokens {
              this.lm.BiasToken(token, bonus);
            }
            token_index := token_index + 1;
          }
          group_index := group_index + 1;
        }
      }
      var next_token := this.lm.ChooseNextToken();
      if next_token == eosToken {
        nextToken := eosToken;
        remainingSteps := (stepsLeft - 1);
        return;
      }
      nextToken := next_token;
      remainingSteps := (stepsLeft - 1);
      return;
    }

    method GroupBoostedConstrainedStep(prompt: Prefix, stablePrefix: Prefix, currentConstrained: Prefix, validTokenGroups: seq<seq<Token>>, bonus: Logit, stepsLeft: int) returns (nextToken: Token, remainingSteps: int)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires parser.IsValidPrefix(currentConstrained)
      requires !parser.IsCompletePrefix(currentConstrained)
      requires stepsLeft >= 1
      requires bonus > 0.0
      requires |validTokenGroups| >= 0
      requires forall g: seq<Token> :: g in validTokenGroups ==> forall t: Token :: t in g ==> t in lm.Tokens
      requires forall t: Token :: t in parser.ValidNextTokens(currentConstrained) ==> t in lm.Tokens
      ensures this.lm.ValidTokensIdsLogits()
    {
      this.lm.GenerateLogits(((prompt + stablePrefix) + currentConstrained));
      this.lm.MaskTokensExcept(this.parser.ValidNextTokens(currentConstrained));
      var next_token := this.lm.ChooseNextToken();
      nextToken := next_token;
      remainingSteps := (stepsLeft - 1);
      return;
    }

    method PenalizedConstrainedStep(prompt: Prefix, stablePrefix: Prefix, currentConstrained: Prefix, penaltyTokens: Prefix, penalty: Logit, stepsLeft: int) returns (nextToken: Token, remainingSteps: int)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires parser.IsValidPrefix(currentConstrained)
      requires !parser.IsCompletePrefix(currentConstrained)
      requires stepsLeft >= 1
      requires penalty > 0.0
      requires |penaltyTokens| > 0
      requires forall t: Token :: t in penaltyTokens ==> t in lm.Tokens
      requires forall t: Token :: t in parser.ValidNextTokens(currentConstrained) ==> t in lm.Tokens
      ensures this.lm.ValidTokensIdsLogits()
    {
      this.lm.GenerateLogits(((prompt + stablePrefix) + currentConstrained));
      this.lm.MaskTokensExcept(this.parser.ValidNextTokens(currentConstrained));
      var next_token := this.lm.ChooseNextToken();
      nextToken := next_token;
      remainingSteps := (stepsLeft - 1);
      return;
    }

    function Checkpoint(prefix: Prefix): (result: Prefix)
      ensures result == prefix
    {
      prefix
    }

    function RestoreCheckpoint(checkpoint: Prefix): (result: Prefix)
      ensures result == checkpoint
    {
      checkpoint
    }

    function RestoreIfDead(prefix: Prefix, checkpoint: Prefix): (result: Prefix)
      reads this
      reads this.parser
      requires parser.IsValidPrefix([])
      ensures IsDead(prefix) ==> result == checkpoint
      ensures !IsDead(prefix) ==> result == prefix
    {
      (if this.IsDead(prefix) then checkpoint else prefix)
    }

    predicate HasBudget(stepsLeft: int, needed: int)
    {
      stepsLeft >= needed
    }

    function MinStepsToComplete(prefix: Prefix): (result: int)
      reads this
      reads this.parser
      requires parser.IsValidPrefix([])
      ensures result >= 0
      ensures IsComplete(prefix) ==> result == 0
      ensures !IsComplete(prefix) ==> result >= 1
    {
      this.ParserDistanceToComplete(prefix)
    }

  }

}
