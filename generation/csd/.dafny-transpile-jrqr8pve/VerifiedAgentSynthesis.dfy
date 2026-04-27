module VerifiedDecoderAgent {
  type Token = string
  type Prefix = seq<Token>
  type Id = nat
  type Logit = real

  const LeftDelimiter: Token := "<<"
  const RightDelimiter: Token := ">>"

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
      (forall i :: 0 <= i < Logits.Length ==> Logits[i] <= 1e9 && Logits[i] >= -1e9)
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
      ensures -1e9 <= Logits[TokenToId(token)] <= 1e9
      ensures Logits[TokenToId(token)] == if old(Logits[TokenToId(token)]) + delta > 1e9 then 1e9 else if old(Logits[TokenToId(token)]) + delta < -1e9 then -1e9 else old(Logits[TokenToId(token)]) + delta
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
      ensures -1e9 <= Logits[TokenToId(token)] <= 1e9
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

    method ClampLogits(low: Logit, high: Logit)
      modifies this.Logits
      requires ValidTokensIdsLogits()
      requires -1e9 <= low
      requires low <= high
      requires high <= 1e9
      ensures ValidTokensIdsLogits()
      ensures forall id :: 0 <= id < Logits.Length ==> low <= Logits[id] <= high
    {
      var n := this.Logits.Length;
      var i := 0;
      while i < n
        invariant 0 <= i <= n
        invariant ValidTokensIdsLogits()
        invariant forall j :: 0 <= j < i ==> low <= Logits[j] <= high
        invariant forall j :: i <= j < n ==> Logits[j] == old(Logits[j])
      {
        if this.Logits[i] > high {
          this.Logits[i] := high;
        }
        if this.Logits[i] < low {
          this.Logits[i] := low;
        }
        i := i + 1;
      }
    }

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

    function {:axiom} LongestValidSuffix(prefix: Prefix): (result: Prefix)
      reads this
      reads this.parser
      requires parser.IsValidPrefix([])
      ensures parser.IsValidPrefix(result)
      ensures |result| <= |prefix|
      ensures |prefix| > 0 && parser.IsValidPrefix(prefix) ==> result == prefix
      ensures |prefix| == 0 ==> result == []
      ensures forall i :: 0 <= i < |result| ==> result[i] == prefix[|prefix| - |result| + i]
      decreases |prefix|

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

    method UnconstrainedStep(prompt: Prefix, generated: Prefix, stepsLeft: nat) returns (next: Token, stepsLeft': nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures stepsLeft' == stepsLeft - 1
      ensures stepsLeft' >= 0
      ensures next in lm.Tokens
      ensures !lm.IsMasked(next)
    {
      this.lm.ValidTokensIdsLogitsAlways();
      this.lm.GenerateLogits((prompt + generated));
      var next_token := this.lm.ChooseNextToken();
      next := next_token;
      stepsLeft' := (stepsLeft - 1);
      return;
    }

    method ConstrainedStep(prompt: Prefix, generated: Prefix, stepsLeft: nat) returns (next: Token, stepsLeft': nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires !parser.IsCompletePrefix(LongestValidSuffix(generated))
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures stepsLeft' == stepsLeft - 1
      ensures stepsLeft' >= 0
      ensures next in lm.Tokens
      ensures !lm.IsMasked(next)
      ensures parser.ValidNextToken(LongestValidSuffix(generated), next)
      ensures parser.IsValidPrefix(LongestValidSuffix(generated) + [next])
      ensures |LongestValidSuffix(generated + [next])| >= |LongestValidSuffix(generated)| + 1
    {
      this.LongestValidSuffixIsValid(generated);
      var suffix := this.LongestValidSuffix(generated);
      this.AllValidNextTokensInLM(suffix);
      this.lm.GenerateLogits((prompt + generated));
      this.lm.MaskTokensExcept(this.parser.ValidNextTokens(suffix));
      var next_token := this.lm.ChooseNextToken();
      this.LongestValidSuffixAppend(generated, next_token);
      next := next_token;
      stepsLeft' := (stepsLeft - 1);
      return;
    }

    method SoftConstrainedStep(prompt: Prefix, generated: Prefix, penalty: Logit, stepsLeft: nat) returns (next: Token, stepsLeft': nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires !parser.IsCompletePrefix(LongestValidSuffix(generated))
      requires stepsLeft >= 1
      requires penalty > 0.0
      ensures this.lm.ValidTokensIdsLogits()
      ensures stepsLeft' == stepsLeft - 1
      ensures stepsLeft' >= 0
      ensures next in lm.Tokens
      ensures !lm.IsMasked(next)
    {
      this.LongestValidSuffixIsValid(generated);
      var suffix := this.LongestValidSuffix(generated);
      this.AllValidNextTokensInLM(suffix);
      var valid_tokens := this.parser.ValidNextTokens(suffix);
      var invalid_tokens: Prefix := [];
      var n := |this.lm.Tokens|;
      var i := 0;
      while i < n
        invariant 0 <= i <= n
        invariant lm.ValidTokensIdsLogits()
        invariant forall j :: 0 <= j < i && !(lm.Tokens[j] in valid_tokens) ==> lm.Tokens[j] in invalid_tokens
        invariant forall t :: t in invalid_tokens ==> t in lm.Tokens && !(t in valid_tokens)
      {
        if this.lm.Tokens[i] !in valid_tokens {
          invalid_tokens := (invalid_tokens + [this.lm.Tokens[i]]);
        }
        i := i + 1;
      }
      this.lm.GenerateLogits((prompt + generated));
      if |invalid_tokens| > 0 {
        this.lm.BiasTokens(invalid_tokens, -penalty);
      }
      var next_token := this.lm.ChooseNextToken();
      next := next_token;
      stepsLeft' := (stepsLeft - 1);
      return;
    }

    method TopKConstrainedStep(prompt: Prefix, generated: Prefix, k: int, stepsLeft: nat) returns (next: Token, stepsLeft': nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires !parser.IsCompletePrefix(LongestValidSuffix(generated))
      requires stepsLeft >= 1
      requires 1 <= k <= |lm.Tokens|
      ensures this.lm.ValidTokensIdsLogits()
      ensures stepsLeft' == stepsLeft - 1
      ensures stepsLeft' >= 0
      ensures next in lm.Tokens
      ensures !lm.IsMasked(next)
      ensures parser.ValidNextToken(LongestValidSuffix(generated), next)
      ensures parser.IsValidPrefix(LongestValidSuffix(generated) + [next])
    {
      this.LongestValidSuffixIsValid(generated);
      this.lm.GenerateLogits((prompt + generated));
      this.lm.TopKFilter(k);
      var suffix := this.LongestValidSuffix(generated);
      this.AllValidNextTokensInLM(suffix);
      this.lm.MaskTokensExcept(this.parser.ValidNextTokens(suffix));
      var next_token := this.lm.ChooseNextToken();
      this.LongestValidSuffixAppend(generated, next_token);
      next := next_token;
      stepsLeft' := (stepsLeft - 1);
      return;
    }

    method ForcedTokenStep(prompt: Prefix, generated: Prefix, token: Token, stepsLeft: nat) returns (next: Token, stepsLeft': nat)
      requires this.lm.ValidTokensIdsLogits()
      requires token in lm.Tokens
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures stepsLeft' == stepsLeft - 1
      ensures stepsLeft' >= 0
      ensures next == token
      ensures next in lm.Tokens
    {
      this.lm.ValidTokensIdsLogitsAlways();
      next := token;
      stepsLeft' := (stepsLeft - 1);
      return;
    }

    method BudgetAwareStep(prompt: Prefix, generated: Prefix, stepsLeft: nat, completionThreshold: nat) returns (next: Token, stepsLeft': nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires stepsLeft >= 1
      requires completionThreshold >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures stepsLeft' == stepsLeft - 1
      ensures stepsLeft' >= 0
      ensures next in lm.Tokens
      ensures !lm.IsMasked(next)
      ensures stepsLeft <= completionThreshold && !parser.IsCompletePrefix(LongestValidSuffix(generated)) ==> parser.ValidNextToken(LongestValidSuffix(generated), next)
    {
      var suffix := this.LongestValidSuffix(generated);
      var next_token := this.lm.Tokens[0];
      var steps_left_prime := (stepsLeft - 1);
      if ((stepsLeft <= completionThreshold) && (!this.parser.IsCompletePrefix(suffix))) {
        next_token, steps_left_prime := this.ConstrainedStep(prompt, generated, stepsLeft);
      } else {
        next_token, steps_left_prime := this.UnconstrainedStep(prompt, generated, stepsLeft);
      }
      next := next_token;
      stepsLeft' := steps_left_prime;
      return;
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
      ensures !lm.IsMasked(updated[|prefix|])
    {
      var next_token, remaining_steps := this.UnconstrainedStep(prompt, prefix, stepsLeft);
      updated := (prefix + [next_token]);
      remainingSteps := remaining_steps;
      return;
    }

    method AppendConstrainedStep(prompt: Prefix, prefix: Prefix, stepsLeft: nat) returns (updated: Prefix, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires CanConstrain(prefix)
      requires stepsLeft >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures |updated| == |prefix| + 1
      ensures |updated| + remainingSteps == |prefix| + stepsLeft
      ensures updated[|prefix|] in lm.Tokens
      ensures !lm.IsMasked(updated[|prefix|])
      ensures parser.ValidNextToken(LongestValidSuffix(prefix), updated[|prefix|])
      ensures parser.IsValidPrefix(LongestValidSuffix(prefix) + [updated[|prefix|]])
    {
      var next_token, remaining_steps := this.ConstrainedStep(prompt, prefix, stepsLeft);
      updated := (prefix + [next_token]);
      remainingSteps := remaining_steps;
      return;
    }

    method AppendSoftConstrainedStep(prompt: Prefix, prefix: Prefix, penalty: Logit, stepsLeft: nat) returns (updated: Prefix, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires CanConstrain(prefix)
      requires stepsLeft >= 1
      requires penalty > 0.0
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures |updated| == |prefix| + 1
      ensures |updated| + remainingSteps == |prefix| + stepsLeft
      ensures updated[|prefix|] in lm.Tokens
      ensures !lm.IsMasked(updated[|prefix|])
    {
      var next_token, remaining_steps := this.SoftConstrainedStep(prompt, prefix, penalty, stepsLeft);
      updated := (prefix + [next_token]);
      remainingSteps := remaining_steps;
      return;
    }

    method AppendTopKConstrainedStep(prompt: Prefix, prefix: Prefix, k: int, stepsLeft: nat) returns (updated: Prefix, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires CanConstrain(prefix)
      requires stepsLeft >= 1
      requires 1 <= k <= |lm.Tokens|
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures |updated| == |prefix| + 1
      ensures |updated| + remainingSteps == |prefix| + stepsLeft
      ensures updated[|prefix|] in lm.Tokens
      ensures !lm.IsMasked(updated[|prefix|])
      ensures parser.ValidNextToken(LongestValidSuffix(prefix), updated[|prefix|])
      ensures parser.IsValidPrefix(LongestValidSuffix(prefix) + [updated[|prefix|]])
    {
      var next_token, remaining_steps := this.TopKConstrainedStep(prompt, prefix, k, stepsLeft);
      updated := (prefix + [next_token]);
      remainingSteps := remaining_steps;
      return;
    }

    method AppendBudgetAwareStep(prompt: Prefix, prefix: Prefix, stepsLeft: nat, completionThreshold: nat) returns (updated: Prefix, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires stepsLeft >= 1
      requires completionThreshold >= 1
      ensures this.lm.ValidTokensIdsLogits()
      ensures remainingSteps == stepsLeft - 1
      ensures remainingSteps >= 0
      ensures |updated| == |prefix| + 1
      ensures |updated| + remainingSteps == |prefix| + stepsLeft
      ensures updated[|prefix|] in lm.Tokens
      ensures !lm.IsMasked(updated[|prefix|])
      ensures stepsLeft <= completionThreshold && CanConstrain(prefix) ==> parser.ValidNextToken(LongestValidSuffix(prefix), updated[|prefix|])
    {
      var next_token, remaining_steps := this.BudgetAwareStep(prompt, prefix, stepsLeft, completionThreshold);
      updated := (prefix + [next_token]);
      remainingSteps := remaining_steps;
      return;
    }

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

    method RollbackToValidPrefix(generated: Prefix) returns (repaired: Prefix)
      requires parser.IsValidPrefix([])
      ensures parser.IsValidPrefix(repaired)
      ensures |repaired| <= |generated|
      ensures !parser.IsDeadPrefix(repaired)
      ensures |repaired| == 0 ==> parser.IsValidPrefix([])
      ensures forall i :: 0 <= i < |repaired| ==> repaired[i] == generated[i]
    {
      repaired := generated;
      while ((|repaired| > 0) && (((!this.parser.IsValidPrefix(repaired)) || (this.parser.IsDeadPrefix(repaired)))))
        invariant |repaired| <= |generated|
        invariant parser.IsValidPrefix(repaired) || |repaired| > 0
        invariant forall i :: 0 <= i < |repaired| ==> repaired[i] == generated[i]
      {
        repaired := repaired[..|repaired|-1];
      }
      repaired := repaired;
      return;
    }

    method {:axiom} FindLongestValidSpan(generated: Prefix) returns (result: Prefix)
      requires parser.IsValidPrefix([])
      ensures parser.IsValidPrefix(result)
      ensures |result| <= |generated|
      ensures forall t :: t in result ==> t in generated
      ensures |generated| > 0 ==> |result| >= 0

    method {:axiom} ExtractAllValidSpans(generated: Prefix) returns (result: seq<Prefix>)
      requires parser.IsValidPrefix([])
      ensures forall span :: span in result ==> parser.IsValidPrefix(span)
      ensures forall span :: span in result ==> |span| > 0
      ensures forall span :: span in result ==> (forall t :: t in span ==> t in generated)

    method {:axiom} RepairByRetry(prompt: Prefix, generated: Prefix, maxRetries: nat, stepsLeft: nat) returns (result: Prefix, remainingSteps: nat)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires maxRetries >= 1
      requires stepsLeft >= maxRetries
      ensures this.lm.ValidTokensIdsLogits()
      ensures parser.IsValidPrefix(LongestValidSuffix(result))
      ensures |result| >= 0
      ensures remainingSteps >= 0
      ensures remainingSteps >= stepsLeft - maxRetries

    predicate HasBudget(stepsLeft: int, needed: int)
    {
      stepsLeft >= needed
    }

    function MinStepsToComplete(prefix: Prefix): (result: int)
      reads this
      reads this.parser
      requires parser.IsValidPrefix([])
      ensures result >= 0
      ensures parser.IsCompletePrefix(LongestValidSuffix(prefix)) ==> result == 0
      ensures !parser.IsCompletePrefix(LongestValidSuffix(prefix)) ==> result >= 1
    {
      this.parser.ParserDistanceToComplete(this.LongestValidSuffix(prefix))
    }

    method SoftConstrainToGrammar(prefix: Prefix, penalty: Logit)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires penalty > 0.0
      ensures this.lm.ValidTokensIdsLogits()
      ensures forall t :: t in parser.ValidNextTokens(LongestValidSuffix(prefix)) && t in lm.Tokens ==> lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
    {
      this.LongestValidSuffixIsValid(prefix);
      var suffix := this.LongestValidSuffix(prefix);
      if this.parser.IsCompletePrefix(suffix) {
        return;
      }
      var valid_tokens := this.parser.ValidNextTokens(suffix);
      var invalid_tokens: Prefix := [];
      var n := |this.lm.Tokens|;
      var i := 0;
      while i < n
        invariant 0 <= i <= n
        invariant lm.ValidTokensIdsLogits()
        invariant forall j :: 0 <= j < i && !(lm.Tokens[j] in valid_tokens) ==> lm.Tokens[j] in invalid_tokens
        invariant forall t :: t in invalid_tokens ==> t in lm.Tokens && !(t in valid_tokens)
      {
        if this.lm.Tokens[i] !in valid_tokens {
          invalid_tokens := (invalid_tokens + [this.lm.Tokens[i]]);
        }
        i := i + 1;
      }
      if |invalid_tokens| > 0 {
        this.lm.BiasTokens(invalid_tokens, -penalty);
      }
    }

    method IntersectWithGrammar(prefix: Prefix)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      ensures this.lm.ValidTokensIdsLogits()
      ensures !parser.IsCompletePrefix(LongestValidSuffix(prefix)) ==> forall t :: t in lm.Tokens && !(t in parser.ValidNextTokens(LongestValidSuffix(prefix))) ==> lm.IsMasked(t)
      ensures !parser.IsCompletePrefix(LongestValidSuffix(prefix)) ==> forall t :: t in parser.ValidNextTokens(LongestValidSuffix(prefix)) && t in lm.Tokens ==> lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
      ensures parser.IsCompletePrefix(LongestValidSuffix(prefix)) ==> forall t :: t in lm.Tokens ==> lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
    {
      this.LongestValidSuffixIsValid(prefix);
      var suffix := this.LongestValidSuffix(prefix);
      if this.parser.IsCompletePrefix(suffix) {
        return;
      }
      var valid_tokens := this.parser.ValidNextTokens(suffix);
      this.AllValidNextTokensInLM(suffix);
      this.lm.MaskTokensExcept(valid_tokens);
    }

    method BiasForCompletion(prefix: Prefix, bonus: Logit)
      modifies this.lm.Logits
      requires this.lm.ValidTokensIdsLogits()
      requires parser.IsValidPrefix([])
      requires bonus > 0.0
      ensures this.lm.ValidTokensIdsLogits()
      ensures forall t :: t in lm.Tokens && !(exists ct :: ct in parser.ValidNextTokens(LongestValidSuffix(prefix)) && parser.IsCompletePrefix(LongestValidSuffix(prefix) + [ct]) && ct == t) ==> lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
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
        invariant forall t :: t in lm.Tokens && !(t in valid_next[..i] && parser.IsCompletePrefix(suffix + [t])) ==> lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])
      {
        if this.parser.IsCompletePrefix((suffix + [valid_next[i]])) {
          this.lm.BiasToken(valid_next[i], bonus);
        }
        i := i + 1;
      }
    }

  }

  class CheckpointStack {
    var stack: seq<Prefix>

    constructor ()
      ensures Depth() == 0
      ensures IsEmpty()
    {
      this.stack := [];
    }

    method Push(prefix: Prefix)
      modifies this
      ensures Depth() == old(Depth()) + 1
      ensures Peek() == prefix
      ensures !IsEmpty()
      ensures Depth() >= 1
    {
      this.stack := (this.stack + [prefix]);
    }

    method Pop() returns (result: Prefix)
      modifies this
      requires Depth() > 0
      requires !IsEmpty()
      ensures Depth() == old(Depth()) - 1
      ensures Depth() >= 0
      ensures |result| >= 0
    {
      if |this.stack| == 0 {
        assert false;
      }
      result := this.stack[(|this.stack| - 1)];
      this.stack := this.stack[..|this.stack|-1];
      result := result;
      return;
    }

    function Peek(): (result: Prefix)
      reads this
      requires Depth() > 0
      ensures |result| >= 0
    {
      this.stack[(|this.stack| - 1)]
    }

    function Depth(): (result: int)
      reads this
      ensures result >= 0
      ensures result == 0 <==> IsEmpty()
    {
      |this.stack|
    }

    predicate IsEmpty()
      reads this
    {
      |this.stack| == 0
    }

  }

  class RepetitionTracker {
    const ngramSize: int

    constructor (ngramSize: int)
      requires ngramSize >= 1
      ensures this.ngramSize == ngramSize
    {
      this.ngramSize := ngramSize;
    }

    method {:extern} {:axiom} RecordToken(token: Token)
      modifies this

    function {:extern} {:axiom} GetCount(ngram: Prefix): (result: int)
      reads this
      requires |ngram| == this.ngramSize
      ensures result >= 0

    function {:extern} {:axiom} GetRepetitionPenalty(token: Token): (result: Logit)
      reads this
      ensures result >= 0.0

    method {:extern} {:axiom} ApplyRepetitionPenalties(lm: LM)
      modifies this
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()

  }

}