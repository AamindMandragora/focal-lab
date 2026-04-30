from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias


Token: TypeAlias = str
Prefix: TypeAlias = list[Token]
Id: TypeAlias = int
Logit: TypeAlias = float

MODULE_NAME = "VerifiedDecoderAgent"
LeftDelimiter: Token = "<<"
RightDelimiter: Token = ">>"
SpacedLeftDelimiter: Token = " <<"
SpacedRightDelimiter: Token = " >>"


@dataclass(frozen=True)
class DafnySpec:
    kind: str
    reads: tuple[str, ...] = ()
    modifies: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    ensures: tuple[str, ...] = ()
    decreases: tuple[str, ...] = ()
    axiom: bool = False
    extern: bool = False


def dafny_spec(
    *,
    kind: str,
    reads: tuple[str, ...] = (),
    modifies: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    ensures: tuple[str, ...] = (),
    decreases: tuple[str, ...] = (),
    axiom: bool = False,
    extern: bool = False,
):
    def decorate(obj: Any) -> Any:
        obj.__dafny_spec__ = DafnySpec(
            kind=kind,
            reads=reads,
            modifies=modifies,
            requires=requires,
            ensures=ensures,
            decreases=decreases,
            axiom=axiom,
            extern=extern,
        )
        return obj

    return decorate


# ── Top-Level Predicates ──────────────────────────────────────────────

@dafny_spec(kind="predicate")
def Contains(s: str, sub: str) -> bool:
    return sub in s


@dafny_spec(kind="predicate")
def PrefixContains(p: Prefix, t: Token) -> bool:
    return any(tok == t for tok in p)


@dafny_spec(kind="predicate")
def DelimitedAnswerValidForParser(parser: "Parser", prefix: Prefix) -> bool:
    delim = Delimiter(LeftDelimiter, RightDelimiter)
    content = delim.GetDelimitedContent(prefix)
    return (
        PrefixContains(prefix, LeftDelimiter)
        and PrefixContains(prefix, RightDelimiter)
        and (not delim.InsideDelimitedWindow(prefix))
        and parser.IsValidPrefix(content)
        and len(content) > 0
    )


# ══════════════════════════════════════════════════════════════════════
#  LM — Language Model Interface
# ══════════════════════════════════════════════════════════════════════

class LM:
    Tokens: Prefix
    Ids: list[Id]
    Logits: list[Logit]

    @dafny_spec(
        kind="constructor",
        ensures=("ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def __init__(self) -> None:
        self.Tokens = [LeftDelimiter, RightDelimiter]
        self.Ids = [0, 1]
        self.Logits = [0.0, 0.0]

    @dafny_spec(
        kind="predicate",
        reads=("this", "this.Logits"),
    )
    def ValidTokensIdsLogits(self) -> bool:
        return (
            len(self.Tokens) == len(self.Ids)
            and len(self.Ids) == len(self.Logits)
            and len(self.Ids) > 0
            and self.Ids[0] == 0
            and all(i == self.Ids[i] and i in self.Ids for i in range(len(self.Ids)))
            and all(
                self.Tokens[i] != self.Tokens[j]
                for i in range(len(self.Tokens))
                for j in range(len(self.Tokens))
                if i != j
            )
            and all(token in self.Tokens for token in self.Tokens)
            and all(any(self.Tokens[i] == token for i in range(len(self.Ids))) for token in self.Tokens)
            and all(-1e9 <= logit <= 1e9 for logit in self.Logits)
        )

    @dafny_spec(
        kind="lemma",
        ensures=("ValidTokensIdsLogits()",),
        axiom=True,
    )
    def ValidTokensIdsLogitsAlways(self) -> None:
        assert self.ValidTokensIdsLogits()

    # ── Token / Id / Logit Conversions ────────────────────────────────

    @dafny_spec(
        kind="function",
        reads=("this", "this.Logits"),
        requires=("ValidTokensIdsLogits()", "id in Ids"),
        ensures=(
            "token in Tokens",
            "Tokens[id] == token",
            "id == TokenToId(token)",
            "ValidTokensIdsLogits()",
        ),
    )
    def IdToToken(self, id: Id) -> Token:
        return self.Tokens[id]

    @dafny_spec(
        kind="function",
        reads=("this", "this.Logits"),
        requires=("ValidTokensIdsLogits()", "token in Tokens"),
        ensures=(
            "id in Ids",
            "Tokens[id] == token",
            "TokenToId(Tokens[id]) == id",
            "ValidTokensIdsLogits()",
        ),
    )
    def TokenToId(self, token: Token) -> Id:
        return self.TokenToIdRecursive(token, 0)

    @dafny_spec(
        kind="function",
        reads=("this", "this.Logits"),
        requires=(
            "ValidTokensIdsLogits()",
            "token in Tokens",
            "0 <= offset < |Tokens|",
            "(Tokens[offset] == token) || (token in Tokens[offset + 1..])",
        ),
        ensures=(
            "id in Ids",
            "0 <= TokenToIdRecursive(token, offset) < |Ids|",
            "Tokens[id] == token",
            "ValidTokensIdsLogits()",
        ),
        decreases=("|Tokens| - offset",),
    )
    def TokenToIdRecursive(self, token: Token, offset: int) -> Id:
        return offset if self.Tokens[offset] == token else self.TokenToIdRecursive(token, offset + 1)

    @dafny_spec(
        kind="function",
        reads=("this", "this.Logits"),
        requires=("ValidTokensIdsLogits()", "id in Ids"),
        ensures=("logit in Logits[0..Logits.Length]", "ValidTokensIdsLogits()"),
    )
    def IdToLogit(self, id: Id) -> Logit:
        return self.Logits[id]

    @dafny_spec(
        kind="function",
        reads=("this", "this.Logits"),
        requires=("ValidTokensIdsLogits()", "token in Tokens"),
        ensures=("ValidTokensIdsLogits()",),
    )
    def TokenToLogit(self, token: Token) -> Logit:
        return self.IdToLogit(self.TokenToId(token))

    @dafny_spec(
        kind="function",
        reads=("this", "this.Logits"),
        requires=(
            "ValidTokensIdsLogits()",
            "|tokens| > 0",
            "forall token: Token :: token in tokens ==> token in Tokens",
        ),
        ensures=("ValidTokensIdsLogits()",),
    )
    def TokensToLogits(self, tokens: Prefix) -> list[Logit]:
        return (
            [self.TokenToLogit(tokens[0])]
            if len(tokens) == 1
            else [self.TokenToLogit(tokens[0])] + self.TokensToLogits(tokens[1:])
        )

    @dafny_spec(
        kind="function",
        reads=("this", "this.Logits"),
        requires=(
            "ValidTokensIdsLogits()",
            "|ids| > 0",
            "forall id: Id :: id in ids ==> id in Ids",
        ),
        ensures=("ValidTokensIdsLogits()",),
    )
    def IdsToLogits(self, ids: list[Id]) -> list[Logit]:
        return [self.IdToLogit(ids[0])] if len(ids) == 1 else [self.IdToLogit(ids[0])] + self.IdsToLogits(ids[1:])

    # ── Hard Masking ──────────────────────────────────────────────────

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=("ValidTokensIdsLogits()", "token in Tokens"),
        ensures=(
            "ValidTokensIdsLogits()",
            "IsMasked(token)",
            "forall t: Token :: t in Tokens && t != token ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def MaskToken(self, token: Token) -> None:
        token_id = self.TokenToId(token)
        self.Logits[token_id] = -1e9

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=(
            "ValidTokensIdsLogits()",
            "|tokens| > 0",
            "forall token :: token in tokens ==> token in Tokens",
        ),
        ensures=(
            "ValidTokensIdsLogits()",
            "forall t :: t in tokens ==> IsMasked(t)",
            "forall t :: t in Tokens && !(t in tokens) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def MaskTokens(self, tokens: Prefix) -> None:
        n = len(tokens)
        i = 0
        # invariant 0 <= i <= N
        # invariant ValidTokensIdsLogits()
        # invariant forall j :: 0 <= j < i ==> IsMasked(tokens[j])
        # invariant forall t :: t in Tokens && !(t in tokens[..i]) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
        while i < n:
            self.MaskToken(tokens[i])
            i += 1

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=(
            "ValidTokensIdsLogits()",
            "|tokens| > 0",
            "forall token :: token in tokens ==> token in Tokens",
        ),
        ensures=(
            "ValidTokensIdsLogits()",
            "forall t :: t in Tokens && !(t in tokens) ==> IsMasked(t)",
            "forall t :: t in tokens ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def MaskTokensExcept(self, tokens: Prefix) -> None:
        to_mask: Prefix = []
        n = len(self.Tokens)
        i = 0
        # invariant 0 <= i <= N
        # invariant ValidTokensIdsLogits()
        # invariant forall j :: 0 <= j < i && !(Tokens[j] in tokens) ==> Tokens[j] in toMask
        # invariant forall j :: 0 <= j < i && Tokens[j] in tokens ==> !(Tokens[j] in toMask)
        # invariant forall t: Token :: t in toMask ==> t !in tokens && t in Tokens
        while i < n:
            if self.Tokens[i] not in tokens:
                to_mask = to_mask + [self.Tokens[i]]
            i += 1
        if len(to_mask) > 0:
            self.MaskTokens(to_mask)

    @dafny_spec(
        kind="predicate",
        reads=("this", "this.Logits"),
        requires=("ValidTokensIdsLogits()", "token in Tokens"),
        ensures=("ValidTokensIdsLogits()",),
    )
    def IsMasked(self, token: Token) -> bool:
        return self.Logits[self.TokenToId(token)] == -1e9

    @dafny_spec(
        kind="predicate",
        reads=("this", "this.Logits"),
        requires=("ValidTokensIdsLogits()",),
        ensures=("ValidTokensIdsLogits()",),
    )
    def HasUnmaskedToken(self) -> bool:
        return any(token in self.Tokens and not self.IsMasked(token) for token in self.Tokens)

    # ── Soft Logit Shaping ────────────────────────────────────────────

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=("ValidTokensIdsLogits()", "token in Tokens"),
        ensures=(
            "ValidTokensIdsLogits()",
            "-1e9 <= Logits[TokenToId(token)] <= 1e9",
            "Logits[TokenToId(token)] == if old(Logits[TokenToId(token)]) + delta > 1e9 then 1e9 else if old(Logits[TokenToId(token)]) + delta < -1e9 then -1e9 else old(Logits[TokenToId(token)]) + delta",
            "forall t: Token :: t in Tokens && t != token ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def BiasToken(self, token: Token, delta: Logit) -> None:
        token_id = self.TokenToId(token)
        raw = self.Logits[token_id] + delta
        if raw > 1e9:
            raw = 1e9
        if raw < -1e9:
            raw = -1e9
        self.Logits[token_id] = raw

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=(
            "ValidTokensIdsLogits()",
            "|tokens| > 0",
            "forall token :: token in tokens ==> token in Tokens",
        ),
        ensures=(
            "ValidTokensIdsLogits()",
            "forall t :: t in Tokens && !(t in tokens) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def BiasTokens(self, tokens: Prefix, delta: Logit) -> None:
        n = len(tokens)
        i = 0
        # invariant 0 <= i <= n
        # invariant ValidTokensIdsLogits()
        # invariant forall t :: t in Tokens && !(t in tokens[..i]) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
        while i < n:
            self.BiasToken(tokens[i], delta)
            i += 1

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=("ValidTokensIdsLogits()", "token in Tokens", "factor != 0.0"),
        ensures=(
            "ValidTokensIdsLogits()",
            "-1e9 <= Logits[TokenToId(token)] <= 1e9",
            "forall t: Token :: t in Tokens && t != token ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def ScaleToken(self, token: Token, factor: Logit) -> None:
        token_id = self.TokenToId(token)
        raw = self.Logits[token_id] * factor
        if raw > 1e9:
            raw = 1e9
        if raw < -1e9:
            raw = -1e9
        self.Logits[token_id] = raw

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=(
            "ValidTokensIdsLogits()",
            "|tokens| > 0",
            "forall token :: token in tokens ==> token in Tokens",
            "factor != 0.0",
        ),
        ensures=(
            "ValidTokensIdsLogits()",
            "forall t :: t in Tokens && !(t in tokens) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def ScaleTokens(self, tokens: Prefix, factor: Logit) -> None:
        n = len(tokens)
        i = 0
        # invariant 0 <= i <= n
        # invariant ValidTokensIdsLogits()
        # invariant forall t :: t in Tokens && !(t in tokens[..i]) ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])
        while i < n:
            self.ScaleToken(tokens[i], factor)
            i += 1

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=("ValidTokensIdsLogits()", "-1e9 <= low", "low <= high", "high <= 1e9"),
        ensures=(
            "ValidTokensIdsLogits()",
            "forall id :: 0 <= id < Logits.Length ==> low <= Logits[id] <= high",
            "forall id :: 0 <= id < Logits.Length ==> (old(Logits[id]) >= low && old(Logits[id]) <= high ==> Logits[id] == old(Logits[id]))",
        ),
        axiom=True,
        extern=True,
    )
    def ClampLogits(self, low: Logit, high: Logit) -> None:
        n = len(self.Logits)
        i = 0
        # invariant 0 <= i <= n
        # invariant ValidTokensIdsLogits()
        # invariant forall j :: 0 <= j < i ==> low <= Logits[j] <= high
        # invariant forall j :: i <= j < n ==> Logits[j] == old(Logits[j])
        while i < n:
            if self.Logits[i] > high:
                self.Logits[i] = high
            if self.Logits[i] < low:
                self.Logits[i] = low
            i += 1

    # ── Filtering ─────────────────────────────────────────────────────

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=("ValidTokensIdsLogits()", "1 <= k <= |Tokens|"),
        ensures=(
            "ValidTokensIdsLogits()",
            "HasUnmaskedToken()",
            "forall t :: t in Tokens && !IsMasked(t) ==> !old(IsMasked(t))",
        ),
        axiom=True,
        extern=True,
    )
    def TopKFilter(self, k: int) -> None:
        n = len(self.Tokens)
        sorted_ids = sorted(range(n), key=lambda idx: self.Logits[idx], reverse=True)
        keep = set(sorted_ids[:k])
        i = 0
        while i < n:
            if i not in keep:
                self.Logits[i] = -1e9
            i += 1

    # ── Generation ────────────────────────────────────────────────────

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=("ValidTokensIdsLogits()",),
        ensures=("ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def GenerateLogits(self, input: Prefix) -> None:
        if not self.ValidTokensIdsLogits():
            raise ValueError("LM invariant violated before GenerateLogits")

    @dafny_spec(
        kind="method",
        requires=("ValidTokensIdsLogits()",),
        ensures=("token in Tokens", "!IsMasked(token)", "ValidTokensIdsLogits()"),
        axiom=True,
        extern=True,
    )
    def ChooseNextToken(self) -> Token:
        best_token: Token | None = None
        best_logit: Logit | None = None
        for token, logit in zip(self.Tokens, self.Logits):
            if logit == -1e9:
                continue
            if best_logit is None or logit > best_logit:
                best_token = token
                best_logit = logit
        if best_token is None:
            raise ValueError("No unmasked token is available")
        return best_token


# ══════════════════════════════════════════════════════════════════════
#  Parser — Grammar Oracle Interface
# ══════════════════════════════════════════════════════════════════════

class Parser:
    @dafny_spec(
        kind="predicate",
        ensures=("forall k :: 0 <= k < |prefix| ==> IsValidPrefix(prefix[..k])",),
        axiom=True,
        extern=True,
    )
    def IsValidPrefix(self, prefix: Prefix) -> bool:
        raise NotImplementedError

    @dafny_spec(
        kind="lemma",
        ensures=("IsValidPrefix([])",),
        axiom=True,
    )
    def EmptyPrefixIsValid(self) -> None:
        assert self.IsValidPrefix([])

    @dafny_spec(
        kind="predicate",
        ensures=("IsValidPrefix(prefix)",),
        axiom=True,
        extern=True,
    )
    def IsCompletePrefix(self, prefix: Prefix) -> bool:
        raise NotImplementedError

    @dafny_spec(kind="predicate")
    def IsDeadPrefix(self, prefix: Prefix) -> bool:
        return (not self.IsCompletePrefix(prefix)) and len(self.ValidNextTokens(prefix)) == 0

    @dafny_spec(
        kind="predicate",
        requires=("IsValidPrefix(prefix)",),
    )
    def ValidNextToken(self, prefix: Prefix, token: Token) -> bool:
        return token in self.ValidNextTokens(prefix)

    @dafny_spec(
        kind="function",
        requires=("IsValidPrefix(prefix)",),
        ensures=(
            "forall t :: t in ValidNextTokens(prefix) ==> IsValidPrefix(prefix + [t])",
            "(IsCompletePrefix(prefix) || |ValidNextTokens(prefix)| > 0)",
        ),
        axiom=True,
        extern=True,
    )
    def ValidNextTokens(self, prefix: Prefix) -> Prefix:
        raise NotImplementedError

    @dafny_spec(
        kind="function",
        requires=("IsValidPrefix(prefix)",),
        ensures=(
            "result >= 0",
            "result == |ValidNextTokens(prefix)|",
            "result == 0 ==> (IsCompletePrefix(prefix) || IsDeadPrefix(prefix))",
        ),
    )
    def ValidContinuationCount(self, prefix: Prefix) -> int:
        return len(self.ValidNextTokens(prefix))

    @dafny_spec(
        kind="function",
        requires=("IsValidPrefix(prefix)",),
        ensures=(
            "result >= 0",
            "IsCompletePrefix(prefix) ==> result == 0",
            "!IsCompletePrefix(prefix) ==> result >= 1",
        ),
        axiom=True,
        extern=True,
    )
    def ParserDistanceToComplete(self, prefix: Prefix) -> int:
        return 0 if self.IsCompletePrefix(prefix) else 1


# ══════════════════════════════════════════════════════════════════════
#  Delimiter — Answer Extraction Support (evaluator backward compat)
# ══════════════════════════════════════════════════════════════════════

class Delimiter:
    Left: Token
    Right: Token

    @dafny_spec(
        kind="constructor",
        requires=("left != right",),
        ensures=("this.Left == left && this.Right == right", "this.Left != this.Right"),
    )
    def __init__(self, left: Token, right: Token) -> None:
        if left == right:
            raise ValueError("Delimiter endpoints must be distinct")
        self.Left = left
        self.Right = right

    @dafny_spec(
        kind="function",
        ensures=(
            "result <= |prefix|",
            "result < |prefix| ==> prefix[result] == this.Left",
            "result == |prefix| ==> forall i :: 0 <= i < |prefix| ==> prefix[i] != this.Left",
            "result < |prefix| ==> forall i :: result < i < |prefix| ==> prefix[i] != this.Left",
        ),
        decreases=("|prefix|",),
    )
    def LastLeftDelimiterIndex(self, prefix: Prefix) -> int:
        return (
            0
            if len(prefix) == 0
            else (len(prefix) - 1 if prefix[-1] == self.Left else self.LastLeftDelimiterIndex(prefix[:-1]))
        )

    @dafny_spec(
        kind="function",
        ensures=(
            "result <= |content|",
            "result < |content| ==> content[result] == this.Right",
            "forall i :: 0 <= i < result ==> content[i] != this.Right",
        ),
        decreases=("|content|",),
    )
    def FirstRightDelimiterIndex(self, content: Prefix) -> int:
        return 0 if len(content) == 0 or content[0] == self.Right else 1 + self.FirstRightDelimiterIndex(content[1:])

    @dafny_spec(
        kind="lemma",
        requires=("FirstRightDelimiterIndex(content) == |content|",),
        ensures=("!PrefixContains(content, this.Right)",),
    )
    def NoFirstRightDelimiterIndexMeansNoRight(self, content: Prefix) -> None:
        assert self.FirstRightDelimiterIndex(content) == len(content)
        assert not PrefixContains(content, self.Right)

    @dafny_spec(
        kind="function",
        ensures=(
            "|GetDelimitedContent(prefix)| <= |prefix|",
            "forall t: Token :: t in GetDelimitedContent(prefix) ==> t in prefix",
        ),
    )
    def GetDelimitedContent(self, prefix: Prefix) -> Prefix:
        start = self.LastLeftDelimiterIndex(prefix) + 1
        after_left = [] if start > len(prefix) else prefix[start:]
        end_idx = self.FirstRightDelimiterIndex(after_left)
        return after_left[:end_idx]

    @dafny_spec(kind="predicate")
    def InsideDelimitedWindow(self, prefix: Prefix) -> bool:
        start = self.LastLeftDelimiterIndex(prefix) + 1
        return start <= len(prefix) and self.FirstRightDelimiterIndex(prefix[start:]) == len(prefix[start:])

    @dafny_spec(
        kind="lemma",
        requires=("InsideDelimitedWindow(prefix)",),
        ensures=("!PrefixContains(GetDelimitedContent(prefix), this.Right)",),
    )
    def InsideDelimitedWindowNoRight(self, prefix: Prefix) -> None:
        start = self.LastLeftDelimiterIndex(prefix) + 1
        after_left = prefix[start:]
        self.NoFirstRightDelimiterIndexMeansNoRight(after_left)

    @dafny_spec(
        kind="lemma",
        requires=("InsideDelimitedWindow(prefix)", "next != Right", "next != Left"),
        ensures=(
            "GetDelimitedContent(prefix + [next]) == GetDelimitedContent(prefix) + [next]",
            "next != Right ==> InsideDelimitedWindow(prefix + [next])",
        ),
        axiom=True,
    )
    def GetDelimitedContentAppend(self, prefix: Prefix, next: Token) -> None:
        assert self.InsideDelimitedWindow(prefix)
        assert next != self.Right
        assert next != self.Left
        assert self.GetDelimitedContent(prefix + [next]) == self.GetDelimitedContent(prefix) + [next]
        assert self.InsideDelimitedWindow(prefix + [next])

    @dafny_spec(
        kind="lemma",
        ensures=(
            "InsideDelimitedWindow(prefix + [this.Left])",
            "GetDelimitedContent(prefix + [this.Left]) == []",
        ),
    )
    def AppendLeftEntersWindow(self, prefix: Prefix) -> None:
        assert self.InsideDelimitedWindow(prefix + [self.Left])
        assert self.GetDelimitedContent(prefix + [self.Left]) == []

    @dafny_spec(
        kind="lemma",
        requires=("FirstRightDelimiterIndex(content) == |content|",),
        ensures=("FirstRightDelimiterIndex(content + [this.Right]) == |content|",),
    )
    def FirstRightDelimiterAppendRight(self, content: Prefix) -> None:
        assert self.FirstRightDelimiterIndex(content) == len(content)
        assert self.FirstRightDelimiterIndex(content + [self.Right]) == len(content)

    @dafny_spec(
        kind="lemma",
        requires=("tok != this.Left",),
        ensures=(
            "var oldIdx := LastLeftDelimiterIndex(prefix); var newIdx := LastLeftDelimiterIndex(prefix + [tok]); if oldIdx < |prefix| then newIdx == oldIdx else newIdx == |prefix + [tok]|",
        ),
    )
    def LastLeftDelimiterAppendNonLeft(self, prefix: Prefix, tok: Token) -> None:
        assert tok != self.Left
        old_idx = self.LastLeftDelimiterIndex(prefix)
        new_idx = self.LastLeftDelimiterIndex(prefix + [tok])
        if old_idx < len(prefix):
            assert new_idx == old_idx
        else:
            assert new_idx == len(prefix + [tok])

    @dafny_spec(
        kind="lemma",
        requires=("InsideDelimitedWindow(prefix)", "this.Left != this.Right"),
        ensures=("!InsideDelimitedWindow(prefix + [this.Right])",),
    )
    def AppendRightExitsWindow(self, prefix: Prefix) -> None:
        assert self.InsideDelimitedWindow(prefix)
        assert self.Left != self.Right
        assert not self.InsideDelimitedWindow(prefix + [self.Right])


# ══════════════════════════════════════════════════════════════════════
#  CSDHelpers — Composable Strategy Building Blocks
# ══════════════════════════════════════════════════════════════════════

class CSDHelpers:
    lm: LM
    parser: Parser

    @dafny_spec(
        kind="constructor",
        requires=("lm.ValidTokensIdsLogits()",),
        ensures=(
            "this.lm == lm && this.parser == parser",
            "lm.ValidTokensIdsLogits()",
            "this.lm.ValidTokensIdsLogits()",
        ),
    )
    def __init__(self, lm: LM, parser: Parser) -> None:
        self.lm = lm
        self.parser = parser

    # ── Core Lemmas ───────────────────────────────────────────────────

    @dafny_spec(
        kind="lemma",
        requires=("lm.ValidTokensIdsLogits()", "parser.IsValidPrefix(content)"),
        ensures=(
            "lm.ValidTokensIdsLogits()",
            "forall t: Token :: t in parser.ValidNextTokens(content) ==> t in lm.Tokens",
        ),
        axiom=True,
    )
    def AllValidNextTokensInLM(self, content: Prefix) -> None:
        assert self.lm.ValidTokensIdsLogits()
        assert self.parser.IsValidPrefix(content)
        for token in self.parser.ValidNextTokens(content):
            assert token in self.lm.Tokens

    @dafny_spec(
        kind="lemma",
        requires=(
            "lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix(content)",
            "!parser.IsCompletePrefix(content)",
            "forall t: Token :: t in parser.ValidNextTokens(content) ==> t in lm.Tokens",
            "parser.IsValidPrefix(content + [next])",
        ),
        ensures=("forall t: Token :: t in parser.ValidNextTokens(content + [next]) ==> t in lm.Tokens",),
        axiom=True,
    )
    def ValidNextTokensInLMAfterStep(self, content: Prefix, next: Token) -> None:
        assert self.lm.ValidTokensIdsLogits()
        assert self.parser.IsValidPrefix(content + [next])

    # ── Suffix-Based Grammar Alignment ────────────────────────────────

    @dafny_spec(
        kind="function",
        reads=("this", "this.parser"),
        requires=("parser.IsValidPrefix([])",),
        ensures=(
            "parser.IsValidPrefix(result)",
            "|result| <= |prefix|",
            "|prefix| > 0 && parser.IsValidPrefix(prefix) ==> result == prefix",
            "|prefix| == 0 ==> result == []",
            "forall i :: 0 <= i < |result| ==> result[i] == prefix[|prefix| - |result| + i]",
        ),
        decreases=("|prefix|",),
    )
    def LongestValidSuffix(self, prefix: Prefix) -> Prefix:
        return (
            []
            if len(prefix) == 0
            else (prefix if self.parser.IsValidPrefix(prefix) else self.LongestValidSuffix(prefix[1:]))
        )

    @dafny_spec(
        kind="lemma",
        requires=(
            "parser.IsValidPrefix([])",
            "parser.IsValidPrefix(LongestValidSuffix(prefix))",
            "parser.ValidNextToken(LongestValidSuffix(prefix), next)",
        ),
        ensures=(
            "parser.IsValidPrefix(LongestValidSuffix(prefix) + [next])",
            "|LongestValidSuffix(prefix + [next])| >= |LongestValidSuffix(prefix)| + 1",
        ),
        axiom=True,
    )
    def LongestValidSuffixAppend(self, prefix: Prefix, next: Token) -> None:
        assert self.parser.IsValidPrefix(self.LongestValidSuffix(prefix + [next]))

    @dafny_spec(
        kind="lemma",
        requires=("parser.IsValidPrefix([])",),
        ensures=("parser.IsValidPrefix(LongestValidSuffix(prefix))",),
    )
    def LongestValidSuffixIsValid(self, prefix: Prefix) -> None:
        assert self.parser.IsValidPrefix(self.LongestValidSuffix(prefix))

    @dafny_spec(
        kind="lemma",
        requires=(
            "parser.IsValidPrefix([])",
            "parser.IsValidPrefix(LongestValidSuffix(prefix))",
        ),
        ensures=(
            "parser.IsCompletePrefix(LongestValidSuffix(prefix)) || |parser.ValidNextTokens(LongestValidSuffix(prefix))| > 0",
        ),
    )
    def LongestValidSuffixNotDead(self, prefix: Prefix) -> None:
        suffix = self.LongestValidSuffix(prefix)
        assert self.parser.IsCompletePrefix(suffix) or len(self.parser.ValidNextTokens(suffix)) > 0

    # ── Grammar State Queries ─────────────────────────────────────────
    # These route all parser queries through LongestValidSuffix so
    # strategies never need to call parser.* directly.

    @dafny_spec(
        kind="predicate",
        reads=("this", "this.parser"),
        requires=("parser.IsValidPrefix([])",),
    )
    def CanConstrain(self, prefix: Prefix) -> bool:
        return not self.parser.IsCompletePrefix(self.LongestValidSuffix(prefix))

    @dafny_spec(
        kind="predicate",
        reads=("this", "this.parser"),
        requires=("parser.IsValidPrefix([])",),
    )
    def IsComplete(self, prefix: Prefix) -> bool:
        return self.parser.IsCompletePrefix(self.LongestValidSuffix(prefix))

    @dafny_spec(
        kind="predicate",
        reads=("this", "this.parser"),
        requires=("parser.IsValidPrefix([])",),
    )
    def IsDead(self, prefix: Prefix) -> bool:
        return self.parser.IsDeadPrefix(self.LongestValidSuffix(prefix))

    @dafny_spec(
        kind="function",
        reads=("this", "this.parser"),
        requires=("parser.IsValidPrefix([])",),
        ensures=("result >= 0",),
    )
    def ValidContinuationCount(self, prefix: Prefix) -> int:
        return self.parser.ValidContinuationCount(self.LongestValidSuffix(prefix))

    @dafny_spec(
        kind="function",
        reads=("this", "this.parser"),
        requires=("parser.IsValidPrefix([])",),
        ensures=(
            "result >= 0",
            "IsComplete(prefix) ==> result == 0",
            "!IsComplete(prefix) ==> result >= 1",
        ),
    )
    def ParserDistanceToComplete(self, prefix: Prefix) -> int:
        return self.parser.ParserDistanceToComplete(self.LongestValidSuffix(prefix))

    # ── Delimiter Predicates ──────────────────────────────────────────
    # Thin wrappers that handle both spaced and unspaced delimiter
    # variants.  Strategies use these to detect structural boundaries.

    @dafny_spec(kind="predicate")
    def IsLeftDelimiterToken(self, token: Token) -> bool:
        return token == LeftDelimiter or token == SpacedLeftDelimiter

    @dafny_spec(kind="predicate")
    def IsRightDelimiterToken(self, token: Token) -> bool:
        return token == RightDelimiter or token == SpacedRightDelimiter

    @dafny_spec(kind="predicate")
    def EndsWithLeftDelimiter(self, prefix: Prefix) -> bool:
        return len(prefix) > 0 and self.IsLeftDelimiterToken(prefix[len(prefix) - 1])

    @dafny_spec(kind="predicate")
    def EndsWithRightDelimiter(self, prefix: Prefix) -> bool:
        return len(prefix) > 0 and self.IsRightDelimiterToken(prefix[len(prefix) - 1])

    @dafny_spec(kind="predicate")
    def ContainsLeftDelimiter(self, prefix: Prefix) -> bool:
        return PrefixContains(prefix, LeftDelimiter) or PrefixContains(prefix, SpacedLeftDelimiter)

    @dafny_spec(kind="predicate")
    def ContainsRightDelimiter(self, prefix: Prefix) -> bool:
        return PrefixContains(prefix, RightDelimiter) or PrefixContains(prefix, SpacedRightDelimiter)

    # ── Primitive Step Functions ──────────────────────────────────────
    # Each does exactly ONE thing: generate logits, apply one shaping
    # policy, choose.  The strategy composes these; the library does
    # not bundle delimiter logic into step functions.

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "nextToken in lm.Tokens",
            "!lm.IsMasked(nextToken)",
        ),
    )
    def UnconstrainedStep(self, prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]:
        self.lm.ValidTokensIdsLogitsAlways()
        self.lm.GenerateLogits(prompt + generated)
        next_token = self.lm.ChooseNextToken()
        return next_token, stepsLeft - 1

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "!parser.IsCompletePrefix(LongestValidSuffix(generated))",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "nextToken in lm.Tokens",
            "!lm.IsMasked(nextToken)",
            "parser.ValidNextToken(LongestValidSuffix(generated), nextToken)",
            "parser.IsValidPrefix(LongestValidSuffix(generated) + [nextToken])",
            "|LongestValidSuffix(generated + [nextToken])| >= |LongestValidSuffix(generated)| + 1",
        ),
    )
    def ConstrainedStep(self, prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]:
        self.LongestValidSuffixIsValid(generated)
        suffix = self.LongestValidSuffix(generated)
        self.AllValidNextTokensInLM(suffix)
        self.lm.GenerateLogits(prompt + generated)
        self.lm.MaskTokensExcept(self.parser.ValidNextTokens(suffix))
        next_token = self.lm.ChooseNextToken()
        self.LongestValidSuffixAppend(generated, next_token)
        return next_token, stepsLeft - 1

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "stepsLeft >= 1",
            "penalty > 0.0",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "nextToken in lm.Tokens",
            "!lm.IsMasked(nextToken)",
        ),
        axiom=True,
        extern=True,
    )
    def SoftConstrainedStep(self, prompt: Prefix, generated: Prefix, penalty: Logit, stepsLeft: int) -> tuple[Token, int]:
        self.LongestValidSuffixIsValid(generated)
        suffix = self.LongestValidSuffix(generated)
        valid_tokens = self.parser.ValidNextTokens(suffix)
        invalid_tokens: Prefix = []
        n = len(self.lm.Tokens)
        i = 0
        # invariant 0 <= i <= n
        # invariant lm.ValidTokensIdsLogits()
        # invariant forall j :: 0 <= j < i && !(lm.Tokens[j] in valid_tokens) ==> lm.Tokens[j] in invalid_tokens
        # invariant forall t :: t in invalid_tokens ==> t in lm.Tokens && !(t in valid_tokens)
        while i < n:
            if self.lm.Tokens[i] not in valid_tokens:
                invalid_tokens = invalid_tokens + [self.lm.Tokens[i]]
            i += 1
        self.lm.GenerateLogits(prompt + generated)
        if len(invalid_tokens) > 0:
            self.lm.BiasTokens(invalid_tokens, -penalty)
        next_token = self.lm.ChooseNextToken()
        remaining_steps = stepsLeft - 1
        return next_token, remaining_steps

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "stepsLeft >= 1",
            "1 <= k <= |lm.Tokens|",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "nextToken in lm.Tokens",
            "!lm.IsMasked(nextToken)",
            "parser.ValidNextToken(LongestValidSuffix(generated), nextToken)",
            "parser.IsValidPrefix(LongestValidSuffix(generated) + [nextToken])",
        ),
        axiom=True,
        extern=True,
    )
    def TopKConstrainedStep(self, prompt: Prefix, generated: Prefix, k: int, stepsLeft: int) -> tuple[Token, int]:
        self.LongestValidSuffixIsValid(generated)
        self.lm.GenerateLogits(prompt + generated)
        self.lm.TopKFilter(k)
        suffix = self.LongestValidSuffix(generated)
        self.AllValidNextTokensInLM(suffix)
        self.lm.MaskTokensExcept(self.parser.ValidNextTokens(suffix))
        next_token = self.lm.ChooseNextToken()
        self.LongestValidSuffixAppend(generated, next_token)
        remaining_steps = stepsLeft - 1
        return next_token, remaining_steps

    @dafny_spec(
        kind="method",
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "token in lm.Tokens",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "nextToken == token",
            "nextToken in lm.Tokens",
        ),
    )
    def ForcedTokenStep(self, prompt: Prefix, generated: Prefix, token: Token, stepsLeft: int) -> tuple[Token, int]:
        self.lm.ValidTokensIdsLogitsAlways()
        return token, stepsLeft - 1

    # ── Logit Shaping Composites ──────────────────────────────────────
    # These modify logits in place BEFORE a ChooseNextToken call.
    # Strategies call GenerateLogits, then zero or more of these,
    # then ChooseNextToken.  They compose freely.

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "penalty > 0.0",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "forall t :: t in parser.ValidNextTokens(LongestValidSuffix(prefix)) && t in lm.Tokens ==> lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])",
        ),
        axiom=True,
        extern=True,
    )
    def SoftConstrainToGrammar(self, prefix: Prefix, penalty: Logit) -> None:
        self.LongestValidSuffixIsValid(prefix)
        suffix = self.LongestValidSuffix(prefix)
        valid_tokens = self.parser.ValidNextTokens(suffix)
        invalid_tokens: Prefix = []
        n = len(self.lm.Tokens)
        i = 0
        # invariant 0 <= i <= n
        # invariant lm.ValidTokensIdsLogits()
        while i < n:
            if self.lm.Tokens[i] not in valid_tokens:
                invalid_tokens = invalid_tokens + [self.lm.Tokens[i]]
            i += 1
        if len(invalid_tokens) > 0:
            self.lm.BiasTokens(invalid_tokens, -penalty)

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "forall t :: t in lm.Tokens && !(t in parser.ValidNextTokens(LongestValidSuffix(prefix))) ==> lm.IsMasked(t)",
            "forall t :: t in parser.ValidNextTokens(LongestValidSuffix(prefix)) && t in lm.Tokens ==> lm.Logits[lm.TokenToId(t)] == old(lm.Logits[lm.TokenToId(t)])",
        ),
        axiom=True,
        extern=True,
    )
    def IntersectWithGrammar(self, prefix: Prefix) -> None:
        self.LongestValidSuffixIsValid(prefix)
        suffix = self.LongestValidSuffix(prefix)
        valid_tokens = self.parser.ValidNextTokens(suffix)
        self.AllValidNextTokensInLM(suffix)
        self.lm.MaskTokensExcept(valid_tokens)

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "bonus > 0.0",
        ),
        ensures=("this.lm.ValidTokensIdsLogits()",),
    )
    def BiasForCompletion(self, prefix: Prefix, bonus: Logit) -> None:
        self.LongestValidSuffixIsValid(prefix)
        suffix = self.LongestValidSuffix(prefix)
        if self.parser.IsCompletePrefix(suffix):
            return
        valid_next = self.parser.ValidNextTokens(suffix)
        self.AllValidNextTokensInLM(suffix)
        n = len(valid_next)
        i = 0
        # invariant 0 <= i <= n
        # invariant lm.ValidTokensIdsLogits()
        while i < n:
            if self.parser.IsCompletePrefix(suffix + [valid_next[i]]):
                self.lm.BiasToken(valid_next[i], bonus)
            i += 1

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=("this.lm.ValidTokensIdsLogits()",),
        ensures=("this.lm.ValidTokensIdsLogits()",),
    )
    def MaskAllDelimiters(self, generated: Prefix) -> None:
        if LeftDelimiter in self.lm.Tokens:
            self.lm.MaskToken(LeftDelimiter)
        if RightDelimiter in self.lm.Tokens:
            self.lm.MaskToken(RightDelimiter)
        if SpacedLeftDelimiter in self.lm.Tokens:
            self.lm.MaskToken(SpacedLeftDelimiter)
        if SpacedRightDelimiter in self.lm.Tokens:
            self.lm.MaskToken(SpacedRightDelimiter)

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=("this.lm.ValidTokensIdsLogits()",),
        ensures=("this.lm.ValidTokensIdsLogits()",),
    )
    def MaskRightDelimiters(self, generated: Prefix) -> None:
        if RightDelimiter in self.lm.Tokens:
            self.lm.MaskToken(RightDelimiter)
        if SpacedRightDelimiter in self.lm.Tokens:
            self.lm.MaskToken(SpacedRightDelimiter)

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=("this.lm.ValidTokensIdsLogits()", "bias > 0.0"),
        ensures=("this.lm.ValidTokensIdsLogits()",),
    )
    def BiasLeftDelimiters(self, bias: Logit) -> None:
        if LeftDelimiter in self.lm.Tokens:
            self.lm.BiasToken(LeftDelimiter, bias)
        if SpacedLeftDelimiter in self.lm.Tokens:
            self.lm.BiasToken(SpacedLeftDelimiter, bias)

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=("this.lm.ValidTokensIdsLogits()", "bias > 0.0"),
        ensures=("this.lm.ValidTokensIdsLogits()",),
    )
    def BiasRightDelimiters(self, bias: Logit) -> None:
        if RightDelimiter in self.lm.Tokens:
            self.lm.BiasToken(RightDelimiter, bias)
        if SpacedRightDelimiter in self.lm.Tokens:
            self.lm.BiasToken(SpacedRightDelimiter, bias)

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=("this.lm.ValidTokensIdsLogits()",),
        ensures=("this.lm.ValidTokensIdsLogits()",),
    )
    def MaskLeftDelimiters(self, generated: Prefix) -> None:
        if LeftDelimiter in self.lm.Tokens:
            self.lm.MaskToken(LeftDelimiter)
        if SpacedLeftDelimiter in self.lm.Tokens:
            self.lm.MaskToken(SpacedLeftDelimiter)

    # ── Append Wrappers ───────────────────────────────────────────────
    # Convenience methods that call a step function and append the
    # result.  Strategies can also call step functions directly and
    # manage the prefix themselves.

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "|updated| == |prefix| + 1",
            "|updated| + remainingSteps == |prefix| + stepsLeft",
            "updated[|prefix|] in lm.Tokens",
        ),
    )
    def AppendUnconstrainedStep(self, prompt: Prefix, prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]:
        next_token, remaining = self.UnconstrainedStep(prompt, prefix, stepsLeft)
        return prefix + [next_token], remaining

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "!parser.IsCompletePrefix(LongestValidSuffix(prefix))",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "|updated| == |prefix| + 1",
            "|updated| + remainingSteps == |prefix| + stepsLeft",
            "updated[|prefix|] in lm.Tokens",
            "parser.ValidNextToken(LongestValidSuffix(prefix), updated[|prefix|])",
            "parser.IsValidPrefix(LongestValidSuffix(prefix) + [updated[|prefix|]])",
        ),
    )
    def AppendConstrainedStep(self, prompt: Prefix, prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]:
        next_token, remaining = self.ConstrainedStep(prompt, prefix, stepsLeft)
        return prefix + [next_token], remaining

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "stepsLeft >= 1",
            "penalty > 0.0",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "|updated| == |prefix| + 1",
            "|updated| + remainingSteps == |prefix| + stepsLeft",
            "updated[|prefix|] in lm.Tokens",
        ),
        axiom=True,
        extern=True,
    )
    def AppendSoftConstrainedStep(self, prompt: Prefix, prefix: Prefix, penalty: Logit, stepsLeft: int) -> tuple[Prefix, int]:
        next_token, remaining_steps = self.SoftConstrainedStep(prompt, prefix, penalty, stepsLeft)
        return prefix + [next_token], remaining_steps

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "stepsLeft >= 1",
            "1 <= k <= |lm.Tokens|",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "|updated| == |prefix| + 1",
            "|updated| + remainingSteps == |prefix| + stepsLeft",
            "updated[|prefix|] in lm.Tokens",
            "parser.ValidNextToken(LongestValidSuffix(prefix), updated[|prefix|])",
        ),
        axiom=True,
        extern=True,
    )
    def AppendTopKConstrainedStep(self, prompt: Prefix, prefix: Prefix, k: int, stepsLeft: int) -> tuple[Prefix, int]:
        next_token, remaining_steps = self.TopKConstrainedStep(prompt, prefix, k, stepsLeft)
        return prefix + [next_token], remaining_steps

    @dafny_spec(
        kind="method",
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "token in lm.Tokens",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "updated == prefix + [token]",
            "|updated| + remainingSteps == |prefix| + stepsLeft",
        ),
    )
    def AppendForcedToken(self, prefix: Prefix, token: Token, stepsLeft: int) -> tuple[Prefix, int]:
        next_token, remaining_steps = self.ForcedTokenStep([], prefix, token, stepsLeft)
        return prefix + [next_token], remaining_steps

    @dafny_spec(
        kind="method",
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "LeftDelimiter in lm.Tokens",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "updated == prefix + [LeftDelimiter]",
            "|updated| + remainingSteps == |prefix| + stepsLeft",
        ),
    )
    def AppendLeftDelimiter(self, prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]:
        updated, remainingSteps = self.AppendForcedToken(prefix, LeftDelimiter, stepsLeft)
        return updated, remainingSteps

    @dafny_spec(
        kind="method",
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "RightDelimiter in lm.Tokens",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "updated == prefix + [RightDelimiter]",
            "|updated| + remainingSteps == |prefix| + stepsLeft",
        ),
    )
    def AppendRightDelimiter(self, prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]:
        updated, remainingSteps = self.AppendForcedToken(prefix, RightDelimiter, stepsLeft)
        return updated, remainingSteps

    # ── Checkpoint Utilities ──────────────────────────────────────────

    @dafny_spec(
        kind="function",
        ensures=("result == prefix",),
    )
    def Checkpoint(self, prefix: Prefix) -> Prefix:
        return prefix

    @dafny_spec(
        kind="function",
        ensures=("result == checkpoint",),
    )
    def RestoreCheckpoint(self, checkpoint: Prefix) -> Prefix:
        return checkpoint

    @dafny_spec(
        kind="function",
        reads=("this", "this.parser"),
        requires=("parser.IsValidPrefix([])",),
        ensures=(
            "IsDead(prefix) ==> result == checkpoint",
            "!IsDead(prefix) ==> result == prefix",
        ),
    )
    def RestoreIfDead(self, prefix: Prefix, checkpoint: Prefix) -> Prefix:
        return checkpoint if self.IsDead(prefix) else prefix

    # ── Budget Utilities ──────────────────────────────────────────────

    @dafny_spec(kind="predicate")
    def HasBudget(self, stepsLeft: int, needed: int) -> bool:
        return stepsLeft >= needed

    @dafny_spec(
        kind="function",
        reads=("this", "this.parser"),
        requires=("parser.IsValidPrefix([])",),
        ensures=(
            "result >= 0",
            "IsComplete(prefix) ==> result == 0",
            "!IsComplete(prefix) ==> result >= 1",
        ),
    )
    def MinStepsToComplete(self, prefix: Prefix) -> int:
        return self.ParserDistanceToComplete(prefix)


__all__ = [
    "MODULE_NAME",
    "Token",
    "Prefix",
    "Id",
    "Logit",
    "DafnySpec",
    "dafny_spec",
    "Contains",
    "PrefixContains",
    "DelimitedAnswerValidForParser",
    "LeftDelimiter",
    "RightDelimiter",
    "SpacedLeftDelimiter",
    "SpacedRightDelimiter",
    "LM",
    "Parser",
    "Delimiter",
    "CSDHelpers",
]
