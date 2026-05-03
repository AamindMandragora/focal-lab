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
            and all(-1000000000.0 <= logit <= 1000000000.0 for logit in self.Logits)
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
        self.Logits[token_id] = -1000000000.0

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
        return self.Logits[self.TokenToId(token)] == -1000000000.0

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
            "-1000000000.0 <= Logits[TokenToId(token)] <= 1000000000.0",
            "Logits[TokenToId(token)] == if old(Logits[TokenToId(token)]) + delta > 1000000000.0 then 1000000000.0 else if old(Logits[TokenToId(token)]) + delta < -1000000000.0 then -1000000000.0 else old(Logits[TokenToId(token)]) + delta",
            "forall t: Token :: t in Tokens && t != token ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def BiasToken(self, token: Token, delta: Logit) -> None:
        token_id = self.TokenToId(token)
        raw = self.Logits[token_id] + delta
        if raw > 1000000000.0:
            raw = 1000000000.0
        if raw < -1000000000.0:
            raw = -1000000000.0
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
            "-1000000000.0 <= Logits[TokenToId(token)] <= 1000000000.0",
            "forall t: Token :: t in Tokens && t != token ==> Logits[TokenToId(t)] == old(Logits[TokenToId(t)])",
        ),
    )
    def ScaleToken(self, token: Token, factor: Logit) -> None:
        token_id = self.TokenToId(token)
        raw = self.Logits[token_id] * factor
        if raw > 1000000000.0:
            raw = 1000000000.0
        if raw < -1000000000.0:
            raw = -1000000000.0
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
        requires=("ValidTokensIdsLogits()", "-1000000000.0 <= low", "low <= high", "high <= 1000000000.0"),
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
                self.Logits[i] = -1000000000.0
            i += 1

    @dafny_spec(
        kind="method",
        requires=("ValidTokensIdsLogits()", "1 <= k <= |Tokens|"),
        ensures=("ValidTokensIdsLogits()", "|result| == k"),
        axiom=True,
        extern=True,
    )
    def GetTopKTokens(self, k: int) -> Prefix:
        result: Prefix = []
        chosen: list[Id] = []
        remaining = k
        while remaining > 0:
            best_id = -1
            i = 0
            while i < len(self.Tokens):
                if i not in chosen and (best_id == -1 or self.Logits[i] > self.Logits[best_id]):
                    best_id = i
                i += 1
            result = result + [self.Tokens[best_id]]
            chosen = chosen + [best_id]
            remaining -= 1
        return result

    @dafny_spec(
        kind="method",
        requires=("ValidTokensIdsLogits()",),
        ensures=("ValidTokensIdsLogits()", "result in Tokens"),
        axiom=True,
        extern=True,
    )
    def GetMaxLogitToken(self) -> Token:
        best_id = 0
        i = 1
        while i < len(self.Tokens):
            if self.Logits[i] > self.Logits[best_id]:
                best_id = i
            i += 1
        return self.Tokens[best_id]

    @dafny_spec(
        kind="method",
        requires=("ValidTokensIdsLogits()", "HasUnmaskedToken()"),
        ensures=("ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def GetMaxUnmaskedLogit(self) -> Logit:
        found = False
        best = -1000000000.0
        i = 0
        while i < len(self.Tokens):
            token = self.Tokens[i]
            if not self.IsMasked(token):
                if not found or self.Logits[i] > best:
                    best = self.Logits[i]
                    found = True
            i += 1
        if not found:
            raise ValueError("No unmasked token is available")
        return best

    @dafny_spec(
        kind="method",
        requires=(
            "ValidTokensIdsLogits()",
            "exists a: Token, b: Token :: a in Tokens && b in Tokens && a != b && !IsMasked(a) && !IsMasked(b)",
        ),
        ensures=("ValidTokensIdsLogits()", "result >= 0.0"),
        axiom=True,
        extern=True,
    )
    def GetLogitGap(self) -> Logit:
        count = 0
        best = -1000000000.0
        second = -1000000000.0
        i = 0
        while i < len(self.Tokens):
            token = self.Tokens[i]
            if not self.IsMasked(token):
                logit = self.Logits[i]
                if count == 0:
                    best = logit
                    count = 1
                elif count == 1:
                    if logit > best:
                        second = best
                        best = logit
                    else:
                        second = logit
                    count = 2
                else:
                    if logit > best:
                        second = best
                        best = logit
                    elif logit > second:
                        second = logit
            i += 1
        if count < 2:
            raise ValueError("At least two unmasked tokens are required")
        return best - second

    @dafny_spec(
        kind="method",
        requires=("ValidTokensIdsLogits()",),
        ensures=("ValidTokensIdsLogits()", "|result| == Logits.Length"),
        axiom=True,
        extern=True,
    )
    def SnapshotLogits(self) -> list[Logit]:
        snapshot: list[Logit] = []
        i = 0
        while i < len(self.Logits):
            snapshot = snapshot + [self.Logits[i]]
            i += 1
        return snapshot

    @dafny_spec(
        kind="method",
        modifies=("this.Logits",),
        requires=("ValidTokensIdsLogits()", "|snapshot| == Logits.Length"),
        ensures=("ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def RestoreLogits(self, snapshot: list[Logit]) -> None:
        i = 0
        while i < len(self.Logits):
            self.Logits[i] = snapshot[i]
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
            if logit == -1000000000.0:
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
        kind="method",
        requires=("IsValidPrefix(prefix)",),
    )
    def ValidNextTokensInSet(self, prefix: Prefix, candidates: Prefix) -> Prefix:
        valid = self.ValidNextTokens(prefix)
        result: Prefix = []
        i = 0
        while i < len(valid):
            if valid[i] in candidates:
                result = result + [valid[i]]
            i += 1
        return result

    @dafny_spec(
        kind="method",
        requires=("IsValidPrefix(prefix_a)", "IsValidPrefix(prefix_b)"),
    )
    def SharesParserState(self, prefix_a: Prefix, prefix_b: Prefix) -> bool:
        valid_a = self.ValidNextTokens(prefix_a)
        valid_b = self.ValidNextTokens(prefix_b)
        i = 0
        while i < len(valid_a):
            if valid_a[i] not in valid_b:
                return False
            i += 1
        i = 0
        while i < len(valid_b):
            if valid_b[i] not in valid_a:
                return False
            i += 1
        return True

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

    # ── Prefix Scanning Utilities ────────────────────────────────────

    @dafny_spec(
        kind="method",
        ensures=(
            "found ==> token in generated",
            "!found ==> token == \"\"",
        ),
        decreases=("|generated|",),
    )
    def LastTokenBefore(self, generated: Prefix, target: Token) -> tuple[Token, bool]:
        i = len(generated) - 1
        # invariant -1 <= i < |generated|
        # decreases i + 1
        while i >= 1:
            if generated[i] == target:
                return generated[i - 1], True
            i -= 1
        return "", False

    @dafny_spec(
        kind="function",
        ensures=("result >= 0",),
        decreases=("|generated|",),
    )
    def CountOccurrences(self, generated: Prefix, target: Token) -> int:
        return (
            0
            if len(generated) == 0
            else ((1 if generated[0] == target else 0) + self.CountOccurrences(generated[1:], target))
        )

    @dafny_spec(
        kind="function",
        ensures=("0 <= result <= |generated|",),
        decreases=("|generated|",),
    )
    def TokensSinceLastDelimiter(self, generated: Prefix) -> int:
        return (
            0
            if len(generated) == 0
            else (
                0
                if (
                    generated[len(generated) - 1] == LeftDelimiter
                    or generated[len(generated) - 1] == RightDelimiter
                    or generated[len(generated) - 1] == SpacedLeftDelimiter
                    or generated[len(generated) - 1] == SpacedRightDelimiter
                )
                else 1 + self.TokensSinceLastDelimiter(generated[:-1])
            )
        )

    @dafny_spec(
        kind="function",
        requires=("|ngram| >= 1",),
        ensures=("result >= 0",),
        decreases=("|generated|",),
    )
    def NgramCount(self, generated: Prefix, ngram: Prefix) -> int:
        return (
            0
            if len(generated) < len(ngram)
            else ((1 if generated[: len(ngram)] == ngram else 0) + self.NgramCount(generated[1:], ngram))
        )

    @dafny_spec(
        kind="function",
        requires=("n >= 0",),
        ensures=("0 <= |result| <= |generated|",),
    )
    def LastNTokens(self, generated: Prefix, n: int) -> Prefix:
        return generated if n >= len(generated) else generated[len(generated) - n :]

    @dafny_spec(
        kind="method",
        ensures=("-1 <= result < |generated|",),
    )
    def FindLastIndex(self, generated: Prefix, target: Token) -> int:
        i = len(generated) - 1
        while i >= 0:
            if generated[i] == target:
                return i
            i -= 1
        return -1

    @dafny_spec(
        kind="function",
        requires=("0 <= start <= |generated|",),
        ensures=("|result| == |generated| - start",),
    )
    def SliceFrom(self, generated: Prefix, start: int) -> Prefix:
        return generated[start:]

    @dafny_spec(
        kind="function",
        requires=("0 <= start <= end <= |generated|",),
        ensures=("|result| == end - start",),
    )
    def SliceRange(self, generated: Prefix, start: int, end: int) -> Prefix:
        return generated[start:end]

    # ── Primitive Step Functions ──────────────────────────────────────
    # Ordinary unconstrained steps mask delimiters so free-form reasoning
    # cannot accidentally open/close answer spans. Natural delimiter mode
    # uses the allow/nudge variants below when a policy deliberately wants
    # the LM to choose the opening delimiter.

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
        axiom=True,
        extern=True,
    )
    def UnconstrainedStep(self, prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]:
        self.lm.ValidTokensIdsLogitsAlways()
        self.lm.GenerateLogits(generated if len(prompt) == 0 else prompt + generated)
        if len(self.lm.Tokens) > 4:
            self.MaskAllDelimiters(generated)
        next_token = self.lm.ChooseNextToken()
        return next_token, stepsLeft - 1

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
        axiom=True,
        extern=True,
    )
    def UnconstrainedAllowLeftDelimiterStep(self, prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]:
        self.lm.ValidTokensIdsLogitsAlways()
        self.lm.GenerateLogits(generated if len(prompt) == 0 else prompt + generated)
        if len(self.lm.Tokens) > 4:
            self.MaskRightDelimiters(generated)
        next_token = self.lm.ChooseNextToken()
        return next_token, stepsLeft - 1

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "stepsLeft >= 1",
            "bias > 0.0",
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
    def UnconstrainedBiasLeftDelimiterStep(
        self,
        prompt: Prefix,
        generated: Prefix,
        bias: Logit,
        stepsLeft: int,
    ) -> tuple[Token, int]:
        self.lm.ValidTokensIdsLogitsAlways()
        self.lm.GenerateLogits(prompt + generated)
        if len(self.lm.Tokens) > 4:
            self.MaskRightDelimiters(generated)
            self.BiasLeftDelimiters(bias)
        next_token = self.lm.ChooseNextToken()
        return next_token, stepsLeft - 1

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
        axiom=True,
        extern=True,
    )
    def UnconstrainedNudgeLeftDelimiterStep(self, prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]:
        self.lm.ValidTokensIdsLogitsAlways()
        self.lm.GenerateLogits(generated if len(prompt) == 0 else prompt + generated)
        if len(self.lm.Tokens) > 4:
            self.MaskRightDelimiters(generated)
            self.BiasLeftDelimiters(5.0)
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
        axiom=True,
        extern=True,
    )
    def ConstrainedStep(self, prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]:
        self.LongestValidSuffixIsValid(generated)
        suffix = self.LongestValidSuffix(generated)
        self.AllValidNextTokensInLM(suffix)
        self.lm.GenerateLogits(generated if len(prompt) == 0 else prompt + generated)
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
            "RightDelimiter in lm.Tokens",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "nextToken in lm.Tokens",
            "!lm.IsMasked(nextToken)",
            "(nextToken == RightDelimiter || nextToken == SpacedRightDelimiter) ==> parser.IsCompletePrefix(LongestValidSuffix(generated))",
            "(nextToken != RightDelimiter && nextToken != SpacedRightDelimiter) ==> parser.ValidNextToken(LongestValidSuffix(generated), nextToken)",
        ),
        axiom=True,
        extern=True,
    )
    def ConstrainedOrRightDelimiterStep(self, prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]:
        self.LongestValidSuffixIsValid(generated)
        suffix = self.LongestValidSuffix(generated)
        self.AllValidNextTokensInLM(suffix)
        self.lm.GenerateLogits(generated if len(prompt) == 0 else prompt + generated)
        valid_tokens = self.parser.ValidNextTokens(suffix)
        if self.parser.IsCompletePrefix(suffix):
            if SpacedRightDelimiter in self.lm.Tokens:
                self.lm.MaskTokensExcept(valid_tokens + [RightDelimiter, SpacedRightDelimiter])
            else:
                self.lm.MaskTokensExcept(valid_tokens + [RightDelimiter])
        else:
            self.lm.MaskTokensExcept(valid_tokens)
        next_token = self.lm.ChooseNextToken()
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

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix(grammarPrefix)",
            "!parser.IsCompletePrefix(grammarPrefix)",
            "stepsLeft >= 1",
        ),
        ensures=("this.lm.ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def CustomPrefixStep(self, lmInput: Prefix, grammarPrefix: Prefix, stepsLeft: int) -> tuple[Token, int]:
        self.AllValidNextTokensInLM(grammarPrefix)
        self.lm.GenerateLogits(lmInput)
        self.lm.MaskTokensExcept(self.parser.ValidNextTokens(grammarPrefix))
        next_token = self.lm.ChooseNextToken()
        return next_token, stepsLeft - 1

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix(grammarPrefix)",
            "stepsLeft >= 1",
            "penalty > 0.0",
        ),
        ensures=("this.lm.ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def CustomPrefixSoftStep(
        self,
        lmInput: Prefix,
        grammarPrefix: Prefix,
        penalty: Logit,
        stepsLeft: int,
    ) -> tuple[Token, int]:
        self.lm.GenerateLogits(lmInput)
        self.SoftConstrainToGrammarOnPrefix(grammarPrefix, penalty)
        next_token = self.lm.ChooseNextToken()
        return next_token, stepsLeft - 1

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix(grammarPrefix)",
            "!parser.IsCompletePrefix(grammarPrefix)",
            "stepsLeft >= 1",
            "1 <= k <= |lm.Tokens|",
        ),
        ensures=("this.lm.ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def CustomPrefixTopKStep(self, lmInput: Prefix, grammarPrefix: Prefix, k: int, stepsLeft: int) -> tuple[Token, int]:
        self.AllValidNextTokensInLM(grammarPrefix)
        self.lm.GenerateLogits(lmInput)
        self.lm.TopKFilter(k)
        self.lm.MaskTokensExcept(self.parser.ValidNextTokens(grammarPrefix))
        next_token = self.lm.ChooseNextToken()
        return next_token, stepsLeft - 1

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
        axiom=True,
        extern=True,
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
        axiom=True,
        extern=True,
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
        axiom=True,
        extern=True,
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
        axiom=True,
        extern=True,
    )
    def BiasRightDelimiters(self, bias: Logit) -> None:
        if RightDelimiter in self.lm.Tokens:
            self.lm.BiasToken(RightDelimiter, bias)
        if SpacedRightDelimiter in self.lm.Tokens:
            self.lm.BiasToken(SpacedRightDelimiter, bias)

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "forall t :: t in tokens ==> t in lm.Tokens",
            "bonus > 0.0",
        ),
        ensures=("this.lm.ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def BiasTokenGroup(self, tokens: Prefix, bonus: Logit) -> None:
        if len(tokens) > 0:
            self.lm.BiasTokens(tokens, bonus)

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "forall t :: t in tokens ==> t in lm.Tokens",
            "penalty > 0.0",
        ),
        ensures=("this.lm.ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def PenalizeTokenGroup(self, tokens: Prefix, penalty: Logit) -> None:
        if len(tokens) > 0:
            self.lm.BiasTokens(tokens, -penalty)

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=("this.lm.ValidTokensIdsLogits()", "parser.IsValidPrefix(grammarPrefix)"),
        ensures=("this.lm.ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def IntersectWithGrammarOnPrefix(self, grammarPrefix: Prefix) -> None:
        self.AllValidNextTokensInLM(grammarPrefix)
        self.lm.MaskTokensExcept(self.parser.ValidNextTokens(grammarPrefix))

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix(grammarPrefix)",
            "penalty > 0.0",
        ),
        ensures=("this.lm.ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def SoftConstrainToGrammarOnPrefix(self, grammarPrefix: Prefix, penalty: Logit) -> None:
        valid_tokens = self.parser.ValidNextTokens(grammarPrefix)
        invalid_tokens: Prefix = []
        i = 0
        while i < len(self.lm.Tokens):
            if self.lm.Tokens[i] not in valid_tokens:
                invalid_tokens = invalid_tokens + [self.lm.Tokens[i]]
            i += 1
        self.PenalizeTokenGroup(invalid_tokens, penalty)

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=("this.lm.ValidTokensIdsLogits()",),
        ensures=("this.lm.ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
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
        axiom=True,
        extern=True,
    )
    def AppendUnconstrainedStep(self, prompt: Prefix, prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]:
        next_token, remaining = self.UnconstrainedStep(prompt, prefix, stepsLeft)
        return prefix + [next_token], remaining

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
        axiom=True,
        extern=True,
    )
    def AppendUnconstrainedAllowLeftDelimiterStep(
        self,
        prompt: Prefix,
        prefix: Prefix,
        stepsLeft: int,
    ) -> tuple[Prefix, int]:
        next_token, remaining = self.UnconstrainedAllowLeftDelimiterStep(prompt, prefix, stepsLeft)
        return prefix + [next_token], remaining

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
        axiom=True,
        extern=True,
    )
    def AppendUnconstrainedNudgeLeftDelimiterStep(
        self,
        prompt: Prefix,
        prefix: Prefix,
        stepsLeft: int,
    ) -> tuple[Prefix, int]:
        next_token, remaining = self.UnconstrainedNudgeLeftDelimiterStep(prompt, prefix, stepsLeft)
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
        axiom=True,
        extern=True,
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
            "RightDelimiter in lm.Tokens",
            "stepsLeft >= 1",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "remainingSteps == stepsLeft - 1",
            "remainingSteps >= 0",
            "|updated| == |prefix| + 1",
            "|updated| + remainingSteps == |prefix| + stepsLeft",
            "updated[|prefix|] in lm.Tokens",
            "(updated[|prefix|] == RightDelimiter || updated[|prefix|] == SpacedRightDelimiter) ==> parser.IsCompletePrefix(LongestValidSuffix(prefix))",
            "(updated[|prefix|] != RightDelimiter && updated[|prefix|] != SpacedRightDelimiter) ==> parser.ValidNextToken(LongestValidSuffix(prefix), updated[|prefix|])",
        ),
        axiom=True,
        extern=True,
    )
    def AppendConstrainedOrRightDelimiterStep(
        self,
        prompt: Prefix,
        prefix: Prefix,
        stepsLeft: int,
    ) -> tuple[Prefix, int]:
        next_token, remaining = self.ConstrainedOrRightDelimiterStep(prompt, prefix, stepsLeft)
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

    # ── Compatibility Helpers (Span-State Strategies) ────────────────
    # These are convenience adapters for richer strategies that track
    # explicit inside-span/current-constrained state.

    @dafny_spec(
        kind="function",
        reads=("this", "this.parser"),
        requires=("parser.IsValidPrefix([])",),
        ensures=("result >= 0",),
    )
    def ValidTokenCount(self, prefix: Prefix) -> int:
        return self.ValidContinuationCount(prefix)

    @dafny_spec(
        kind="method",
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "LeftDelimiter in lm.Tokens",
            "stepsLeft >= 1",
        ),
        ensures=("this.lm.ValidTokensIdsLogits()",),
    )
    def OpenConstrainedSpan(self, prefix: Prefix, stepsLeft: int) -> tuple[Prefix, bool, Prefix, int]:
        updated, remainingSteps = self.AppendLeftDelimiter(prefix, stepsLeft)
        return updated, True, [], remainingSteps

    @dafny_spec(
        kind="method",
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "RightDelimiter in lm.Tokens",
            "stepsLeft >= 1",
            "parser.IsValidPrefix([])",
            "parser.IsValidPrefix(currentConstrained)",
            "parser.IsCompletePrefix(currentConstrained)",
            "|currentConstrained| <= |prefix|",
            "prefix[|prefix| - |currentConstrained|..] == currentConstrained",
        ),
        ensures=("this.lm.ValidTokensIdsLogits()",),
    )
    def CloseConstrainedSpan(self, prefix: Prefix, currentConstrained: Prefix, stepsLeft: int) -> tuple[Prefix, bool, Prefix, int]:
        updated, remainingSteps = self.AppendRightDelimiter(prefix, stepsLeft)
        return updated, False, [], remainingSteps

    @dafny_spec(
        kind="method",
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "parser.IsValidPrefix(currentConstrained)",
            "!parser.IsCompletePrefix(currentConstrained)",
            "token in lm.Tokens",
            "parser.ValidNextToken(currentConstrained, token)",
        ),
        ensures=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix(currentConstrained + [token])",
        ),
    )
    def AppendConstrainedToken(self, prefix: Prefix, currentConstrained: Prefix, token: Token) -> tuple[Prefix, bool, Prefix]:
        updated = prefix + [token]
        updated_constrained = currentConstrained + [token]
        return updated, True, updated_constrained

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "parser.IsValidPrefix(currentConstrained)",
            "!parser.IsCompletePrefix(currentConstrained)",
            "stepsLeft >= 1",
            "bonus > 0.0",
            "narrowThreshold >= 0",
            "eosToken in lm.Tokens",
            "|validTokenGroups| >= 0",
            "forall g: seq<Token> :: g in validTokenGroups ==> forall t: Token :: t in g ==> t in lm.Tokens",
            "forall t: Token :: t in parser.ValidNextTokens(currentConstrained) ==> t in lm.Tokens",
        ),
        ensures=("this.lm.ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def AdaptiveConstrainedStep(
        self,
        prompt: Prefix,
        stablePrefix: Prefix,
        currentConstrained: Prefix,
        validTokenGroups: list[list[Token]],
        bonus: Logit,
        narrowThreshold: int,
        eosToken: Token,
        stepsLeft: int,
    ) -> tuple[Token, int]:
        self.lm.GenerateLogits(prompt + stablePrefix + currentConstrained)
        valid_tokens = self.parser.ValidNextTokens(currentConstrained)
        self.lm.MaskTokensExcept(valid_tokens)
        if self.parser.ValidContinuationCount(currentConstrained) > narrowThreshold:
            group_index = 0
            # invariant 0 <= group_index <= |validTokenGroups|
            # invariant lm.ValidTokensIdsLogits()
            while group_index < len(validTokenGroups):
                group = validTokenGroups[group_index]
                token_index = 0
                # invariant 0 <= token_index <= |group|
                # invariant lm.ValidTokensIdsLogits()
                while token_index < len(group):
                    token = group[token_index]
                    if token in valid_tokens:
                        self.lm.BiasToken(token, bonus)
                    token_index += 1
                group_index += 1
        next_token = self.lm.ChooseNextToken()
        if next_token == eosToken:
            return eosToken, stepsLeft - 1
        return next_token, stepsLeft - 1

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "parser.IsValidPrefix(currentConstrained)",
            "!parser.IsCompletePrefix(currentConstrained)",
            "stepsLeft >= 1",
            "bonus > 0.0",
            "|validTokenGroups| >= 0",
            "forall g: seq<Token> :: g in validTokenGroups ==> forall t: Token :: t in g ==> t in lm.Tokens",
            "forall t: Token :: t in parser.ValidNextTokens(currentConstrained) ==> t in lm.Tokens",
        ),
        ensures=("this.lm.ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def GroupBoostedConstrainedStep(
        self,
        prompt: Prefix,
        stablePrefix: Prefix,
        currentConstrained: Prefix,
        validTokenGroups: list[list[Token]],
        bonus: Logit,
        stepsLeft: int,
    ) -> tuple[Token, int]:
        self.lm.GenerateLogits(prompt + stablePrefix + currentConstrained)
        valid_tokens = self.parser.ValidNextTokens(currentConstrained)
        self.lm.MaskTokensExcept(valid_tokens)
        group_index = 0
        while group_index < len(validTokenGroups):
            group = validTokenGroups[group_index]
            token_index = 0
            while token_index < len(group):
                token = group[token_index]
                if token in valid_tokens:
                    self.lm.BiasToken(token, bonus)
                token_index += 1
            group_index += 1
        next_token = self.lm.ChooseNextToken()
        return next_token, stepsLeft - 1

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "parser.IsValidPrefix(currentConstrained)",
            "!parser.IsCompletePrefix(currentConstrained)",
            "stepsLeft >= 1",
            "penalty > 0.0",
            "|penaltyTokens| > 0",
            "forall t: Token :: t in penaltyTokens ==> t in lm.Tokens",
            "forall t: Token :: t in parser.ValidNextTokens(currentConstrained) ==> t in lm.Tokens",
        ),
        ensures=("this.lm.ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def PenalizedConstrainedStep(
        self,
        prompt: Prefix,
        stablePrefix: Prefix,
        currentConstrained: Prefix,
        penaltyTokens: Prefix,
        penalty: Logit,
        stepsLeft: int,
    ) -> tuple[Token, int]:
        self.lm.GenerateLogits(prompt + stablePrefix + currentConstrained)
        valid_tokens = self.parser.ValidNextTokens(currentConstrained)
        self.lm.MaskTokensExcept(valid_tokens)
        self.PenalizeTokenGroup(penaltyTokens, penalty)
        next_token = self.lm.ChooseNextToken()
        return next_token, stepsLeft - 1

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "parser.IsValidPrefix([])",
            "parser.IsValidPrefix(currentConstrained)",
            "stepsLeft >= 0",
            "numTokens >= 0",
        ),
        ensures=("this.lm.ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def SpeculativeConstrain(
        self,
        prompt: Prefix,
        generated: Prefix,
        currentConstrained: Prefix,
        numTokens: int,
        stepsLeft: int,
    ) -> tuple[Prefix, Prefix, int]:
        candidate_tokens: Prefix = []
        updated_constrained = currentConstrained
        remaining_steps = stepsLeft
        i = 0
        while (
            i < numTokens
            and remaining_steps > 0
            and not self.parser.IsCompletePrefix(updated_constrained)
        ):
            self.AllValidNextTokensInLM(updated_constrained)
            self.lm.GenerateLogits(prompt + generated + candidate_tokens)
            self.lm.MaskTokensExcept(self.parser.ValidNextTokens(updated_constrained))
            next_token = self.lm.ChooseNextToken()
            candidate_tokens = candidate_tokens + [next_token]
            updated_constrained = updated_constrained + [next_token]
            remaining_steps -= 1
            i += 1
        return candidate_tokens, updated_constrained, remaining_steps

    @dafny_spec(
        kind="method",
        modifies=("this.lm.Logits",),
        requires=(
            "this.lm.ValidTokensIdsLogits()",
            "forall t :: t in candidateTokens ==> t in lm.Tokens",
        ),
        ensures=("this.lm.ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def ScoreCandidate(self, prompt: Prefix, generated: Prefix, candidateTokens: Prefix) -> Logit:
        total = 0.0
        history: Prefix = []
        i = 0
        while i < len(candidateTokens):
            self.lm.GenerateLogits(prompt + generated + history)
            total += self.lm.TokenToLogit(candidateTokens[i])
            history = history + [candidateTokens[i]]
            i += 1
        return total

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


class CheckpointStack:
    stack: list[Prefix]

    @dafny_spec(
        kind="constructor",
        ensures=("Depth() == 0",),
    )
    def __init__(self) -> None:
        self.stack = []

    @dafny_spec(
        kind="method",
        modifies=("this",),
        ensures=("Depth() >= 1",),
    )
    def Push(self, prefix: Prefix) -> None:
        self.stack = self.stack + [prefix]

    @dafny_spec(
        kind="method",
        modifies=("this",),
        requires=("Depth() > 0",),
    )
    def Pop(self) -> Prefix:
        top = self.stack[len(self.stack) - 1]
        self.stack = self.stack[:-1]
        return top

    @dafny_spec(
        kind="method",
        requires=("Depth() > 0",),
    )
    def Peek(self) -> Prefix:
        return self.stack[len(self.stack) - 1]

    @dafny_spec(kind="function", reads=("this",))
    def Depth(self) -> int:
        return len(self.stack)

    @dafny_spec(kind="predicate", reads=("this",))
    def IsEmpty(self) -> bool:
        return self.Depth() == 0


class RepetitionTracker:
    ngramSize: int
    history: Prefix

    @dafny_spec(
        kind="constructor",
        requires=("ngramSize >= 1",),
        ensures=("this.ngramSize == ngramSize",),
    )
    def __init__(self, ngramSize: int) -> None:
        self.ngramSize = ngramSize
        self.history = []

    @dafny_spec(
        kind="function",
        requires=("|ngram| >= 1",),
        decreases=("|prefix|",),
    )
    def CountNgram(self, prefix: Prefix, ngram: Prefix) -> int:
        return (
            0
            if len(prefix) < len(ngram)
            else ((1 if prefix[: len(ngram)] == ngram else 0) + self.CountNgram(prefix[1:], ngram))
        )

    @dafny_spec(
        kind="function",
        requires=("|ngram| >= 1",),
        decreases=("|prefix|",),
    )
    def CountNgramLogit(self, prefix: Prefix, ngram: Prefix) -> Logit:
        return (
            0.0
            if len(prefix) < len(ngram)
            else ((1.0 if prefix[: len(ngram)] == ngram else 0.0) + self.CountNgramLogit(prefix[1:], ngram))
        )

    @dafny_spec(
        kind="method",
        modifies=("this",),
        ensures=("history == old(history) + [token]",),
    )
    def RecordToken(self, token: Token) -> None:
        self.history = self.history + [token]

    @dafny_spec(
        kind="function",
        requires=("|ngram| == ngramSize",),
        ensures=("result >= 0",),
        reads=("this",),
        axiom=True,
    )
    def GetCount(self, ngram: Prefix) -> int:
        return self.CountNgram(self.history, ngram)

    @dafny_spec(kind="function", reads=("this",), axiom=True)
    def GetRepetitionPenalty(self, token: Token) -> Logit:
        return (
            self.CountNgramLogit(self.history, [token])
            if self.ngramSize == 1
            else (
                0.0
                if len(self.history) < self.ngramSize - 1
                else self.CountNgramLogit(
                    self.history,
                    self.history[len(self.history) - (self.ngramSize - 1) :] + [token],
                )
            )
        )

    @dafny_spec(
        kind="method",
        modifies=("lm.Logits",),
        requires=("lm.ValidTokensIdsLogits()",),
        ensures=("lm.ValidTokensIdsLogits()",),
        axiom=True,
        extern=True,
    )
    def ApplyRepetitionPenalties(self, lm: LM) -> None:
        i = 0
        while i < len(lm.Tokens):
            lm.BiasToken(lm.Tokens[i], -self.GetRepetitionPenalty(lm.Tokens[i]))
            i += 1


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
    "CheckpointStack",
    "RepetitionTracker",
]
