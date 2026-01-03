import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import module_ as module_
import _dafny as _dafny
import System_ as System_

# Module: VerifiedDecoderAgent

class default__:
    def  __init__(self):
        pass

    @staticmethod
    def CRANEStyle(startDelim, endDelim):
        return Strategy_Window(startDelim, endDelim, TokenConstraint_GrammarMask(), TokenConstraint_NoConstraint())

    @staticmethod
    def RetryThenConstrained(k):
        return Strategy_TryK(k, Attempt_Unconstrained(), CheckPredicate_ParseOk(), Strategy_Constrained(TokenConstraint_GrammarMask()))

    @staticmethod
    def BestOfNWithRepair(n, repairRules):
        return Strategy_BestOfN(n, Strategy_TryK(1, Attempt_WithRepair(Attempt_Unconstrained(), repairRules), CheckPredicate_ParseOk(), Strategy_Constrained(TokenConstraint_GrammarMask())), CheckPredicate_ParseOk())

    @staticmethod
    def GuaranteesValidOutput(strategy):
        source0_ = strategy
        if True:
            if source0_.is_Constrained:
                return True
        if True:
            if source0_.is_Window:
                return True
        if True:
            if source0_.is_TryK:
                d_0_fallback_ = source0_.fallback
                return default__.GuaranteesValidOutput(d_0_fallback_)
        if True:
            if source0_.is_Cascade:
                d_1_strategies_ = source0_.strategies
                return ((len(d_1_strategies_)) > (0)) and (default__.GuaranteesValidOutput((d_1_strategies_)[(len(d_1_strategies_)) - (1)]))
        if True:
            if source0_.is_BestOfN:
                d_2_base_ = source0_.base
                return default__.GuaranteesValidOutput(d_2_base_)
        if True:
            return False


class LM:
    def  __init__(self):
        self.Logits: _dafny.Seq = _dafny.Seq({})
        self._Tokens: _dafny.Seq = _dafny.Seq({})
        self._Ids: _dafny.Seq = _dafny.Seq({})
        pass

    def __dafnystr__(self) -> str:
        return "VerifiedDecoderAgent.LM"
    def ValidTokensIdsLogits(self):
        def lambda0_(forall_var_0_):
            def lambda1_(forall_var_1_):
                d_1_j_: int = forall_var_1_
                return not (((((0) <= (d_0_i_)) and ((d_0_i_) < (len((self).Tokens)))) and (((0) <= (d_1_j_)) and ((d_1_j_) < (len((self).Tokens))))) and ((d_0_i_) != (d_1_j_))) or ((((self).Tokens)[d_0_i_]) != (((self).Tokens)[d_1_j_]))

            d_0_i_: int = forall_var_0_
            return _dafny.quantifier(_dafny.IntegerRange(0, len((self).Tokens)), True, lambda1_)

        def lambda2_(forall_var_2_):
            def lambda3_(exists_var_0_):
                d_3_i_: int = exists_var_0_
                return (((0) <= (d_3_i_)) and ((d_3_i_) < (len((self).Ids)))) and ((((self).Tokens)[d_3_i_]) == (d_2_token_))

            d_2_token_: _dafny.Seq = forall_var_2_
            return not ((d_2_token_) in ((self).Tokens)) or (_dafny.quantifier(_dafny.IntegerRange(0, len((self).Ids)), False, lambda3_))

        def lambda4_(forall_var_3_):
            d_4_i_: int = forall_var_3_
            return not (((0) <= (d_4_i_)) and ((d_4_i_) < (len((self).Ids)))) or (((d_4_i_) == (((self).Ids)[d_4_i_])) and ((d_4_i_) in ((self).Ids)))

        def lambda5_(forall_var_4_):
            d_5_i_: int = forall_var_4_
            return not (((0) <= (d_5_i_)) and ((d_5_i_) < (len(self.Logits)))) or ((((self.Logits)[d_5_i_]) <= (_dafny.BigRational('1e9'))) and (((self.Logits)[d_5_i_]) >= (_dafny.BigRational('-1e9'))))

        return (((((((len((self).Tokens)) == (len((self).Ids))) and ((len((self).Ids)) == (len(self.Logits)))) and (((len((self).Ids)) > (0)) and ((((self).Ids)[0]) == (0)))) and (_dafny.quantifier(_dafny.IntegerRange(0, len((self).Tokens)), True, lambda0_))) and (_dafny.quantifier(((self).Tokens).UniqueElements, True, lambda2_))) and (_dafny.quantifier(_dafny.IntegerRange(0, len((self).Ids)), True, lambda4_))) and (_dafny.quantifier(_dafny.IntegerRange(0, len(self.Logits)), True, lambda5_))

    def IdToToken(self, id_):
        return ((self).Tokens)[id_]

    def TokenToId(self, token):
        return (self).TokenToIdRecursive(token, (self).Tokens)

    def TokenToIdRecursive(self, token, tokens):
        d_0___accumulator_ = 0
        _this = self
        while True:
            with _dafny.label():
                if ((tokens)[0]) == (token):
                    return (0) + (d_0___accumulator_)
                elif True:
                    d_0___accumulator_ = (d_0___accumulator_) + (1)
                    in0_ = _this
                    in1_ = token
                    in2_ = _dafny.SeqWithoutIsStrInference((tokens)[1::])
                    _this = in0_
                    
                    token = in1_
                    tokens = in2_
                    raise _dafny.TailCall()
                break

    def IdToLogit(self, id_):
        return (self.Logits)[id_]

    def TokenToLogit(self, token):
        return (self).IdToLogit((self).TokenToId(token))

    def TokensToLogits(self, tokens):
        d_0___accumulator_ = _dafny.SeqWithoutIsStrInference([])
        _this = self
        while True:
            with _dafny.label():
                if (len(tokens)) == (1):
                    return (d_0___accumulator_) + (_dafny.SeqWithoutIsStrInference([(_this).TokenToLogit((tokens)[0])]))
                elif True:
                    d_0___accumulator_ = (d_0___accumulator_) + (_dafny.SeqWithoutIsStrInference([(_this).TokenToLogit((tokens)[0])]))
                    in0_ = _this
                    in1_ = _dafny.SeqWithoutIsStrInference((tokens)[1::])
                    _this = in0_
                    
                    tokens = in1_
                    raise _dafny.TailCall()
                break

    def IdsToLogits(self, ids):
        d_0___accumulator_ = _dafny.SeqWithoutIsStrInference([])
        _this = self
        while True:
            with _dafny.label():
                if (len(ids)) == (1):
                    return (d_0___accumulator_) + (_dafny.SeqWithoutIsStrInference([(_this).IdToLogit((ids)[0])]))
                elif True:
                    d_0___accumulator_ = (d_0___accumulator_) + (_dafny.SeqWithoutIsStrInference([(_this).IdToLogit((ids)[0])]))
                    in0_ = _this
                    in1_ = _dafny.SeqWithoutIsStrInference((ids)[1::])
                    _this = in0_
                    
                    ids = in1_
                    raise _dafny.TailCall()
                break

    def MaskToken(self, token):
        d_0_id_: int
        d_0_id_ = (self).TokenToId(token)
        (self).Logits = ((_dafny.SeqWithoutIsStrInference((self.Logits)[:d_0_id_:])) + (_dafny.SeqWithoutIsStrInference([_dafny.BigRational('0e0')]))) + (_dafny.SeqWithoutIsStrInference((self.Logits)[(d_0_id_) + (1)::]))

    def MaskTokens(self, tokens):
        d_0_N_: int
        d_0_N_ = len(tokens)
        d_1_i_: int
        d_1_i_ = 0
        while (d_1_i_) < (d_0_N_):
            (self).MaskToken((tokens)[d_1_i_])
            d_1_i_ = (d_1_i_) + (1)

    @property
    def Tokens(self):
        return self._Tokens
    @property
    def Ids(self):
        return self._Ids

class Parser:
    def  __init__(self):
        pass

    def __dafnystr__(self) -> str:
        return "VerifiedDecoderAgent.Parser"

class TokenConstraint:
    @classmethod
    def default(cls, ):
        return lambda: TokenConstraint_GrammarMask()
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    @property
    def is_GrammarMask(self) -> bool:
        return isinstance(self, TokenConstraint_GrammarMask)
    @property
    def is_Lookahead(self) -> bool:
        return isinstance(self, TokenConstraint_Lookahead)
    @property
    def is_LengthBound(self) -> bool:
        return isinstance(self, TokenConstraint_LengthBound)
    @property
    def is_BanTokens(self) -> bool:
        return isinstance(self, TokenConstraint_BanTokens)
    @property
    def is_AllowOnlyTokens(self) -> bool:
        return isinstance(self, TokenConstraint_AllowOnlyTokens)
    @property
    def is_Intersect(self) -> bool:
        return isinstance(self, TokenConstraint_Intersect)
    @property
    def is_Union(self) -> bool:
        return isinstance(self, TokenConstraint_Union)
    @property
    def is_NoConstraint(self) -> bool:
        return isinstance(self, TokenConstraint_NoConstraint)

class TokenConstraint_GrammarMask(TokenConstraint, NamedTuple('GrammarMask', [])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.TokenConstraint.GrammarMask'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, TokenConstraint_GrammarMask)
    def __hash__(self) -> int:
        return super().__hash__()

class TokenConstraint_Lookahead(TokenConstraint, NamedTuple('Lookahead', [('depth', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.TokenConstraint.Lookahead({_dafny.string_of(self.depth)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, TokenConstraint_Lookahead) and self.depth == __o.depth
    def __hash__(self) -> int:
        return super().__hash__()

class TokenConstraint_LengthBound(TokenConstraint, NamedTuple('LengthBound', [('min_', Any), ('max_', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.TokenConstraint.LengthBound({_dafny.string_of(self.min_)}, {_dafny.string_of(self.max_)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, TokenConstraint_LengthBound) and self.min_ == __o.min_ and self.max_ == __o.max_
    def __hash__(self) -> int:
        return super().__hash__()

class TokenConstraint_BanTokens(TokenConstraint, NamedTuple('BanTokens', [('banned', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.TokenConstraint.BanTokens({_dafny.string_of(self.banned)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, TokenConstraint_BanTokens) and self.banned == __o.banned
    def __hash__(self) -> int:
        return super().__hash__()

class TokenConstraint_AllowOnlyTokens(TokenConstraint, NamedTuple('AllowOnlyTokens', [('allowed', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.TokenConstraint.AllowOnlyTokens({_dafny.string_of(self.allowed)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, TokenConstraint_AllowOnlyTokens) and self.allowed == __o.allowed
    def __hash__(self) -> int:
        return super().__hash__()

class TokenConstraint_Intersect(TokenConstraint, NamedTuple('Intersect', [('a', Any), ('b', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.TokenConstraint.Intersect({_dafny.string_of(self.a)}, {_dafny.string_of(self.b)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, TokenConstraint_Intersect) and self.a == __o.a and self.b == __o.b
    def __hash__(self) -> int:
        return super().__hash__()

class TokenConstraint_Union(TokenConstraint, NamedTuple('Union', [('a', Any), ('b', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.TokenConstraint.Union({_dafny.string_of(self.a)}, {_dafny.string_of(self.b)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, TokenConstraint_Union) and self.a == __o.a and self.b == __o.b
    def __hash__(self) -> int:
        return super().__hash__()

class TokenConstraint_NoConstraint(TokenConstraint, NamedTuple('NoConstraint', [])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.TokenConstraint.NoConstraint'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, TokenConstraint_NoConstraint)
    def __hash__(self) -> int:
        return super().__hash__()


class RepairRules:
    @classmethod
    def default(cls, ):
        return lambda: RepairRules_BracketBalance()
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    @property
    def is_BracketBalance(self) -> bool:
        return isinstance(self, RepairRules_BracketBalance)
    @property
    def is_QuoteFix(self) -> bool:
        return isinstance(self, RepairRules_QuoteFix)
    @property
    def is_WhitespaceNormalize(self) -> bool:
        return isinstance(self, RepairRules_WhitespaceNormalize)
    @property
    def is_ComposedRepair(self) -> bool:
        return isinstance(self, RepairRules_ComposedRepair)
    @property
    def is_NoRepair(self) -> bool:
        return isinstance(self, RepairRules_NoRepair)

class RepairRules_BracketBalance(RepairRules, NamedTuple('BracketBalance', [])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.RepairRules.BracketBalance'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, RepairRules_BracketBalance)
    def __hash__(self) -> int:
        return super().__hash__()

class RepairRules_QuoteFix(RepairRules, NamedTuple('QuoteFix', [])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.RepairRules.QuoteFix'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, RepairRules_QuoteFix)
    def __hash__(self) -> int:
        return super().__hash__()

class RepairRules_WhitespaceNormalize(RepairRules, NamedTuple('WhitespaceNormalize', [])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.RepairRules.WhitespaceNormalize'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, RepairRules_WhitespaceNormalize)
    def __hash__(self) -> int:
        return super().__hash__()

class RepairRules_ComposedRepair(RepairRules, NamedTuple('ComposedRepair', [('a', Any), ('b', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.RepairRules.ComposedRepair({_dafny.string_of(self.a)}, {_dafny.string_of(self.b)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, RepairRules_ComposedRepair) and self.a == __o.a and self.b == __o.b
    def __hash__(self) -> int:
        return super().__hash__()

class RepairRules_NoRepair(RepairRules, NamedTuple('NoRepair', [])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.RepairRules.NoRepair'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, RepairRules_NoRepair)
    def __hash__(self) -> int:
        return super().__hash__()



class SeqOperation:
    @classmethod
    def default(cls, ):
        return lambda: SeqOperation_Repair(RepairRules.default()())
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    @property
    def is_Repair(self) -> bool:
        return isinstance(self, SeqOperation_Repair)
    @property
    def is_PrefixCompleteOp(self) -> bool:
        return isinstance(self, SeqOperation_PrefixCompleteOp)
    @property
    def is_ValidateOp(self) -> bool:
        return isinstance(self, SeqOperation_ValidateOp)
    @property
    def is_Identity(self) -> bool:
        return isinstance(self, SeqOperation_Identity)

class SeqOperation_Repair(SeqOperation, NamedTuple('Repair', [('rules', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.SeqOperation.Repair({_dafny.string_of(self.rules)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, SeqOperation_Repair) and self.rules == __o.rules
    def __hash__(self) -> int:
        return super().__hash__()

class SeqOperation_PrefixCompleteOp(SeqOperation, NamedTuple('PrefixCompleteOp', [('constraint', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.SeqOperation.PrefixCompleteOp({_dafny.string_of(self.constraint)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, SeqOperation_PrefixCompleteOp) and self.constraint == __o.constraint
    def __hash__(self) -> int:
        return super().__hash__()

class SeqOperation_ValidateOp(SeqOperation, NamedTuple('ValidateOp', [('pred', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.SeqOperation.ValidateOp({_dafny.string_of(self.pred)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, SeqOperation_ValidateOp) and self.pred == __o.pred
    def __hash__(self) -> int:
        return super().__hash__()

class SeqOperation_Identity(SeqOperation, NamedTuple('Identity', [])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.SeqOperation.Identity'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, SeqOperation_Identity)
    def __hash__(self) -> int:
        return super().__hash__()


class CheckPredicate:
    @classmethod
    def default(cls, ):
        return lambda: CheckPredicate_ParseOk()
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    @property
    def is_ParseOk(self) -> bool:
        return isinstance(self, CheckPredicate_ParseOk)
    @property
    def is_SemanticOk(self) -> bool:
        return isinstance(self, CheckPredicate_SemanticOk)
    @property
    def is_Both(self) -> bool:
        return isinstance(self, CheckPredicate_Both)
    @property
    def is_Either(self) -> bool:
        return isinstance(self, CheckPredicate_Either)

class CheckPredicate_ParseOk(CheckPredicate, NamedTuple('ParseOk', [])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.CheckPredicate.ParseOk'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, CheckPredicate_ParseOk)
    def __hash__(self) -> int:
        return super().__hash__()

class CheckPredicate_SemanticOk(CheckPredicate, NamedTuple('SemanticOk', [('pred', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.CheckPredicate.SemanticOk({_dafny.string_of(self.pred)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, CheckPredicate_SemanticOk) and self.pred == __o.pred
    def __hash__(self) -> int:
        return super().__hash__()

class CheckPredicate_Both(CheckPredicate, NamedTuple('Both', [('a', Any), ('b', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.CheckPredicate.Both({_dafny.string_of(self.a)}, {_dafny.string_of(self.b)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, CheckPredicate_Both) and self.a == __o.a and self.b == __o.b
    def __hash__(self) -> int:
        return super().__hash__()

class CheckPredicate_Either(CheckPredicate, NamedTuple('Either', [('a', Any), ('b', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.CheckPredicate.Either({_dafny.string_of(self.a)}, {_dafny.string_of(self.b)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, CheckPredicate_Either) and self.a == __o.a and self.b == __o.b
    def __hash__(self) -> int:
        return super().__hash__()


class Attempt:
    @classmethod
    def default(cls, ):
        return lambda: Attempt_Unconstrained()
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    @property
    def is_Unconstrained(self) -> bool:
        return isinstance(self, Attempt_Unconstrained)
    @property
    def is_ConstrainedAttempt(self) -> bool:
        return isinstance(self, Attempt_ConstrainedAttempt)
    @property
    def is_WithRepair(self) -> bool:
        return isinstance(self, Attempt_WithRepair)
    @property
    def is_WithSeqOp(self) -> bool:
        return isinstance(self, Attempt_WithSeqOp)

class Attempt_Unconstrained(Attempt, NamedTuple('Unconstrained', [])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.Attempt.Unconstrained'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Attempt_Unconstrained)
    def __hash__(self) -> int:
        return super().__hash__()

class Attempt_ConstrainedAttempt(Attempt, NamedTuple('ConstrainedAttempt', [('constraint', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.Attempt.ConstrainedAttempt({_dafny.string_of(self.constraint)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Attempt_ConstrainedAttempt) and self.constraint == __o.constraint
    def __hash__(self) -> int:
        return super().__hash__()

class Attempt_WithRepair(Attempt, NamedTuple('WithRepair', [('base', Any), ('rules', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.Attempt.WithRepair({_dafny.string_of(self.base)}, {_dafny.string_of(self.rules)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Attempt_WithRepair) and self.base == __o.base and self.rules == __o.rules
    def __hash__(self) -> int:
        return super().__hash__()

class Attempt_WithSeqOp(Attempt, NamedTuple('WithSeqOp', [('base', Any), ('op', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.Attempt.WithSeqOp({_dafny.string_of(self.base)}, {_dafny.string_of(self.op)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Attempt_WithSeqOp) and self.base == __o.base and self.op == __o.op
    def __hash__(self) -> int:
        return super().__hash__()


class Strategy:
    @classmethod
    def default(cls, ):
        return lambda: Strategy_Window(_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "")), TokenConstraint.default()(), TokenConstraint.default()())
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    @property
    def is_Window(self) -> bool:
        return isinstance(self, Strategy_Window)
    @property
    def is_TryK(self) -> bool:
        return isinstance(self, Strategy_TryK)
    @property
    def is_Cascade(self) -> bool:
        return isinstance(self, Strategy_Cascade)
    @property
    def is_BestOfN(self) -> bool:
        return isinstance(self, Strategy_BestOfN)
    @property
    def is_Constrained(self) -> bool:
        return isinstance(self, Strategy_Constrained)
    @property
    def is_Free(self) -> bool:
        return isinstance(self, Strategy_Free)

class Strategy_Window(Strategy, NamedTuple('Window', [('startDelim', Any), ('endDelim', Any), ('inside', Any), ('outside', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.Strategy.Window({self.startDelim.VerbatimString(True)}, {self.endDelim.VerbatimString(True)}, {_dafny.string_of(self.inside)}, {_dafny.string_of(self.outside)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Strategy_Window) and self.startDelim == __o.startDelim and self.endDelim == __o.endDelim and self.inside == __o.inside and self.outside == __o.outside
    def __hash__(self) -> int:
        return super().__hash__()

class Strategy_TryK(Strategy, NamedTuple('TryK', [('k', Any), ('attempt', Any), ('check', Any), ('fallback', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.Strategy.TryK({_dafny.string_of(self.k)}, {_dafny.string_of(self.attempt)}, {_dafny.string_of(self.check)}, {_dafny.string_of(self.fallback)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Strategy_TryK) and self.k == __o.k and self.attempt == __o.attempt and self.check == __o.check and self.fallback == __o.fallback
    def __hash__(self) -> int:
        return super().__hash__()

class Strategy_Cascade(Strategy, NamedTuple('Cascade', [('strategies', Any), ('check', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.Strategy.Cascade({_dafny.string_of(self.strategies)}, {_dafny.string_of(self.check)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Strategy_Cascade) and self.strategies == __o.strategies and self.check == __o.check
    def __hash__(self) -> int:
        return super().__hash__()

class Strategy_BestOfN(Strategy, NamedTuple('BestOfN', [('n', Any), ('base', Any), ('check', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.Strategy.BestOfN({_dafny.string_of(self.n)}, {_dafny.string_of(self.base)}, {_dafny.string_of(self.check)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Strategy_BestOfN) and self.n == __o.n and self.base == __o.base and self.check == __o.check
    def __hash__(self) -> int:
        return super().__hash__()

class Strategy_Constrained(Strategy, NamedTuple('Constrained', [('constraint', Any)])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.Strategy.Constrained({_dafny.string_of(self.constraint)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Strategy_Constrained) and self.constraint == __o.constraint
    def __hash__(self) -> int:
        return super().__hash__()

class Strategy_Free(Strategy, NamedTuple('Free', [])):
    def __dafnystr__(self) -> str:
        return f'VerifiedDecoderAgent.Strategy.Free'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Strategy_Free)
    def __hash__(self) -> int:
        return super().__hash__()

