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
    def Contains(s, sub):
        def lambda0_(exists_var_0_):
            def lambda1_(exists_var_1_):
                d_1_j_: int = exists_var_1_
                return ((((0) <= (d_0_i_)) and ((d_0_i_) <= (d_1_j_))) and ((d_1_j_) <= (len(s)))) and ((_dafny.SeqWithoutIsStrInference((s)[d_0_i_:d_1_j_:])) == (sub))

            d_0_i_: int = exists_var_0_
            return _dafny.quantifier(_dafny.IntegerRange(d_0_i_, (len(s)) + (1)), False, lambda1_)

        return _dafny.quantifier(_dafny.IntegerRange(0, ((len(s)) + (1)) + (1)), False, lambda0_)

    @staticmethod
    def PrefixContains(p, t):
        def lambda0_(exists_var_0_):
            d_0_i_: int = exists_var_0_
            return (((0) <= (d_0_i_)) and ((d_0_i_) < (len(p)))) and (((p)[d_0_i_]) == (t))

        return _dafny.quantifier(_dafny.IntegerRange(0, len(p)), False, lambda0_)

    @staticmethod
    def DelimitedAnswerValidForParser(parser, prefix):
        return (default__.PrefixContains(prefix, default__.LeftDelimiter)) and (default__.PrefixContains(prefix, default__.RightDelimiter))

    @_dafny.classproperty
    def LeftDelimiter(instance):
        return _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
    @_dafny.classproperty
    def RightDelimiter(instance):
        return _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))
    @_dafny.classproperty
    def SpacedLeftDelimiter(instance):
        return _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<"))
    @_dafny.classproperty
    def SpacedRightDelimiter(instance):
        return _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>"))

class LM:
    def  __init__(self):
        self.Logits: _dafny.Array = _dafny.Array(None, 0)
        self._Tokens: _dafny.Seq = _dafny.Seq({})
        self._Ids: _dafny.Seq = _dafny.Seq({})
        pass

    def __dafnystr__(self) -> str:
        return "VerifiedDecoderAgent.LM"
    def ValidTokensIdsLogits(self):
        def lambda0_(forall_var_0_):
            d_0_i_: int = forall_var_0_
            return not (((0) <= (d_0_i_)) and ((d_0_i_) < (len((self).Ids)))) or (((d_0_i_) == (((self).Ids)[d_0_i_])) and ((d_0_i_) in ((self).Ids)))

        def lambda1_(forall_var_1_):
            def lambda2_(forall_var_2_):
                d_2_j_: int = forall_var_2_
                return not (((((0) <= (d_1_i_)) and ((d_1_i_) < (len((self).Tokens)))) and (((0) <= (d_2_j_)) and ((d_2_j_) < (len((self).Tokens))))) and ((d_1_i_) != (d_2_j_))) or ((((self).Tokens)[d_1_i_]) != (((self).Tokens)[d_2_j_]))

            d_1_i_: int = forall_var_1_
            return _dafny.quantifier(_dafny.IntegerRange(0, len((self).Tokens)), True, lambda2_)

        def lambda3_(forall_var_3_):
            def lambda4_(exists_var_0_):
                d_4_i_: int = exists_var_0_
                return (((0) <= (d_4_i_)) and ((d_4_i_) < (len((self).Ids)))) and ((((self).Tokens)[d_4_i_]) == (d_3_token_))

            d_3_token_: _dafny.Seq = forall_var_3_
            return not ((d_3_token_) in ((self).Tokens)) or (_dafny.quantifier(_dafny.IntegerRange(0, len((self).Ids)), False, lambda4_))

        def lambda5_(forall_var_4_):
            d_5_i_: int = forall_var_4_
            return not (((0) <= (d_5_i_)) and ((d_5_i_) < ((self.Logits).length(0)))) or ((((self.Logits)[d_5_i_]) <= (_dafny.BigRational('1e9'))) and (((self.Logits)[d_5_i_]) >= (_dafny.BigRational('-1e9'))))

        return (((((((len((self).Tokens)) == (len((self).Ids))) and ((len((self).Ids)) == ((self.Logits).length(0)))) and (((len((self).Ids)) > (0)) and ((((self).Ids)[0]) == (0)))) and (_dafny.quantifier(_dafny.IntegerRange(0, len((self).Ids)), True, lambda0_))) and (_dafny.quantifier(_dafny.IntegerRange(0, len((self).Tokens)), True, lambda1_))) and (_dafny.quantifier(((self).Tokens).UniqueElements, True, lambda3_))) and (_dafny.quantifier(_dafny.IntegerRange(0, (self.Logits).length(0)), True, lambda5_))

    def IdToToken(self, id):
        return ((self).Tokens)[id]

    def TokenToId(self, token):
        return (self).TokenToIdRecursive(token, 0)

    def TokenToIdRecursive(self, token, offset):
        _this = self
        while True:
            with _dafny.label():
                if (((_this).Tokens)[offset]) == (token):
                    return offset
                elif True:
                    in0_ = _this
                    in1_ = token
                    in2_ = (offset) + (1)
                    _this = in0_
                    
                    token = in1_
                    offset = in2_
                    raise _dafny.TailCall()
                break

    def IdToLogit(self, id):
        return (self.Logits)[id]

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
        arr0_ = self.Logits
        arr0_[(d_0_id_)] = _dafny.BigRational('-1e9')

    def MaskTokens(self, tokens):
        d_0_N_: int
        d_0_N_ = len(tokens)
        d_1_i_: int
        d_1_i_ = 0
        while (d_1_i_) < (d_0_N_):
            (self).MaskToken((tokens)[d_1_i_])
            d_1_i_ = (d_1_i_) + (1)

    def MaskTokensExcept(self, tokens):
        d_0_toMask_: _dafny.Seq
        d_0_toMask_ = _dafny.SeqWithoutIsStrInference([])
        d_1_N_: int
        d_1_N_ = len((self).Tokens)
        d_2_i_: int
        d_2_i_ = 0
        while (d_2_i_) < (d_1_N_):
            if (((self).Tokens)[d_2_i_]) not in (tokens):
                d_0_toMask_ = (d_0_toMask_) + (_dafny.SeqWithoutIsStrInference([((self).Tokens)[d_2_i_]]))
            d_2_i_ = (d_2_i_) + (1)
        if (len(d_0_toMask_)) > (0):
            (self).MaskTokens(d_0_toMask_)

    def IsMasked(self, token):
        return ((self.Logits)[(self).TokenToId(token)]) == (_dafny.BigRational('-1e9'))

    def HasUnmaskedToken(self):
        def lambda0_(exists_var_0_):
            d_0_t_: _dafny.Seq = exists_var_0_
            return ((d_0_t_) in ((self).Tokens)) and (not((self).IsMasked(d_0_t_)))

        return _dafny.quantifier(((self).Tokens).UniqueElements, False, lambda0_)

    def BiasToken(self, token, delta):
        d_0_token__id_: int
        d_0_token__id_ = (self).TokenToId(token)
        d_1_raw_: _dafny.BigRational
        d_1_raw_ = ((self.Logits)[d_0_token__id_]) + (delta)
        if (d_1_raw_) > (_dafny.BigRational('1e9')):
            d_1_raw_ = _dafny.BigRational('1e9')
        if (d_1_raw_) < (_dafny.BigRational('-1e9')):
            d_1_raw_ = _dafny.BigRational('-1e9')
        arr0_ = self.Logits
        arr0_[(d_0_token__id_)] = d_1_raw_

    def BiasTokens(self, tokens, delta):
        d_0_n_: int
        d_0_n_ = len(tokens)
        d_1_i_: int
        d_1_i_ = 0
        while (d_1_i_) < (d_0_n_):
            (self).BiasToken((tokens)[d_1_i_], delta)
            d_1_i_ = (d_1_i_) + (1)

    def ScaleToken(self, token, factor):
        d_0_token__id_: int
        d_0_token__id_ = (self).TokenToId(token)
        d_1_raw_: _dafny.BigRational
        d_1_raw_ = ((self.Logits)[d_0_token__id_]) * (factor)
        if (d_1_raw_) > (_dafny.BigRational('1e9')):
            d_1_raw_ = _dafny.BigRational('1e9')
        if (d_1_raw_) < (_dafny.BigRational('-1e9')):
            d_1_raw_ = _dafny.BigRational('-1e9')
        arr0_ = self.Logits
        arr0_[(d_0_token__id_)] = d_1_raw_

    def ScaleTokens(self, tokens, factor):
        d_0_n_: int
        d_0_n_ = len(tokens)
        d_1_i_: int
        d_1_i_ = 0
        while (d_1_i_) < (d_0_n_):
            (self).ScaleToken((tokens)[d_1_i_], factor)
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
    def IsDeadPrefix(self, prefix):
        return (not((self).IsCompletePrefix(prefix))) and ((len((self).ValidNextTokens(prefix))) == (0))

    def ValidNextToken(self, prefix, token):
        return (token) in ((self).ValidNextTokens(prefix))

    def ValidContinuationCount(self, prefix):
        return len((self).ValidNextTokens(prefix))


class Delimiter:
    def  __init__(self):
        self._Left: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        self._Right: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        pass

    def __dafnystr__(self) -> str:
        return "VerifiedDecoderAgent.Delimiter"
    def ctor__(self, left, right):
        (self)._Left = left
        (self)._Right = right

    def LastLeftDelimiterIndex(self, prefix):
        if (len(prefix)) == (0):
            return 0
        elif ((prefix)[(len(prefix)) - (1)]) == ((self).Left):
            return (len(prefix)) - (1)
        elif True:
            d_0_lastInRest_ = (self).LastLeftDelimiterIndex(_dafny.SeqWithoutIsStrInference((prefix)[:(len(prefix)) - (1):]))
            if (d_0_lastInRest_) < ((len(prefix)) - (1)):
                return d_0_lastInRest_
            elif True:
                return len(prefix)

    def FirstRightDelimiterIndex(self, content):
        d_0___accumulator_ = 0
        _this = self
        while True:
            with _dafny.label():
                if (len(content)) == (0):
                    return (0) + (d_0___accumulator_)
                elif ((content)[0]) == ((_this).Right):
                    return (0) + (d_0___accumulator_)
                elif True:
                    d_0___accumulator_ = (d_0___accumulator_) + (1)
                    in0_ = _this
                    in1_ = _dafny.SeqWithoutIsStrInference((content)[1::])
                    _this = in0_
                    
                    content = in1_
                    raise _dafny.TailCall()
                break

    def GetDelimitedContent(self, prefix):
        d_0_start_ = ((self).LastLeftDelimiterIndex(prefix)) + (1)
        if (d_0_start_) > (len(prefix)):
            return _dafny.SeqWithoutIsStrInference([])
        elif True:
            d_1_afterLeft_ = _dafny.SeqWithoutIsStrInference((prefix)[d_0_start_:len(prefix):])
            d_2_endIdx_ = (self).FirstRightDelimiterIndex(d_1_afterLeft_)
            return _dafny.SeqWithoutIsStrInference((d_1_afterLeft_)[:d_2_endIdx_:])

    def InsideDelimitedWindow(self, prefix):
        d_0_start_ = ((self).LastLeftDelimiterIndex(prefix)) + (1)
        return ((d_0_start_) <= (len(prefix))) and (((self).FirstRightDelimiterIndex(_dafny.SeqWithoutIsStrInference((prefix)[d_0_start_:len(prefix):]))) == (len(_dafny.SeqWithoutIsStrInference((prefix)[d_0_start_:len(prefix):]))))

    @property
    def Left(self):
        return self._Left
    @property
    def Right(self):
        return self._Right

class CSDHelpers:
    def  __init__(self):
        self._lm: LM = None
        self._parser: Parser = None
        pass

    def __dafnystr__(self) -> str:
        return "VerifiedDecoderAgent.CSDHelpers"
    def ctor__(self, lm, parser):
        (self)._lm = lm
        (self)._parser = parser

    def LongestValidSuffix(self, prefix):
        _this = self
        while True:
            with _dafny.label():
                if (len(prefix)) == (0):
                    return _dafny.SeqWithoutIsStrInference([])
                elif True:
                    if ((_this).parser).IsValidPrefix(prefix):
                        return prefix
                    elif True:
                        in0_ = _this
                        in1_ = _dafny.SeqWithoutIsStrInference((prefix)[1::])
                        _this = in0_
                        
                        prefix = in1_
                        raise _dafny.TailCall()
                break

    def CanConstrain(self, prefix):
        return not(((self).parser).IsCompletePrefix((self).LongestValidSuffix(prefix)))

    def IsComplete(self, prefix):
        return ((self).parser).IsCompletePrefix((self).LongestValidSuffix(prefix))

    def IsDead(self, prefix):
        return ((self).parser).IsDeadPrefix((self).LongestValidSuffix(prefix))

    def ValidContinuationCount(self, prefix):
        return ((self).parser).ValidContinuationCount((self).LongestValidSuffix(prefix))

    def ParserDistanceToComplete(self, prefix):
        return ((self).parser).ParserDistanceToComplete((self).LongestValidSuffix(prefix))

    def IsLeftDelimiterToken(self, token):
        return ((token) == (default__.LeftDelimiter)) or ((token) == (default__.SpacedLeftDelimiter))

    def IsRightDelimiterToken(self, token):
        return ((token) == (default__.RightDelimiter)) or ((token) == (default__.SpacedRightDelimiter))

    def EndsWithLeftDelimiter(self, prefix):
        return ((len(prefix)) > (0)) and ((self).IsLeftDelimiterToken((prefix)[(len(prefix)) - (1)]))

    def EndsWithRightDelimiter(self, prefix):
        return ((len(prefix)) > (0)) and ((self).IsRightDelimiterToken((prefix)[(len(prefix)) - (1)]))

    def ContainsLeftDelimiter(self, prefix):
        return (default__.PrefixContains(prefix, default__.LeftDelimiter)) or (default__.PrefixContains(prefix, default__.SpacedLeftDelimiter))

    def ContainsRightDelimiter(self, prefix):
        return (default__.PrefixContains(prefix, default__.RightDelimiter)) or (default__.PrefixContains(prefix, default__.SpacedRightDelimiter))

    def LastTokenBefore(self, generated, target):
        token: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        found: bool = False
        d_0_i_: int
        d_0_i_ = (len(generated)) - (1)
        while (d_0_i_) >= (1):
            if ((generated)[d_0_i_]) == (target):
                token = (generated)[(d_0_i_) - (1)]
                found = True
                return token, found
            d_0_i_ = (d_0_i_) - (1)
        token = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        found = False
        return token, found
        return token, found

    def CountOccurrences(self, generated, target):
        d_0___accumulator_ = 0
        _this = self
        while True:
            with _dafny.label():
                if (len(generated)) == (0):
                    return (0) + (d_0___accumulator_)
                elif True:
                    d_0___accumulator_ = (d_0___accumulator_) + ((1 if ((generated)[0]) == (target) else 0))
                    in0_ = _this
                    in1_ = _dafny.SeqWithoutIsStrInference((generated)[1::])
                    in2_ = target
                    _this = in0_
                    
                    generated = in1_
                    target = in2_
                    raise _dafny.TailCall()
                break

    def TokensSinceLastDelimiter(self, generated):
        d_0___accumulator_ = 0
        _this = self
        while True:
            with _dafny.label():
                if (len(generated)) == (0):
                    return (0) + (d_0___accumulator_)
                elif True:
                    if (((((generated)[(len(generated)) - (1)]) == (default__.LeftDelimiter)) or (((generated)[(len(generated)) - (1)]) == (default__.RightDelimiter))) or (((generated)[(len(generated)) - (1)]) == (default__.SpacedLeftDelimiter))) or (((generated)[(len(generated)) - (1)]) == (default__.SpacedRightDelimiter)):
                        return (0) + (d_0___accumulator_)
                    elif True:
                        d_0___accumulator_ = (d_0___accumulator_) + (1)
                        in0_ = _this
                        in1_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (1):])
                        _this = in0_
                        
                        generated = in1_
                        raise _dafny.TailCall()
                break

    def UnconstrainedStep(self, prompt, generated, stepsLeft):
        nextToken: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        remainingSteps: int = int(0)
        ((self).lm).GenerateLogits((prompt) + (generated))
        if (len(((self).lm).Tokens)) > (4):
            (self).MaskAllDelimiters(generated)
        d_0_next__token_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = ((self).lm).ChooseNextToken()
        d_0_next__token_ = out0_
        nextToken = d_0_next__token_
        remainingSteps = (stepsLeft) - (1)
        return nextToken, remainingSteps
        return nextToken, remainingSteps

    def UnconstrainedAllowLeftDelimiterStep(self, prompt, generated, stepsLeft):
        nextToken: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        remainingSteps: int = int(0)
        ((self).lm).GenerateLogits((prompt) + (generated))
        if (len(((self).lm).Tokens)) > (4):
            (self).MaskRightDelimiters(generated)
        d_0_next__token_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = ((self).lm).ChooseNextToken()
        d_0_next__token_ = out0_
        nextToken = d_0_next__token_
        remainingSteps = (stepsLeft) - (1)
        return nextToken, remainingSteps
        return nextToken, remainingSteps

    def UnconstrainedNudgeLeftDelimiterStep(self, prompt, generated, stepsLeft):
        nextToken: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        remainingSteps: int = int(0)
        ((self).lm).GenerateLogits((prompt) + (generated))
        if (len(((self).lm).Tokens)) > (4):
            (self).MaskRightDelimiters(generated)
            (self).BiasLeftDelimiters(_dafny.BigRational('5e0'))
        d_0_next__token_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = ((self).lm).ChooseNextToken()
        d_0_next__token_ = out0_
        nextToken = d_0_next__token_
        remainingSteps = (stepsLeft) - (1)
        return nextToken, remainingSteps
        return nextToken, remainingSteps

    def ConstrainedStep(self, prompt, generated, stepsLeft):
        nextToken: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        remainingSteps: int = int(0)
        d_0_suffix_: _dafny.Seq
        d_0_suffix_ = (self).LongestValidSuffix(generated)
        ((self).lm).GenerateLogits((prompt) + (generated))
        ((self).lm).MaskTokensExcept(((self).parser).ValidNextTokens(d_0_suffix_))
        d_1_next__token_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = ((self).lm).ChooseNextToken()
        d_1_next__token_ = out0_
        nextToken = d_1_next__token_
        remainingSteps = (stepsLeft) - (1)
        return nextToken, remainingSteps
        return nextToken, remainingSteps

    def ConstrainedOrRightDelimiterStep(self, prompt, generated, stepsLeft):
        nextToken: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        remainingSteps: int = int(0)
        d_0_suffix_: _dafny.Seq
        d_0_suffix_ = (self).LongestValidSuffix(generated)
        ((self).lm).GenerateLogits((prompt) + (generated))
        d_1_valid__tokens_: _dafny.Seq
        d_1_valid__tokens_ = ((self).parser).ValidNextTokens(d_0_suffix_)
        if ((self).parser).IsCompletePrefix(d_0_suffix_):
            if (default__.SpacedRightDelimiter) in (((self).lm).Tokens):
                ((self).lm).MaskTokensExcept((d_1_valid__tokens_) + (_dafny.SeqWithoutIsStrInference([default__.RightDelimiter, default__.SpacedRightDelimiter])))
            elif True:
                ((self).lm).MaskTokensExcept((d_1_valid__tokens_) + (_dafny.SeqWithoutIsStrInference([default__.RightDelimiter])))
        elif True:
            ((self).lm).MaskTokensExcept(d_1_valid__tokens_)
        d_2_next__token_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = ((self).lm).ChooseNextToken()
        d_2_next__token_ = out0_
        nextToken = d_2_next__token_
        remainingSteps = (stepsLeft) - (1)
        return nextToken, remainingSteps
        return nextToken, remainingSteps

    def ForcedTokenStep(self, prompt, generated, token, stepsLeft):
        nextToken: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        remainingSteps: int = int(0)
        nextToken = token
        remainingSteps = (stepsLeft) - (1)
        return nextToken, remainingSteps
        return nextToken, remainingSteps

    def BiasForCompletion(self, prefix, bonus):
        d_0_suffix_: _dafny.Seq
        d_0_suffix_ = (self).LongestValidSuffix(prefix)
        if ((self).parser).IsCompletePrefix(d_0_suffix_):
            return
        d_1_valid__next_: _dafny.Seq
        d_1_valid__next_ = ((self).parser).ValidNextTokens(d_0_suffix_)
        d_2_n_: int
        d_2_n_ = len(d_1_valid__next_)
        d_3_i_: int
        d_3_i_ = 0
        while (d_3_i_) < (d_2_n_):
            if ((self).parser).IsCompletePrefix((d_0_suffix_) + (_dafny.SeqWithoutIsStrInference([(d_1_valid__next_)[d_3_i_]]))):
                ((self).lm).BiasToken((d_1_valid__next_)[d_3_i_], bonus)
            d_3_i_ = (d_3_i_) + (1)

    def MaskAllDelimiters(self, generated):
        if (default__.LeftDelimiter) in (((self).lm).Tokens):
            ((self).lm).MaskToken(default__.LeftDelimiter)
        if (default__.RightDelimiter) in (((self).lm).Tokens):
            ((self).lm).MaskToken(default__.RightDelimiter)
        if (default__.SpacedLeftDelimiter) in (((self).lm).Tokens):
            ((self).lm).MaskToken(default__.SpacedLeftDelimiter)
        if (default__.SpacedRightDelimiter) in (((self).lm).Tokens):
            ((self).lm).MaskToken(default__.SpacedRightDelimiter)

    def MaskRightDelimiters(self, generated):
        if (default__.RightDelimiter) in (((self).lm).Tokens):
            ((self).lm).MaskToken(default__.RightDelimiter)
        if (default__.SpacedRightDelimiter) in (((self).lm).Tokens):
            ((self).lm).MaskToken(default__.SpacedRightDelimiter)

    def BiasLeftDelimiters(self, bias):
        if (default__.LeftDelimiter) in (((self).lm).Tokens):
            ((self).lm).BiasToken(default__.LeftDelimiter, bias)
        if (default__.SpacedLeftDelimiter) in (((self).lm).Tokens):
            ((self).lm).BiasToken(default__.SpacedLeftDelimiter, bias)

    def BiasRightDelimiters(self, bias):
        if (default__.RightDelimiter) in (((self).lm).Tokens):
            ((self).lm).BiasToken(default__.RightDelimiter, bias)
        if (default__.SpacedRightDelimiter) in (((self).lm).Tokens):
            ((self).lm).BiasToken(default__.SpacedRightDelimiter, bias)

    def MaskLeftDelimiters(self, generated):
        if (default__.LeftDelimiter) in (((self).lm).Tokens):
            ((self).lm).MaskToken(default__.LeftDelimiter)
        if (default__.SpacedLeftDelimiter) in (((self).lm).Tokens):
            ((self).lm).MaskToken(default__.SpacedLeftDelimiter)

    def AppendUnconstrainedStep(self, prompt, prefix, stepsLeft):
        updated: _dafny.Seq = _dafny.Seq({})
        remainingSteps: int = int(0)
        d_0_next__token_: _dafny.Seq
        d_1_remaining_: int
        out0_: _dafny.Seq
        out1_: int
        out0_, out1_ = (self).UnconstrainedStep(prompt, prefix, stepsLeft)
        d_0_next__token_ = out0_
        d_1_remaining_ = out1_
        updated = (prefix) + (_dafny.SeqWithoutIsStrInference([d_0_next__token_]))
        remainingSteps = d_1_remaining_
        return updated, remainingSteps
        return updated, remainingSteps

    def AppendUnconstrainedAllowLeftDelimiterStep(self, prompt, prefix, stepsLeft):
        updated: _dafny.Seq = _dafny.Seq({})
        remainingSteps: int = int(0)
        d_0_next__token_: _dafny.Seq
        d_1_remaining_: int
        out0_: _dafny.Seq
        out1_: int
        out0_, out1_ = (self).UnconstrainedAllowLeftDelimiterStep(prompt, prefix, stepsLeft)
        d_0_next__token_ = out0_
        d_1_remaining_ = out1_
        updated = (prefix) + (_dafny.SeqWithoutIsStrInference([d_0_next__token_]))
        remainingSteps = d_1_remaining_
        return updated, remainingSteps
        return updated, remainingSteps

    def AppendUnconstrainedNudgeLeftDelimiterStep(self, prompt, prefix, stepsLeft):
        updated: _dafny.Seq = _dafny.Seq({})
        remainingSteps: int = int(0)
        d_0_next__token_: _dafny.Seq
        d_1_remaining_: int
        out0_: _dafny.Seq
        out1_: int
        out0_, out1_ = (self).UnconstrainedNudgeLeftDelimiterStep(prompt, prefix, stepsLeft)
        d_0_next__token_ = out0_
        d_1_remaining_ = out1_
        updated = (prefix) + (_dafny.SeqWithoutIsStrInference([d_0_next__token_]))
        remainingSteps = d_1_remaining_
        return updated, remainingSteps
        return updated, remainingSteps

    def AppendConstrainedStep(self, prompt, prefix, stepsLeft):
        updated: _dafny.Seq = _dafny.Seq({})
        remainingSteps: int = int(0)
        d_0_next__token_: _dafny.Seq
        d_1_remaining_: int
        out0_: _dafny.Seq
        out1_: int
        out0_, out1_ = (self).ConstrainedStep(prompt, prefix, stepsLeft)
        d_0_next__token_ = out0_
        d_1_remaining_ = out1_
        updated = (prefix) + (_dafny.SeqWithoutIsStrInference([d_0_next__token_]))
        remainingSteps = d_1_remaining_
        return updated, remainingSteps
        return updated, remainingSteps

    def AppendConstrainedOrRightDelimiterStep(self, prompt, prefix, stepsLeft):
        updated: _dafny.Seq = _dafny.Seq({})
        remainingSteps: int = int(0)
        d_0_next__token_: _dafny.Seq
        d_1_remaining_: int
        out0_: _dafny.Seq
        out1_: int
        out0_, out1_ = (self).ConstrainedOrRightDelimiterStep(prompt, prefix, stepsLeft)
        d_0_next__token_ = out0_
        d_1_remaining_ = out1_
        updated = (prefix) + (_dafny.SeqWithoutIsStrInference([d_0_next__token_]))
        remainingSteps = d_1_remaining_
        return updated, remainingSteps
        return updated, remainingSteps

    def AppendForcedToken(self, prefix, token, stepsLeft):
        updated: _dafny.Seq = _dafny.Seq({})
        remainingSteps: int = int(0)
        d_0_next__token_: _dafny.Seq
        d_1_remaining__steps_: int
        out0_: _dafny.Seq
        out1_: int
        out0_, out1_ = (self).ForcedTokenStep(_dafny.SeqWithoutIsStrInference([]), prefix, token, stepsLeft)
        d_0_next__token_ = out0_
        d_1_remaining__steps_ = out1_
        updated = (prefix) + (_dafny.SeqWithoutIsStrInference([d_0_next__token_]))
        remainingSteps = d_1_remaining__steps_
        return updated, remainingSteps
        return updated, remainingSteps

    def AppendLeftDelimiter(self, prefix, stepsLeft):
        updated: _dafny.Seq = _dafny.Seq({})
        remainingSteps: int = int(0)
        out0_: _dafny.Seq
        out1_: int
        out0_, out1_ = (self).AppendForcedToken(prefix, default__.LeftDelimiter, stepsLeft)
        updated = out0_
        remainingSteps = out1_
        return updated, remainingSteps

    def AppendRightDelimiter(self, prefix, stepsLeft):
        updated: _dafny.Seq = _dafny.Seq({})
        remainingSteps: int = int(0)
        out0_: _dafny.Seq
        out1_: int
        out0_, out1_ = (self).AppendForcedToken(prefix, default__.RightDelimiter, stepsLeft)
        updated = out0_
        remainingSteps = out1_
        return updated, remainingSteps

    def ValidTokenCount(self, prefix):
        return (self).ValidContinuationCount(prefix)

    def OpenConstrainedSpan(self, prefix, stepsLeft):
        updated: _dafny.Seq = _dafny.Seq({})
        insideSpan: bool = False
        currentConstrained: _dafny.Seq = _dafny.Seq({})
        remainingSteps: int = int(0)
        out0_: _dafny.Seq
        out1_: int
        out0_, out1_ = (self).AppendLeftDelimiter(prefix, stepsLeft)
        updated = out0_
        remainingSteps = out1_
        updated = updated
        insideSpan = True
        currentConstrained = _dafny.SeqWithoutIsStrInference([])
        remainingSteps = remainingSteps
        return updated, insideSpan, currentConstrained, remainingSteps
        return updated, insideSpan, currentConstrained, remainingSteps

    def CloseConstrainedSpan(self, prefix, currentConstrained, stepsLeft):
        updated: _dafny.Seq = _dafny.Seq({})
        insideSpan: bool = False
        updatedConstrained: _dafny.Seq = _dafny.Seq({})
        remainingSteps: int = int(0)
        out0_: _dafny.Seq
        out1_: int
        out0_, out1_ = (self).AppendRightDelimiter(prefix, stepsLeft)
        updated = out0_
        remainingSteps = out1_
        updated = updated
        insideSpan = False
        updatedConstrained = _dafny.SeqWithoutIsStrInference([])
        remainingSteps = remainingSteps
        return updated, insideSpan, updatedConstrained, remainingSteps
        return updated, insideSpan, updatedConstrained, remainingSteps

    def AppendConstrainedToken(self, prefix, currentConstrained, token):
        updated: _dafny.Seq = _dafny.Seq({})
        insideSpan: bool = False
        updatedConstrained: _dafny.Seq = _dafny.Seq({})
        updated = (prefix) + (_dafny.SeqWithoutIsStrInference([token]))
        d_0_updated__constrained_: _dafny.Seq
        d_0_updated__constrained_ = (currentConstrained) + (_dafny.SeqWithoutIsStrInference([token]))
        updated = updated
        insideSpan = True
        updatedConstrained = d_0_updated__constrained_
        return updated, insideSpan, updatedConstrained
        return updated, insideSpan, updatedConstrained

    def AdaptiveConstrainedStep(self, prompt, stablePrefix, currentConstrained, validTokenGroups, bonus, narrowThreshold, eosToken, stepsLeft):
        nextToken: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        remainingSteps: int = int(0)
        ((self).lm).GenerateLogits(((prompt) + (stablePrefix)) + (currentConstrained))
        d_0_valid__tokens_: _dafny.Seq
        d_0_valid__tokens_ = ((self).parser).ValidNextTokens(currentConstrained)
        ((self).lm).MaskTokensExcept(d_0_valid__tokens_)
        if (((self).parser).ValidContinuationCount(currentConstrained)) > (narrowThreshold):
            d_1_group__index_: int
            d_1_group__index_ = 0
            while (d_1_group__index_) < (len(validTokenGroups)):
                d_2_group_: _dafny.Seq
                d_2_group_ = (validTokenGroups)[d_1_group__index_]
                d_3_token__index_: int
                d_3_token__index_ = 0
                while (d_3_token__index_) < (len(d_2_group_)):
                    d_4_token_: _dafny.Seq
                    d_4_token_ = (d_2_group_)[d_3_token__index_]
                    if (d_4_token_) in (d_0_valid__tokens_):
                        ((self).lm).BiasToken(d_4_token_, bonus)
                    d_3_token__index_ = (d_3_token__index_) + (1)
                d_1_group__index_ = (d_1_group__index_) + (1)
        d_5_next__token_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = ((self).lm).ChooseNextToken()
        d_5_next__token_ = out0_
        if (d_5_next__token_) == (eosToken):
            nextToken = eosToken
            remainingSteps = (stepsLeft) - (1)
            return nextToken, remainingSteps
        nextToken = d_5_next__token_
        remainingSteps = (stepsLeft) - (1)
        return nextToken, remainingSteps
        return nextToken, remainingSteps

    def GroupBoostedConstrainedStep(self, prompt, stablePrefix, currentConstrained, validTokenGroups, bonus, stepsLeft):
        nextToken: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        remainingSteps: int = int(0)
        ((self).lm).GenerateLogits(((prompt) + (stablePrefix)) + (currentConstrained))
        ((self).lm).MaskTokensExcept(((self).parser).ValidNextTokens(currentConstrained))
        d_0_next__token_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = ((self).lm).ChooseNextToken()
        d_0_next__token_ = out0_
        nextToken = d_0_next__token_
        remainingSteps = (stepsLeft) - (1)
        return nextToken, remainingSteps
        return nextToken, remainingSteps

    def PenalizedConstrainedStep(self, prompt, stablePrefix, currentConstrained, penaltyTokens, penalty, stepsLeft):
        nextToken: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        remainingSteps: int = int(0)
        ((self).lm).GenerateLogits(((prompt) + (stablePrefix)) + (currentConstrained))
        ((self).lm).MaskTokensExcept(((self).parser).ValidNextTokens(currentConstrained))
        d_0_next__token_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = ((self).lm).ChooseNextToken()
        d_0_next__token_ = out0_
        nextToken = d_0_next__token_
        remainingSteps = (stepsLeft) - (1)
        return nextToken, remainingSteps
        return nextToken, remainingSteps

    def Checkpoint(self, prefix):
        return prefix

    def RestoreCheckpoint(self, checkpoint):
        return checkpoint

    def RestoreIfDead(self, prefix, checkpoint):
        if (self).IsDead(prefix):
            return checkpoint
        elif True:
            return prefix

    def HasBudget(self, stepsLeft, needed):
        return (stepsLeft) >= (needed)

    def MinStepsToComplete(self, prefix):
        return (self).ParserDistanceToComplete(prefix)

    @property
    def lm(self):
        return self._lm
    @property
    def parser(self):
        return self._parser
