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
            return not (((0) <= (d_5_i_)) and ((d_5_i_) < ((self.Logits).length(0)))) or (((_dafny.BigRational('-1e9')) <= ((self.Logits)[d_5_i_])) and (((self.Logits)[d_5_i_]) <= (_dafny.BigRational('1e9'))))

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
            if not((((self).Tokens)[d_2_i_]) in (tokens)):
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
        return (not((self).IsCompletePrefix(prefix))) and (((self).ValidNextTokenCount(prefix)) == (0))


class CSDHelpers:
    def  __init__(self):
        self.cost: int = int(0)
        pass

    def __dafnystr__(self) -> str:
        return "VerifiedDecoderAgent.CSDHelpers"
    def ctor__(self):
        (self).cost = 0

    def UnconstrainedStep(self, lm, prompt, generated):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        (lm).GenerateLogits((prompt) + (generated))
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextTokenUnconstrained()
        next = out0_
        (self).cost = (self.cost) + (1)
        return next

    def UnconstrainedChunk(self, lm, prompt, generated, maxChunkTokens, openSpanToken, eosToken):
        generatedOut: _dafny.Seq = _dafny.Seq({})
        stoppedOnOpenSpan: bool = False
        stoppedOnEos: bool = False
        stepsUsed: int = int(0)
        d_0_chunk_: _dafny.Seq = _dafny.Seq({})
        out0_: _dafny.Seq
        out1_: bool
        out2_: bool
        out3_: int
        out0_, out1_, out2_, out3_ = (lm).GenerateUnconstrainedChunk((prompt) + (generated), maxChunkTokens, openSpanToken, eosToken)
        d_0_chunk_ = out0_
        stoppedOnOpenSpan = out1_
        stoppedOnEos = out2_
        stepsUsed = out3_
        generatedOut = (generated) + (d_0_chunk_)
        (self).cost = (self.cost) + (stepsUsed)
        return generatedOut, stoppedOnOpenSpan, stoppedOnEos, stepsUsed

    def ConstrainedSymbol(self, lm, parser, constrainedPrompt, currentConstrained, maxSymbolTokens, eosToken):
        currentOut: _dafny.Seq = _dafny.Seq({})
        hitEos: bool = False
        stepsUsed: int = int(0)
        d_0_chunk_: _dafny.Seq = _dafny.Seq({})
        d_1_stoppedOnOpen_: bool = False
        d_2_stoppedOnEos_: bool = False
        out0_: _dafny.Seq
        out1_: bool
        out2_: bool
        out3_: int
        out0_, out1_, out2_, out3_ = (lm).GenerateUnconstrainedChunk((constrainedPrompt) + (currentConstrained), maxSymbolTokens, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
        d_0_chunk_ = out0_
        d_1_stoppedOnOpen_ = out1_
        d_2_stoppedOnEos_ = out2_
        stepsUsed = out3_
        (self).cost = (self.cost) + (stepsUsed)
        hitEos = d_2_stoppedOnEos_
        currentOut = currentConstrained
        d_3_i_: int
        d_3_i_ = 0
        with _dafny.label("0"):
            while (d_3_i_) < (len(d_0_chunk_)):
                with _dafny.c_label("0"):
                    d_4_tok_: _dafny.Seq
                    d_4_tok_ = (d_0_chunk_)[d_3_i_]
                    d_5_extended_: _dafny.Seq
                    d_5_extended_ = (currentOut) + (_dafny.SeqWithoutIsStrInference([d_4_tok_]))
                    if (parser).IsValidPrefix(d_5_extended_):
                        currentOut = d_5_extended_
                    elif True:
                        raise _dafny.Break("0")
                    d_3_i_ = (d_3_i_) + (1)
                    pass
            pass
        return currentOut, hitEos, stepsUsed

    def ConstrainedSymbolInGenerated(self, lm, parser, constrainedPrompt, generated, currentConstrained, maxSymbolTokens, eosToken):
        generatedOut: _dafny.Seq = _dafny.Seq({})
        currentOut: _dafny.Seq = _dafny.Seq({})
        hitEos: bool = False
        stepsUsed: int = int(0)
        d_0_stablePrefix_: _dafny.Seq
        d_0_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrained)):])
        out0_: _dafny.Seq
        out1_: bool
        out2_: int
        out0_, out1_, out2_ = (self).ConstrainedSymbol(lm, parser, constrainedPrompt, currentConstrained, maxSymbolTokens, eosToken)
        currentOut = out0_
        hitEos = out1_
        stepsUsed = out2_
        generatedOut = (d_0_stablePrefix_) + (currentOut)
        return generatedOut, currentOut, hitEos, stepsUsed

    def OpenConstrainedSpan(self, lm, generated):
        generatedOut: _dafny.Seq = _dafny.Seq({})
        insideOut: bool = False
        currentOut: _dafny.Seq = _dafny.Seq({})
        generatedOut = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
        insideOut = True
        currentOut = _dafny.SeqWithoutIsStrInference([])
        (self).cost = (self.cost) + (1)
        return generatedOut, insideOut, currentOut

    def EnterObservedConstrainedSpan(self, lm, generated):
        generatedOut: _dafny.Seq = _dafny.Seq({})
        insideOut: bool = False
        currentOut: _dafny.Seq = _dafny.Seq({})
        generatedOut = generated
        insideOut = True
        currentOut = _dafny.SeqWithoutIsStrInference([])
        return generatedOut, insideOut, currentOut

    def AppendConstrainedToken(self, lm, parser, generated, currentConstrained, next):
        generatedOut: _dafny.Seq = _dafny.Seq({})
        insideOut: bool = False
        currentOut: _dafny.Seq = _dafny.Seq({})
        generatedOut = (generated) + (_dafny.SeqWithoutIsStrInference([next]))
        insideOut = True
        currentOut = (currentConstrained) + (_dafny.SeqWithoutIsStrInference([next]))
        return generatedOut, insideOut, currentOut

    def CloseConstrainedSpan(self, lm, parser, generated, currentConstrained):
        generatedOut: _dafny.Seq = _dafny.Seq({})
        insideOut: bool = False
        currentOut: _dafny.Seq = _dafny.Seq({})
        if ((len(currentConstrained)) > (0)) and (((currentConstrained)[(len(currentConstrained)) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
            generatedOut = generated
        elif True:
            generatedOut = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
        insideOut = False
        currentOut = _dafny.SeqWithoutIsStrInference([])
        (self).cost = (self.cost) + (1)
        return generatedOut, insideOut, currentOut

    def ConstrainedStep(self, lm, parser, prompt, generated, eosToken):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        (lm).GenerateLogits((prompt) + (generated))
        (lm).MaskValidNextAndEos(parser, generated, eosToken)
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextToken()
        next = out0_
        (self).cost = (self.cost) + (1)
        return next

    def GroupHasValidMember(self, parser, prefix, group):
        anyValid: bool = False
        anyValid = False
        d_0_i_: int
        d_0_i_ = 0
        while (d_0_i_) < (len(group)):
            if (parser).ValidNextToken(prefix, (group)[d_0_i_]):
                anyValid = True
            d_0_i_ = (d_0_i_) + (1)
        return anyValid

    def BoostValidGroups(self, lm, parser, prefix, groups, amount):
        d_0_i_: int
        d_0_i_ = 0
        while (d_0_i_) < (len(groups)):
            d_1_anyValid_: bool
            out0_: bool
            out0_ = (self).GroupHasValidMember(parser, prefix, (groups)[d_0_i_])
            d_1_anyValid_ = out0_
            if d_1_anyValid_:
                (self).SafeBoostTokenLogits(lm, (groups)[d_0_i_], amount)
            d_0_i_ = (d_0_i_) + (1)

    def GroupBoostedConstrainedStep(self, lm, parser, prompt, constrainedPrefix, groups, boostAmount, eosToken):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        (lm).GenerateLogits((prompt) + (constrainedPrefix))
        if (len(groups)) > (0):
            (self).BoostValidGroups(lm, parser, constrainedPrefix, groups, boostAmount)
        (lm).MaskValidNextAndEos(parser, constrainedPrefix, eosToken)
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextToken()
        next = out0_
        (self).cost = (self.cost) + (1)
        return next

    def AdaptiveConstrainedStep(self, lm, parser, prompt, constrainedPrefix, groups, boostAmount, narrowThreshold, eosToken):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        (lm).GenerateLogits((prompt) + (constrainedPrefix))
        if (len(groups)) > (0):
            d_0_validCount_: int
            d_0_validCount_ = (parser).ValidNextTokenCount(constrainedPrefix)
            if (d_0_validCount_) <= (narrowThreshold):
                (self).BoostValidGroups(lm, parser, constrainedPrefix, groups, boostAmount)
        (lm).MaskValidNextAndEos(parser, constrainedPrefix, eosToken)
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextToken()
        next = out0_
        (self).cost = (self.cost) + (1)
        return next

    def AdaptiveConstrainedStepWithPenalties(self, lm, parser, prompt, constrainedPrefix, boostGroups, boostAmount, penaltyTokens, penaltyAmount, narrowThreshold, eosToken):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        (lm).GenerateLogits((prompt) + (constrainedPrefix))
        if (len(boostGroups)) > (0):
            d_0_validCount_: int
            d_0_validCount_ = (parser).ValidNextTokenCount(constrainedPrefix)
            if (d_0_validCount_) <= (narrowThreshold):
                (self).BoostValidGroups(lm, parser, constrainedPrefix, boostGroups, boostAmount)
        (self).SafePenalizeTokenLogits(lm, penaltyTokens, penaltyAmount)
        (lm).MaskValidNextAndEos(parser, constrainedPrefix, eosToken)
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextToken()
        next = out0_
        (self).cost = (self.cost) + (1)
        return next

    def PenalizedConstrainedStep(self, lm, parser, prompt, constrainedPrefix, tokensToPenalize, penaltyAmount, eosToken):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        (lm).GenerateLogits((prompt) + (constrainedPrefix))
        (self).PenalizeTokenLogits(lm, tokensToPenalize, penaltyAmount)
        (lm).MaskValidNextAndEos(parser, constrainedPrefix, eosToken)
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextToken()
        next = out0_
        (self).cost = (self.cost) + (1)
        return next

    def BoostedConstrainedStep(self, lm, parser, prompt, constrainedPrefix, tokensToBoost, boostAmount, eosToken):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        (lm).GenerateLogits((prompt) + (constrainedPrefix))
        (self).BoostTokenLogits(lm, tokensToBoost, boostAmount)
        (lm).MaskValidNextAndEos(parser, constrainedPrefix, eosToken)
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextToken()
        next = out0_
        (self).cost = (self.cost) + (1)
        return next

    def SafeBoostedConstrainedStep(self, lm, parser, prompt, constrainedPrefix, tokensToBoost, boostAmount, eosToken):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        (lm).GenerateLogits((prompt) + (constrainedPrefix))
        (self).SafeBoostTokenLogits(lm, tokensToBoost, boostAmount)
        (lm).MaskValidNextAndEos(parser, constrainedPrefix, eosToken)
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextToken()
        next = out0_
        (self).cost = (self.cost) + (1)
        return next

    def SafePenalizedConstrainedStep(self, lm, parser, prompt, constrainedPrefix, tokensToPenalize, penaltyAmount, eosToken):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        (lm).GenerateLogits((prompt) + (constrainedPrefix))
        (self).SafePenalizeTokenLogits(lm, tokensToPenalize, penaltyAmount)
        (lm).MaskValidNextAndEos(parser, constrainedPrefix, eosToken)
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextToken()
        next = out0_
        (self).cost = (self.cost) + (1)
        return next

    @staticmethod
    def ExtractAfterKeyword(prefix, keyword):
        following: _dafny.Seq = _dafny.Seq({})
        following = _dafny.SeqWithoutIsStrInference([])
        d_0_i_: int
        d_0_i_ = 0
        while (d_0_i_) < (len(prefix)):
            if (((prefix)[d_0_i_]) == (keyword)) and (((d_0_i_) + (1)) < (len(prefix))):
                following = (following) + (_dafny.SeqWithoutIsStrInference([(prefix)[(d_0_i_) + (1)]]))
            d_0_i_ = (d_0_i_) + (1)
        return following

    @staticmethod
    def IntersectTokenSets(a, b):
        result: _dafny.Seq = _dafny.Seq({})
        result = _dafny.SeqWithoutIsStrInference([])
        d_0_i_: int
        d_0_i_ = 0
        while (d_0_i_) < (len(a)):
            if ((a)[d_0_i_]) in (b):
                result = (result) + (_dafny.SeqWithoutIsStrInference([(a)[d_0_i_]]))
            d_0_i_ = (d_0_i_) + (1)
        return result

    @staticmethod
    def SubtractTokenSets(a, b):
        result: _dafny.Seq = _dafny.Seq({})
        result = _dafny.SeqWithoutIsStrInference([])
        d_0_i_: int
        d_0_i_ = 0
        while (d_0_i_) < (len(a)):
            if ((a)[d_0_i_]) not in (b):
                result = (result) + (_dafny.SeqWithoutIsStrInference([(a)[d_0_i_]]))
            d_0_i_ = (d_0_i_) + (1)
        return result

    @staticmethod
    def RollbackToValidPrefix(parser, generated):
        repaired: _dafny.Seq = _dafny.Seq({})
        repaired = generated
        while (not((parser).IsValidPrefix(repaired))) or ((parser).IsDeadPrefix(repaired)):
            repaired = _dafny.SeqWithoutIsStrInference((repaired)[:(len(repaired)) - (1):])
        return repaired

    def RollbackConstrainedSpan(self, parser, stablePrefix, generated, currentConstrained):
        generatedOut: _dafny.Seq = _dafny.Seq({})
        currentOut: _dafny.Seq = _dafny.Seq({})
        out0_: _dafny.Seq
        out0_ = CSDHelpers.RollbackToValidPrefix(parser, currentConstrained)
        currentOut = out0_
        generatedOut = (stablePrefix) + (currentOut)
        return generatedOut, currentOut

    def RollbackConstrainedSuffix(self, parser, generated, currentConstrained):
        generatedOut: _dafny.Seq = _dafny.Seq({})
        currentOut: _dafny.Seq = _dafny.Seq({})
        d_0_stablePrefix_: _dafny.Seq
        d_0_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrained)):])
        out0_: _dafny.Seq
        out0_ = CSDHelpers.RollbackToValidPrefix(parser, currentConstrained)
        currentOut = out0_
        generatedOut = (d_0_stablePrefix_) + (currentOut)
        return generatedOut, currentOut

    @staticmethod
    def FlattenTokenGroups(groups):
        flat: _dafny.Seq = _dafny.Seq({})
        flat = _dafny.SeqWithoutIsStrInference([])
        d_0_i_: int
        d_0_i_ = 0
        while (d_0_i_) < (len(groups)):
            flat = (flat) + ((groups)[d_0_i_])
            d_0_i_ = (d_0_i_) + (1)
        return flat

    @staticmethod
    def GroupContaining(groups, tok):
        idx: int = int(0)
        idx = -1
        d_0_i_: int
        d_0_i_ = 0
        while (d_0_i_) < (len(groups)):
            if (tok) in ((groups)[d_0_i_]):
                idx = d_0_i_
                return idx
            d_0_i_ = (d_0_i_) + (1)
        return idx

    def LastTokenBefore(self, s, sep):
        tok: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        found: bool = False
        d_0_idx_: int
        d_0_idx_ = len(s)
        while ((d_0_idx_) > (0)) and (((s)[(d_0_idx_) - (1)]) != (sep)):
            d_0_idx_ = (d_0_idx_) - (1)
        if (d_0_idx_) >= (2):
            found = True
            tok = (s)[(d_0_idx_) - (2)]
        elif True:
            found = False
            tok = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        return tok, found

    @staticmethod
    def PrefixToString(p):
        d_0___accumulator_ = _dafny.SeqWithoutIsStrInference([])
        while True:
            with _dafny.label():
                if (len(p)) == (0):
                    return (d_0___accumulator_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "")))
                elif True:
                    d_0___accumulator_ = (d_0___accumulator_) + ((p)[0])
                    in0_ = _dafny.SeqWithoutIsStrInference((p)[1::])
                    p = in0_
                    raise _dafny.TailCall()
                break

    @staticmethod
    def ExtractContentBetweenDelimiters(input, startDelim, endDelim):
        return CSDHelpers.ExtractContentExtern(input, startDelim, endDelim)

    def BoostTokenLogits(self, lm, tokens, amount):
        d_0_i_: int
        d_0_i_ = 0
        while (d_0_i_) < (len(tokens)):
            d_1_id_: int
            d_1_id_ = (lm).TokenToId((tokens)[d_0_i_])
            d_2_newVal_: _dafny.BigRational
            d_2_newVal_ = ((lm.Logits)[d_1_id_]) + (amount)
            if (d_2_newVal_) > (_dafny.BigRational('1e9')):
                d_2_newVal_ = _dafny.BigRational('1e9')
            arr0_ = lm.Logits
            arr0_[(d_1_id_)] = d_2_newVal_
            d_0_i_ = (d_0_i_) + (1)

    def SafeBoostTokenLogits(self, lm, tokens, amount):
        d_0_validTokens_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = CSDHelpers.IntersectTokenSets((lm).Tokens, tokens)
        d_0_validTokens_ = out0_
        (self).BoostTokenLogits(lm, d_0_validTokens_, amount)

    def PenalizeTokenLogits(self, lm, tokens, amount):
        d_0_i_: int
        d_0_i_ = 0
        while (d_0_i_) < (len(tokens)):
            d_1_id_: int
            d_1_id_ = (lm).TokenToId((tokens)[d_0_i_])
            d_2_newVal_: _dafny.BigRational
            d_2_newVal_ = ((lm.Logits)[d_1_id_]) - (amount)
            if (d_2_newVal_) < (_dafny.BigRational('-1e9')):
                d_2_newVal_ = _dafny.BigRational('-1e9')
            arr0_ = lm.Logits
            arr0_[(d_1_id_)] = d_2_newVal_
            d_0_i_ = (d_0_i_) + (1)

    def SafePenalizeTokenLogits(self, lm, tokens, amount):
        d_0_validTokens_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = CSDHelpers.IntersectTokenSets((lm).Tokens, tokens)
        d_0_validTokens_ = out0_
        (self).PenalizeTokenLogits(lm, d_0_validTokens_, amount)

    def MaskTokensInPrefix(self, lm, prefix):
        d_0_i_: int
        d_0_i_ = 0
        while (d_0_i_) < (len(prefix)):
            if ((prefix)[d_0_i_]) in ((lm).Tokens):
                (lm).MaskToken((prefix)[d_0_i_])
            d_0_i_ = (d_0_i_) + (1)

    def GetHighestLogitToken(self, lm):
        token: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_0_bestIdx_: int
        d_0_bestIdx_ = 0
        d_1_i_: int
        d_1_i_ = 1
        while (d_1_i_) < (len((lm).Tokens)):
            if ((lm.Logits)[d_1_i_]) > ((lm.Logits)[d_0_bestIdx_]):
                d_0_bestIdx_ = d_1_i_
            d_1_i_ = (d_1_i_) + (1)
        token = (lm).IdToToken(d_0_bestIdx_)
        return token

    def GetLogitGap(self, lm):
        gap: _dafny.BigRational = _dafny.BigRational()
        d_0_top1_: _dafny.BigRational
        d_0_top1_ = _dafny.BigRational('-1000000001e0')
        d_1_top2_: _dafny.BigRational
        d_1_top2_ = _dafny.BigRational('-1000000001e0')
        d_2_i_: int
        d_2_i_ = 0
        while (d_2_i_) < (len((lm).Tokens)):
            if not((lm).IsMasked(((lm).Tokens)[d_2_i_])):
                d_3_L_: _dafny.BigRational
                d_3_L_ = (lm.Logits)[d_2_i_]
                if (d_3_L_) > (d_0_top1_):
                    d_1_top2_ = d_0_top1_
                    d_0_top1_ = d_3_L_
                elif (d_3_L_) > (d_1_top2_):
                    d_1_top2_ = d_3_L_
            d_2_i_ = (d_2_i_) + (1)
        if (d_1_top2_) < (_dafny.BigRational('-1e9')):
            gap = _dafny.BigRational('0e0')
        elif True:
            gap = (d_0_top1_) - (d_1_top2_)
        return gap

    def GetTopKTokens(self, lm, k):
        tokens: _dafny.Seq = _dafny.Seq({})
        tokens = _dafny.SeqWithoutIsStrInference([])
        d_0_chosenIdx_: _dafny.Seq
        d_0_chosenIdx_ = _dafny.SeqWithoutIsStrInference([])
        d_1_picked_: int
        d_1_picked_ = 0
        while (d_1_picked_) < (k):
            d_2_seed_: int
            d_2_seed_ = 0
            while ((d_2_seed_) < (len((lm).Tokens))) and ((d_2_seed_) in (d_0_chosenIdx_)):
                d_2_seed_ = (d_2_seed_) + (1)
            d_3_bestIdx_: int
            d_3_bestIdx_ = d_2_seed_
            d_4_j_: int
            d_4_j_ = (d_2_seed_) + (1)
            while (d_4_j_) < (len((lm).Tokens)):
                if not((d_4_j_) in (d_0_chosenIdx_)):
                    if (((lm.Logits)[d_4_j_]) > ((lm.Logits)[d_3_bestIdx_])) or ((((lm.Logits)[d_4_j_]) == ((lm.Logits)[d_3_bestIdx_])) and ((d_4_j_) < (d_3_bestIdx_))):
                        d_3_bestIdx_ = d_4_j_
                d_4_j_ = (d_4_j_) + (1)
            d_0_chosenIdx_ = (d_0_chosenIdx_) + (_dafny.SeqWithoutIsStrInference([d_3_bestIdx_]))
            tokens = (tokens) + (_dafny.SeqWithoutIsStrInference([((lm).Tokens)[d_3_bestIdx_]]))
            d_1_picked_ = (d_1_picked_) + (1)
        return tokens

    def DeadEndDetection(self, parser, prefix, minValidCount):
        isNarrow: bool = False
        d_0_validCount_: int
        d_0_validCount_ = (parser).ValidNextTokenCount(prefix)
        isNarrow = (d_0_validCount_) < (minValidCount)
        return isNarrow

    def SoftConstrainedStep(self, lm, parser, prompt, constrainedPrefix, boostAmount, eosToken):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        isValid: bool = False
        (lm).GenerateLogits((prompt) + (constrainedPrefix))
        (lm).BoostValidNextAndEos(parser, constrainedPrefix, boostAmount, eosToken)
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextTokenUnconstrained()
        next = out0_
        (self).cost = (self.cost) + (1)
        isValid = ((next) == (eosToken)) or ((parser).IsValidPrefix((constrainedPrefix) + (_dafny.SeqWithoutIsStrInference([next]))))
        return next, isValid

    def SafeSoftConstrainedStep(self, lm, parser, prompt, constrainedPrefix, boostAmount, eosToken):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        usedFallback: bool = False
        (lm).GenerateLogits((prompt) + (constrainedPrefix))
        (lm).BoostValidNextAndEos(parser, constrainedPrefix, boostAmount, eosToken)
        d_0_softNext_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextTokenUnconstrained()
        d_0_softNext_ = out0_
        if ((d_0_softNext_) == (eosToken)) or ((parser).IsValidPrefix((constrainedPrefix) + (_dafny.SeqWithoutIsStrInference([d_0_softNext_])))):
            next = d_0_softNext_
            usedFallback = False
        elif True:
            (lm).MaskValidNextAndEos(parser, constrainedPrefix, eosToken)
            out1_: _dafny.Seq
            out1_ = (lm).ChooseNextToken()
            next = out1_
            usedFallback = True
        (self).cost = (self.cost) + (1)
        return next, usedFallback

    def ConfidenceGatedStep(self, lm, parser, prompt, constrainedPrefix, eosToken):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        wasConstrained: bool = False
        (lm).GenerateLogits((prompt) + (constrainedPrefix))
        d_0_topToken_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = (self).GetHighestLogitToken(lm)
        d_0_topToken_ = out0_
        if (d_0_topToken_) == (eosToken):
            next = d_0_topToken_
            wasConstrained = False
        elif (parser).IsValidPrefix((constrainedPrefix) + (_dafny.SeqWithoutIsStrInference([d_0_topToken_]))):
            next = d_0_topToken_
            wasConstrained = False
        elif True:
            (lm).MaskValidNextAndEos(parser, constrainedPrefix, eosToken)
            out1_: _dafny.Seq
            out1_ = (lm).ChooseNextToken()
            next = out1_
            wasConstrained = True
        (self).cost = (self.cost) + (1)
        return next, wasConstrained

    @staticmethod
    def CountSubstring(s, sub):
        d_0___accumulator_ = 0
        while True:
            with _dafny.label():
                if (len(s)) < (len(sub)):
                    return (0) + (d_0___accumulator_)
                elif (_dafny.SeqWithoutIsStrInference((s)[:len(sub):])) == (sub):
                    d_0___accumulator_ = (d_0___accumulator_) + (1)
                    in0_ = _dafny.SeqWithoutIsStrInference((s)[len(sub)::])
                    in1_ = sub
                    s = in0_
                    sub = in1_
                    raise _dafny.TailCall()
                elif True:
                    in2_ = _dafny.SeqWithoutIsStrInference((s)[1::])
                    in3_ = sub
                    s = in2_
                    sub = in3_
                    raise _dafny.TailCall()
                break

    @staticmethod
    def OccurrencesInRange(prefix, target, hi):
        d_0___accumulator_ = 0
        while True:
            with _dafny.label():
                if (hi) == (0):
                    return (0) + (d_0___accumulator_)
                elif True:
                    d_0___accumulator_ = ((1 if ((prefix)[(hi) - (1)]) == (target) else 0)) + (d_0___accumulator_)
                    in0_ = prefix
                    in1_ = target
                    in2_ = (hi) - (1)
                    prefix = in0_
                    target = in1_
                    hi = in2_
                    raise _dafny.TailCall()
                break

    @staticmethod
    def CountTokenOccurrences(prefix, target):
        count: int = int(0)
        count = 0
        d_0_i_: int
        d_0_i_ = 0
        while (d_0_i_) < (len(prefix)):
            if ((prefix)[d_0_i_]) == (target):
                count = (count) + (1)
            d_0_i_ = (d_0_i_) + (1)
        return count

    @staticmethod
    def TokensSinceLastOccurrence(prefix, target):
        dist: int = int(0)
        dist = 0
        while ((dist) < (len(prefix))) and (((prefix)[((len(prefix)) - (1)) - (dist)]) != (target)):
            dist = (dist) + (1)
        return dist

    def GetTokenLogit(self, lm, token):
        logit: _dafny.BigRational = _dafny.BigRational()
        logit = (lm.Logits)[(lm).TokenToId(token)]
        return logit

    def ScaleAllLogits(self, lm, scalar):
        d_0_i_: int
        d_0_i_ = 0
        while (d_0_i_) < (len((lm).Tokens)):
            d_1_id_: int
            d_1_id_ = (lm).TokenToId(((lm).Tokens)[d_0_i_])
            d_2_newVal_: _dafny.BigRational
            d_2_newVal_ = ((lm.Logits)[d_1_id_]) * (scalar)
            if (d_2_newVal_) > (_dafny.BigRational('1e9')):
                d_2_newVal_ = _dafny.BigRational('1e9')
            if (d_2_newVal_) < (_dafny.BigRational('-1e9')):
                d_2_newVal_ = _dafny.BigRational('-1e9')
            arr0_ = lm.Logits
            arr0_[(d_1_id_)] = d_2_newVal_
            d_0_i_ = (d_0_i_) + (1)

    def ValidTokenCount(self, parser, prefix):
        count: int = int(0)
        count = (parser).ValidNextTokenCount(prefix)
        return count

    def TopValidCandidates(self, lm, parser, prompt, prefix, maxCandidates, eosToken):
        candidates: _dafny.Seq = _dafny.Seq({})
        d_0_baseCost_: int
        d_0_baseCost_ = self.cost
        (lm).GenerateLogits((prompt) + (prefix))
        d_1_validWithEos_: _dafny.Seq
        d_1_validWithEos_ = ((parser).ValidNextTokens(prefix)) + (_dafny.SeqWithoutIsStrInference([eosToken]))
        d_2_pool_: _dafny.Seq
        d_2_pool_ = _dafny.SeqWithoutIsStrInference([])
        d_3_i_: int
        d_3_i_ = 0
        while (d_3_i_) < (len(d_1_validWithEos_)):
            d_4_tok_: _dafny.Seq
            d_4_tok_ = (d_1_validWithEos_)[d_3_i_]
            if not((d_4_tok_) in (d_2_pool_)):
                d_2_pool_ = (d_2_pool_) + (_dafny.SeqWithoutIsStrInference([d_4_tok_]))
            d_3_i_ = (d_3_i_) + (1)
        if (len(d_2_pool_)) == (0):
            d_2_pool_ = _dafny.SeqWithoutIsStrInference([eosToken])
        d_5_target_: int
        if (maxCandidates) < (len(d_2_pool_)):
            d_5_target_ = maxCandidates
        elif True:
            d_5_target_ = len(d_2_pool_)
        d_6_chosen_: _dafny.Seq
        d_6_chosen_ = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("1"):
            while (len(d_6_chosen_)) < (d_5_target_):
                with _dafny.c_label("1"):
                    d_7_bestTok_: _dafny.Seq
                    d_7_bestTok_ = (d_2_pool_)[0]
                    d_8_bestLogit_: _dafny.BigRational
                    d_8_bestLogit_ = _dafny.BigRational('-1e9')
                    d_9_found_: bool
                    d_9_found_ = False
                    d_10_j_: int
                    d_10_j_ = 0
                    while (d_10_j_) < (len(d_2_pool_)):
                        d_11_tok_: _dafny.Seq
                        d_11_tok_ = (d_2_pool_)[d_10_j_]
                        if not((d_11_tok_) in (d_6_chosen_)):
                            d_12_tokLogit_: _dafny.BigRational
                            d_12_tokLogit_ = (lm.Logits)[(lm).TokenToId(d_11_tok_)]
                            if (not(d_9_found_)) or ((d_12_tokLogit_) > (d_8_bestLogit_)):
                                d_7_bestTok_ = d_11_tok_
                                d_8_bestLogit_ = d_12_tokLogit_
                                d_9_found_ = True
                        d_10_j_ = (d_10_j_) + (1)
                    if d_9_found_:
                        d_6_chosen_ = (d_6_chosen_) + (_dafny.SeqWithoutIsStrInference([d_7_bestTok_]))
                    elif True:
                        raise _dafny.Break("1")
                    pass
            pass
        if (len(d_6_chosen_)) == (0):
            candidates = _dafny.SeqWithoutIsStrInference([(d_2_pool_)[0]])
        elif True:
            candidates = d_6_chosen_
        (self).cost = (self.cost) + (1)
        return candidates

    def IsTokenValidNext(self, parser, prefix, token):
        isValid: bool = False
        isValid = (parser).ValidNextToken(prefix, token)
        return isValid

    def RepetitionPenaltyStep(self, lm, parser, prompt, prefix, generated, penaltyAmount, eosToken):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        (lm).GenerateLogits((prompt) + (prefix))
        (self).PenalizeTokenLogits(lm, generated, penaltyAmount)
        (lm).MaskValidNextAndEos(parser, prefix, eosToken)
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextToken()
        next = out0_
        (self).cost = (self.cost) + (1)
        return next

    def SafeRepetitionPenaltyStep(self, lm, parser, prompt, prefix, generated, penaltyAmount, eosToken):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        (lm).GenerateLogits((prompt) + (prefix))
        (self).SafePenalizeTokenLogits(lm, generated, penaltyAmount)
        (lm).MaskValidNextAndEos(parser, prefix, eosToken)
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextToken()
        next = out0_
        (self).cost = (self.cost) + (1)
        return next

    def TemperatureConstrainedStep(self, lm, parser, prompt, prefix, temperature, eosToken):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        (lm).GenerateLogits((prompt) + (prefix))
        d_0_scalar_: _dafny.BigRational
        d_0_scalar_ = (_dafny.BigRational('1e0')) / (temperature)
        if (d_0_scalar_) > (_dafny.BigRational('1e8')):
            d_0_scalar_ = _dafny.BigRational('1e8')
        (self).ScaleAllLogits(lm, d_0_scalar_)
        (lm).MaskValidNextAndEos(parser, prefix, eosToken)
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextToken()
        next = out0_
        (self).cost = (self.cost) + (1)
        return next

    def SafeTemperatureConstrainedStep(self, lm, parser, prompt, prefix, temperature, eosToken):
        next: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        (lm).GenerateLogits((prompt) + (prefix))
        d_0_safeTemperature_: _dafny.BigRational
        d_0_safeTemperature_ = temperature
        if (d_0_safeTemperature_) < (_dafny.BigRational('1e-8')):
            d_0_safeTemperature_ = _dafny.BigRational('1e-8')
        if (d_0_safeTemperature_) > (_dafny.BigRational('1e8')):
            d_0_safeTemperature_ = _dafny.BigRational('1e8')
        d_1_scalar_: _dafny.BigRational
        d_1_scalar_ = (_dafny.BigRational('1e0')) / (d_0_safeTemperature_)
        if (d_1_scalar_) > (_dafny.BigRational('1e8')):
            d_1_scalar_ = _dafny.BigRational('1e8')
        if (d_1_scalar_) <= (_dafny.BigRational('0e0')):
            d_1_scalar_ = _dafny.BigRational('1e0')
        (self).ScaleAllLogits(lm, d_1_scalar_)
        (lm).MaskValidNextAndEos(parser, prefix, eosToken)
        out0_: _dafny.Seq
        out0_ = (lm).ChooseNextToken()
        next = out0_
        (self).cost = (self.cost) + (1)
        return next

    def SaveLogitsSnapshot(self, lm):
        snapshot: _dafny.Seq = _dafny.Seq({})
        snapshot = _dafny.SeqWithoutIsStrInference((lm.Logits)[0:(lm.Logits).length(0):])
        return snapshot

    def RestoreLogitsSnapshot(self, lm, snapshot):
        d_0_i_: int
        d_0_i_ = 0
        while (d_0_i_) < ((lm.Logits).length(0)):
            arr0_ = lm.Logits
            arr0_[(d_0_i_)] = (snapshot)[d_0_i_]
            d_0_i_ = (d_0_i_) + (1)

    def RolloutConstrainedWithPenalties(self, lm, parser, prompt, startPrefix, totalBudget, penalties, penaltyAmount, eosToken):
        generatedOut: _dafny.Seq = _dafny.Seq({})
        stepsUsed: int = int(0)
        terminatedByEos: bool = False
        generatedOut = startPrefix
        stepsUsed = 0
        terminatedByEos = False
        with _dafny.label("2"):
            while ((stepsUsed) < (totalBudget)) and (not((parser).IsCompletePrefix(generatedOut))):
                with _dafny.c_label("2"):
                    d_0_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (self).SafePenalizedConstrainedStep(lm, parser, prompt, generatedOut, penalties, penaltyAmount, eosToken)
                    d_0_next_ = out0_
                    if (d_0_next_) == (eosToken):
                        terminatedByEos = True
                        stepsUsed = (stepsUsed) + (1)
                        raise _dafny.Break("2")
                    generatedOut = (generatedOut) + (_dafny.SeqWithoutIsStrInference([d_0_next_]))
                    stepsUsed = (stepsUsed) + (1)
                    pass
            pass
        return generatedOut, stepsUsed, terminatedByEos

    def SpeculativeConstrainedRollout(self, lm, parser, prompt, constrainedPrefix, numTokens, eosToken):
        candidateTokens: _dafny.Seq = _dafny.Seq({})
        candidatePrefix: _dafny.Seq = _dafny.Seq({})
        hitComplete: bool = False
        hitEos: bool = False
        stepsUsed: int = int(0)
        d_0_snap_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = (self).SaveLogitsSnapshot(lm)
        d_0_snap_ = out0_
        candidateTokens = _dafny.SeqWithoutIsStrInference([])
        d_1_cur_: _dafny.Seq
        d_1_cur_ = constrainedPrefix
        stepsUsed = 0
        hitEos = False
        while (((stepsUsed) < (numTokens)) and (not((parser).IsCompletePrefix(d_1_cur_)))) and (not(hitEos)):
            d_2_next_: _dafny.Seq
            out1_: _dafny.Seq
            out1_ = (self).ConstrainedStep(lm, parser, prompt, d_1_cur_, eosToken)
            d_2_next_ = out1_
            stepsUsed = (stepsUsed) + (1)
            if (d_2_next_) == (eosToken):
                hitEos = True
            elif True:
                d_1_cur_ = (d_1_cur_) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                candidateTokens = (candidateTokens) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
        (self).RestoreLogitsSnapshot(lm, d_0_snap_)
        candidatePrefix = d_1_cur_
        hitComplete = (parser).IsCompletePrefix(d_1_cur_)
        return candidateTokens, candidatePrefix, hitComplete, hitEos, stepsUsed

