import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import module_ as module_
import _dafny as _dafny
import System_ as System_
import VerifiedDecoderAgent as VerifiedDecoderAgent

# Module: GeneratedCSD

class default__:
    def  __init__(self):
        pass

    @staticmethod
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        insideConstrainedOut: bool = False
        currentConstrainedOut: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatGroups_ = out0_
        d_3_rollbackToken_: _dafny.Seq
        d_3_rollbackToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))
        if not((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))) in (d_2_flatGroups_)):
            d_3_rollbackToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        if not((d_3_rollbackToken_) in (d_2_flatGroups_)):
            d_3_rollbackToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE"))
        if not((d_3_rollbackToken_) in (d_2_flatGroups_)):
            d_3_rollbackToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND"))
        d_4_continuationTokens_: _dafny.Seq
        d_4_continuationTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR"))])
        d_5_longPrefixThreshold_: int
        d_5_longPrefixThreshold_ = 64
        d_6_chunkBudget_: int
        d_6_chunkBudget_ = stepTokenBudget
        if (d_6_chunkBudget_) == (0):
            d_6_chunkBudget_ = 4
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_7_next_ = out1_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_7_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                            if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_8_completeNow_: bool
                        d_8_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_completeNow_:
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out2_
                            d_10_closedInside_ = out3_
                            d_11_closedCurrent_ = out4_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if (len(currentConstrainedOut)) >= (d_5_longPrefixThreshold_):
                                d_12_repaired_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_3_rollbackToken_)
                                d_12_repaired_ = out5_
                                d_13_trimCount_: int
                                d_13_trimCount_ = (len(currentConstrainedOut)) - (len(d_12_repaired_))
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_13_trimCount_):])
                                currentConstrainedOut = d_12_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_14_stablePrefix_: _dafny.Seq
                                d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_15_localChunkBudget_: int
                                d_15_localChunkBudget_ = d_6_chunkBudget_
                                if ((maxSteps) - (d_1_steps_)) < (d_15_localChunkBudget_):
                                    d_15_localChunkBudget_ = (maxSteps) - (d_1_steps_)
                                (lm).GenerateLogits(((prompt) + (d_14_stablePrefix_)) + (currentConstrainedOut))
                                (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'))
                                d_16_completeForBias_: bool
                                d_16_completeForBias_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_16_completeForBias_:
                                    (d_0_helpers_).PenalizeTokenLogits(lm, d_4_continuationTokens_, _dafny.BigRational('8e0'))
                                d_17_currentOut_: _dafny.Seq
                                d_18_hitEos_: bool
                                d_19_stepsUsed_: int
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: int
                                out6_, out7_, out8_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, (prompt) + (d_14_stablePrefix_), currentConstrainedOut, d_15_localChunkBudget_, eosToken)
                                d_17_currentOut_ = out6_
                                d_18_hitEos_ = out7_
                                d_19_stepsUsed_ = out8_
                                d_1_steps_ = (d_1_steps_) + (d_19_stepsUsed_)
                                generated = (d_14_stablePrefix_) + (d_17_currentOut_)
                                currentConstrainedOut = d_17_currentOut_
                                insideConstrainedOut = True
                                if d_18_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

