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
        d_4_continuationTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))])
        d_5_longPrefixThreshold_: int
        d_5_longPrefixThreshold_ = 96
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                            d_6_next_: _dafny.Seq
                            out1_: _dafny.Seq
                            out1_ = (lm).ChooseNextTokenUnconstrained()
                            d_6_next_ = out1_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            if (d_6_next_) == (eosToken):
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_7_openedGenerated_: _dafny.Seq
                                d_8_openedInside_: bool
                                d_9_openedCurrent_: _dafny.Seq
                                out2_: _dafny.Seq
                                out3_: bool
                                out4_: _dafny.Seq
                                out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_7_openedGenerated_ = out2_
                                d_8_openedInside_ = out3_
                                d_9_openedCurrent_ = out4_
                                generated = d_7_openedGenerated_
                                insideConstrainedOut = d_8_openedInside_
                                currentConstrainedOut = d_9_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_next2_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next2_ = out5_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next2_]))
                    elif True:
                        d_11_completeNow_: bool
                        d_11_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_11_completeNow_:
                            d_12_closedGenerated_: _dafny.Seq
                            d_13_closedInside_: bool
                            d_14_closedCurrent_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_12_closedGenerated_ = out6_
                            d_13_closedInside_ = out7_
                            d_14_closedCurrent_ = out8_
                            generated = d_12_closedGenerated_
                            insideConstrainedOut = d_13_closedInside_
                            currentConstrainedOut = d_14_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (len(currentConstrainedOut)) >= (d_5_longPrefixThreshold_):
                            d_15_repaired_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_3_rollbackToken_)
                            d_15_repaired_ = out9_
                            d_16_trimCount_: int
                            d_16_trimCount_ = (len(currentConstrainedOut)) - (len(d_15_repaired_))
                            generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_16_trimCount_):])
                            currentConstrainedOut = d_15_repaired_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_17_stablePrefix_: _dafny.Seq
                            d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            (lm).GenerateLogits(((prompt) + (d_17_stablePrefix_)) + (currentConstrainedOut))
                            (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'))
                            (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('1e0'))
                            if (len(currentConstrainedOut)) > (0):
                                d_18_lastTok_: _dafny.Seq
                                d_18_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_18_lastTok_]), _dafny.BigRational('3e0'))
                            d_19_almostComplete_: bool
                            d_19_almostComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_19_almostComplete_:
                                (d_0_helpers_).PenalizeTokenLogits(lm, d_4_continuationTokens_, _dafny.BigRational('12e0'))
                            d_20_next3_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                            d_20_next3_ = out10_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_20_next3_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_21_appendedGenerated_: _dafny.Seq
                                d_22_appendedInside_: bool
                                d_23_appendedCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next3_)
                                d_21_appendedGenerated_ = out11_
                                d_22_appendedInside_ = out12_
                                d_23_appendedCurrent_ = out13_
                                generated = d_21_appendedGenerated_
                                insideConstrainedOut = d_22_appendedInside_
                                currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

