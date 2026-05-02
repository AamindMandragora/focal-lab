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
        d_2_canOpen_: bool
        d_2_canOpen_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)
        d_3_canClose_: bool
        d_3_canClose_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))) in ((lm).Tokens)
        d_4_continuationTokens_: _dafny.Seq
        d_4_continuationTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "UNION")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTERSECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXCEPT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_canOpen_:
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out0_
                            d_6_openedInside_ = out1_
                            d_7_openedCurrent_ = out2_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (VerifiedDecoderAgent.default__.Contains(d_8_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (d_2_canOpen_):
                                d_9_openedGenerated2_: _dafny.Seq
                                d_10_openedInside2_: bool
                                d_11_openedCurrent2_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_9_openedGenerated2_ = out4_
                                d_10_openedInside2_ = out5_
                                d_11_openedCurrent2_ = out6_
                                generated = d_9_openedGenerated2_
                                insideConstrainedOut = d_10_openedInside2_
                                currentConstrainedOut = d_11_openedCurrent2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                    elif True:
                        d_12_completeNow_: bool
                        d_12_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_12_completeNow_:
                            d_13_stablePrefix_: _dafny.Seq
                            d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            (lm).GenerateLogits(((prompt) + (d_13_stablePrefix_)) + (currentConstrainedOut))
                            (d_0_helpers_).PenalizeTokenLogits(lm, d_4_continuationTokens_, _dafny.BigRational('1e2'))
                            d_14_nextComplete_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                            d_14_nextComplete_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_nextComplete_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_3_canClose_) and ((d_1_steps_) < (maxSteps)):
                                d_15_closedGenerated_: _dafny.Seq
                                d_16_closedInside_: bool
                                d_17_closedCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_15_closedGenerated_ = out8_
                                d_16_closedInside_ = out9_
                                d_17_closedCurrent_ = out10_
                                generated = d_15_closedGenerated_
                                insideConstrainedOut = d_16_closedInside_
                                currentConstrainedOut = d_17_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_18_dead_: bool
                            out11_: bool
                            out11_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 0)
                            d_18_dead_ = out11_
                            if d_18_dead_:
                                d_19_repairedFrom_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))
                                d_19_repairedFrom_ = out12_
                                d_20_repairedWhere_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")))
                                d_20_repairedWhere_ = out13_
                                d_21_repairedComma_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                                d_21_repairedComma_ = out14_
                                d_22_repaired_: _dafny.Seq
                                d_22_repaired_ = d_21_repairedComma_
                                if (len(d_20_repairedWhere_)) < (len(d_22_repaired_)):
                                    d_22_repaired_ = d_20_repairedWhere_
                                if (len(d_19_repairedFrom_)) < (len(d_22_repaired_)):
                                    d_22_repaired_ = d_19_repairedFrom_
                                d_23_stablePrefix2_: _dafny.Seq
                                d_23_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                generated = (d_23_stablePrefix2_) + (d_22_repaired_)
                                insideConstrainedOut = True
                                currentConstrainedOut = d_22_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_24_stablePrefix3_: _dafny.Seq
                                d_24_stablePrefix3_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_25_next2_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, (prompt) + (d_24_stablePrefix3_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('8e0'), eosToken)
                                d_25_next2_ = out15_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_25_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_26_appendedGenerated_: _dafny.Seq
                                    d_27_appendedInside_: bool
                                    d_28_appendedCurrent_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next2_)
                                    d_26_appendedGenerated_ = out16_
                                    d_27_appendedInside_ = out17_
                                    d_28_appendedCurrent_ = out18_
                                    generated = d_26_appendedGenerated_
                                    insideConstrainedOut = d_27_appendedInside_
                                    currentConstrainedOut = d_28_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

