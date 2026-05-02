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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 6
        d_3_longSpanThreshold_: int
        d_3_longSpanThreshold_ = 12
        d_4_flattenedGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_4_flattenedGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remaining_: int
                        d_5_remaining_ = (maxSteps) - (d_1_steps_)
                        d_6_chunkBudget_: int
                        d_6_chunkBudget_ = d_5_remaining_
                        if (8) < (d_6_chunkBudget_):
                            d_6_chunkBudget_ = 8
                        d_7_chunkedGenerated_: _dafny.Seq
                        d_8_stoppedOnOpenSpan_: bool
                        d_9_stoppedOnEos_: bool
                        d_10_stepsUsed_: int
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_7_chunkedGenerated_ = out1_
                        d_8_stoppedOnOpenSpan_ = out2_
                        d_9_stoppedOnEos_ = out3_
                        d_10_stepsUsed_ = out4_
                        generated = d_7_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                        if d_9_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_8_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_11_isComplete_: bool
                        d_11_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_11_isComplete_:
                            d_12_closedGenerated_: _dafny.Seq
                            d_13_closedInside_: bool
                            d_14_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_12_closedGenerated_ = out5_
                            d_13_closedInside_ = out6_
                            d_14_closedCurrent_ = out7_
                            generated = d_12_closedGenerated_
                            insideConstrainedOut = d_13_closedInside_
                            currentConstrainedOut = d_14_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_15_stablePrefix_: _dafny.Seq
                            d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_16_validCount_: int
                            out8_: int
                            out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_16_validCount_ = out8_
                            if (len(currentConstrainedOut)) >= (d_3_longSpanThreshold_):
                                d_17_penalizeTokens_: _dafny.Seq
                                d_17_penalizeTokens_ = _dafny.SeqWithoutIsStrInference([eosToken])
                                d_18_next_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, (prompt) + (d_15_stablePrefix_), currentConstrainedOut, d_17_penalizeTokens_, _dafny.BigRational('5e0'), eosToken)
                                d_18_next_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_18_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_appendedGenerated1_: _dafny.Seq
                                    d_20_appendedInside1_: bool
                                    d_21_appendedCurrent1_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                    d_19_appendedGenerated1_ = out10_
                                    d_20_appendedInside1_ = out11_
                                    d_21_appendedCurrent1_ = out12_
                                    generated = d_19_appendedGenerated1_
                                    insideConstrainedOut = d_20_appendedInside1_
                                    currentConstrainedOut = d_21_appendedCurrent1_
                            elif (d_16_validCount_) <= (d_2_narrowThreshold_):
                                d_22_constrainedGenerated_: _dafny.Seq
                                d_23_constrainedInside_: bool
                                d_24_constrainedCurrent_: _dafny.Seq
                                d_25_hitEos_: bool
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out16_: bool
                                out13_, out14_, out15_, out16_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_15_stablePrefix_), generated, currentConstrainedOut, eosToken)
                                d_22_constrainedGenerated_ = out13_
                                d_23_constrainedInside_ = out14_
                                d_24_constrainedCurrent_ = out15_
                                d_25_hitEos_ = out16_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_25_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_22_constrainedGenerated_
                                    insideConstrainedOut = d_23_constrainedInside_
                                    currentConstrainedOut = d_24_constrainedCurrent_
                            elif True:
                                d_26_next2_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, (prompt) + (d_15_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_26_next2_ = out17_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_26_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_27_appendedGenerated2_: _dafny.Seq
                                    d_28_appendedInside2_: bool
                                    d_29_appendedCurrent2_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next2_)
                                    d_27_appendedGenerated2_ = out18_
                                    d_28_appendedInside2_ = out19_
                                    d_29_appendedCurrent2_ = out20_
                                    generated = d_27_appendedGenerated2_
                                    insideConstrainedOut = d_28_appendedInside2_
                                    currentConstrainedOut = d_29_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

