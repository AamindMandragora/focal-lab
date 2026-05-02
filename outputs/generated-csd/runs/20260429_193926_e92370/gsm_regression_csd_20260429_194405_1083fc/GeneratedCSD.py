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
        d_2_openTok_: _dafny.Seq
        d_2_openTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_3_boundaryTok_: _dafny.Seq
        d_3_boundaryTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkedGenerated_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, d_2_openTok_, eosToken)
                        d_5_chunkedGenerated_ = out0_
                        d_6_stoppedOnOpenSpan_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_6_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out4_
                            d_10_closedInside_ = out5_
                            d_11_closedCurrent_ = out6_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_12_validCount_ = out7_
                            d_13_narrow_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_13_narrow_ = out8_
                            if (d_13_narrow_) or ((d_12_validCount_) == (0)):
                                d_14_repaired_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_3_boundaryTok_)
                                d_14_repaired_ = out9_
                                d_15_stablePrefix_: _dafny.Seq
                                d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                generated = (d_15_stablePrefix_) + (d_14_repaired_)
                                currentConstrainedOut = d_14_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_16_constrainedBase_: _dafny.Seq
                                d_16_constrainedBase_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_17_constrainedPrompt_: _dafny.Seq
                                d_17_constrainedPrompt_ = (prompt) + (d_16_constrainedBase_)
                                (lm).GenerateLogits((d_17_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    d_18_flatAll_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_18_flatAll_ = out10_
                                    if (len(d_18_flatAll_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_18_flatAll_, _dafny.BigRational('2e0'))
                                    d_19_i_: int
                                    d_19_i_ = 0
                                    while (d_19_i_) < (len(validTokenGroups)):
                                        d_20_g_: _dafny.Seq
                                        d_20_g_ = (validTokenGroups)[d_19_i_]
                                        d_21_anyValid_: bool
                                        out11_: bool
                                        out11_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_20_g_)
                                        d_21_anyValid_ = out11_
                                        if d_21_anyValid_:
                                            (d_0_helpers_).BoostTokenLogits(lm, d_20_g_, _dafny.BigRational('8e0'))
                                        elif True:
                                            (d_0_helpers_).PenalizeTokenLogits(lm, d_20_g_, _dafny.BigRational('1e0'))
                                        d_19_i_ = (d_19_i_) + (1)
                                (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('12e-1'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_22_next_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (lm).ChooseNextToken()
                                d_22_next_ = out12_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_22_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_23_appendedGenerated_: _dafny.Seq
                                    d_24_appendedInside_: bool
                                    d_25_appendedCurrent_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                    d_23_appendedGenerated_ = out13_
                                    d_24_appendedInside_ = out14_
                                    d_25_appendedCurrent_ = out15_
                                    generated = d_23_appendedGenerated_
                                    insideConstrainedOut = d_24_appendedInside_
                                    currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

