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
        d_2_outsideTokensSinceLastSpan_: int
        d_2_outsideTokensSinceLastSpan_ = 0
        d_3_outsideSchedule_: int
        d_3_outsideSchedule_ = 24
        d_4_innerSpanCap_: int
        d_4_innerSpanCap_ = 16
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_outsideTokensSinceLastSpan_) >= (d_3_outsideSchedule_):
                            d_5_openedG_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCur_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedG_ = out0_
                            d_6_openedInside_ = out1_
                            d_7_openedCur_ = out2_
                            generated = d_5_openedG_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCur_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_outsideTokensSinceLastSpan_ = 0
                        elif True:
                            d_8_remaining_: int
                            d_8_remaining_ = (maxSteps) - (d_1_steps_)
                            d_9_slackToSchedule_: int
                            d_9_slackToSchedule_ = (d_3_outsideSchedule_) - (d_2_outsideTokensSinceLastSpan_)
                            d_10_chunkBudget_: int
                            if (d_9_slackToSchedule_) < (d_8_remaining_):
                                d_10_chunkBudget_ = d_9_slackToSchedule_
                            elif True:
                                d_10_chunkBudget_ = d_8_remaining_
                            if (d_10_chunkBudget_) == (0):
                                d_11_openedG_: _dafny.Seq
                                d_12_openedInside_: bool
                                d_13_openedCur_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_11_openedG_ = out3_
                                d_12_openedInside_ = out4_
                                d_13_openedCur_ = out5_
                                generated = d_11_openedG_
                                insideConstrainedOut = d_12_openedInside_
                                currentConstrainedOut = d_13_openedCur_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_outsideTokensSinceLastSpan_ = 0
                            elif True:
                                d_14_genLenBefore_: int
                                d_14_genLenBefore_ = len(generated)
                                d_15_chunkedG_: _dafny.Seq
                                d_16_stoppedOpen_: bool
                                d_17_stoppedEos_: bool
                                d_18_stepsUsed_: int
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: bool
                                out9_: int
                                out6_, out7_, out8_, out9_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_15_chunkedG_ = out6_
                                d_16_stoppedOpen_ = out7_
                                d_17_stoppedEos_ = out8_
                                d_18_stepsUsed_ = out9_
                                generated = d_15_chunkedG_
                                d_1_steps_ = (d_1_steps_) + (d_18_stepsUsed_)
                                d_19_grew_: int
                                d_19_grew_ = (len(generated)) - (d_14_genLenBefore_)
                                d_2_outsideTokensSinceLastSpan_ = (d_2_outsideTokensSinceLastSpan_) + (d_19_grew_)
                                if d_17_stoppedEos_:
                                    raise _dafny.Break("0")
                                elif d_16_stoppedOpen_:
                                    d_20_entG_: _dafny.Seq
                                    d_21_entInside_: bool
                                    d_22_entCur_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_20_entG_ = out10_
                                    d_21_entInside_ = out11_
                                    d_22_entCur_ = out12_
                                    generated = d_20_entG_
                                    insideConstrainedOut = d_21_entInside_
                                    currentConstrainedOut = d_22_entCur_
                                    d_2_outsideTokensSinceLastSpan_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_23_closedG_: _dafny.Seq
                        d_24_closedInside_: bool
                        d_25_closedCur_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_23_closedG_ = out13_
                        d_24_closedInside_ = out14_
                        d_25_closedCur_ = out15_
                        generated = d_23_closedG_
                        insideConstrainedOut = d_24_closedInside_
                        currentConstrainedOut = d_25_closedCur_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_outsideTokensSinceLastSpan_ = 0
                    elif True:
                        d_26_stablePrefix_: _dafny.Seq
                        d_26_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_27_constrainedPrompt_: _dafny.Seq
                        d_27_constrainedPrompt_ = (prompt) + (d_26_stablePrefix_)
                        d_28_remaining_: int
                        d_28_remaining_ = (maxSteps) - (d_1_steps_)
                        d_29_symBudget_: int
                        if (d_4_innerSpanCap_) < (d_28_remaining_):
                            d_29_symBudget_ = d_4_innerSpanCap_
                        elif True:
                            d_29_symBudget_ = d_28_remaining_
                        if (d_29_symBudget_) == (0):
                            raise _dafny.Break("0")
                        d_30_symG_: _dafny.Seq
                        d_31_symCur_: _dafny.Seq
                        d_32_hitEos_: bool
                        d_33_stepsUsed_: int
                        out16_: _dafny.Seq
                        out17_: _dafny.Seq
                        out18_: bool
                        out19_: int
                        out16_, out17_, out18_, out19_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_27_constrainedPrompt_, generated, currentConstrainedOut, d_29_symBudget_, eosToken)
                        d_30_symG_ = out16_
                        d_31_symCur_ = out17_
                        d_32_hitEos_ = out18_
                        d_33_stepsUsed_ = out19_
                        generated = d_30_symG_
                        currentConstrainedOut = d_31_symCur_
                        d_1_steps_ = (d_1_steps_) + (d_33_stepsUsed_)
                        if d_32_hitEos_:
                            raise _dafny.Break("0")
                        if (d_33_stepsUsed_) == (0):
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

