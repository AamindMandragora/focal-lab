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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 4
        d_3_openedOnce_: bool
        d_3_openedOnce_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_3_openedOnce_:
                            d_4_rem0_: int
                            d_4_rem0_ = (maxSteps) - (d_1_steps_)
                            if (d_4_rem0_) == (0):
                                raise _dafny.Break("0")
                            elif True:
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
                                d_3_openedOnce_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_rem1_: int
                            d_8_rem1_ = (maxSteps) - (d_1_steps_)
                            if (d_8_rem1_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_9_chunkBudget_: int
                                d_9_chunkBudget_ = 1
                                d_10_chunkedGenerated_: _dafny.Seq
                                d_11_stoppedOnOpenSpan_: bool
                                d_12_stoppedOnEos_: bool
                                d_13_stepsUsed_: int
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: bool
                                out6_: int
                                out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_10_chunkedGenerated_ = out3_
                                d_11_stoppedOnOpenSpan_ = out4_
                                d_12_stoppedOnEos_ = out5_
                                d_13_stepsUsed_ = out6_
                                generated = d_10_chunkedGenerated_
                                d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                                if d_12_stoppedOnEos_:
                                    raise _dafny.Break("0")
                                elif d_11_stoppedOnOpenSpan_:
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_openedOnce_ = True
                                elif True:
                                    if (d_1_steps_) < (maxSteps):
                                        d_14_openedGenerated2_: _dafny.Seq
                                        d_15_openedInside2_: bool
                                        d_16_openedCurrent2_: _dafny.Seq
                                        out7_: _dafny.Seq
                                        out8_: bool
                                        out9_: _dafny.Seq
                                        out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                        d_14_openedGenerated2_ = out7_
                                        d_15_openedInside2_ = out8_
                                        d_16_openedCurrent2_ = out9_
                                        generated = d_14_openedGenerated2_
                                        insideConstrainedOut = d_15_openedInside2_
                                        currentConstrainedOut = d_16_openedCurrent2_
                                        d_3_openedOnce_ = True
                                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_17_complete_: bool
                        d_17_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_17_complete_:
                            d_18_rem2_: int
                            d_18_rem2_ = (maxSteps) - (d_1_steps_)
                            if (d_18_rem2_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_19_closedGenerated_: _dafny.Seq
                                d_20_closedInside_: bool
                                d_21_closedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_19_closedGenerated_ = out10_
                                d_20_closedInside_ = out11_
                                d_21_closedCurrent_ = out12_
                                generated = d_19_closedGenerated_
                                insideConstrainedOut = d_20_closedInside_
                                currentConstrainedOut = d_21_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_22_stablePrefix_: _dafny.Seq
                            d_22_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_23_constrainedPrompt_: _dafny.Seq
                            d_23_constrainedPrompt_ = (prompt) + (d_22_stablePrefix_)
                            d_24_validCount_: int
                            out13_: int
                            out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_24_validCount_ = out13_
                            d_25_rem3_: int
                            d_25_rem3_ = (maxSteps) - (d_1_steps_)
                            if (d_25_rem3_) == (0):
                                raise _dafny.Break("0")
                            elif (d_24_validCount_) <= (d_2_narrowThreshold_):
                                d_26_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_26_next_ = out14_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_26_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_27_appendedGenerated_: _dafny.Seq
                                    d_28_appendedInside_: bool
                                    d_29_appendedCurrent_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                    d_27_appendedGenerated_ = out15_
                                    d_28_appendedInside_ = out16_
                                    d_29_appendedCurrent_ = out17_
                                    generated = d_27_appendedGenerated_
                                    insideConstrainedOut = d_28_appendedInside_
                                    currentConstrainedOut = d_29_appendedCurrent_
                            elif True:
                                d_30_symbolOut_: _dafny.Seq
                                d_31_hitEos_: bool
                                d_32_stepsUsed2_: int
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: int
                                out18_, out19_, out20_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, d_25_rem3_, eosToken)
                                d_30_symbolOut_ = out18_
                                d_31_hitEos_ = out19_
                                d_32_stepsUsed2_ = out20_
                                generated = (d_22_stablePrefix_) + (d_30_symbolOut_)
                                insideConstrainedOut = True
                                currentConstrainedOut = d_30_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_32_stepsUsed2_)
                                if d_31_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

