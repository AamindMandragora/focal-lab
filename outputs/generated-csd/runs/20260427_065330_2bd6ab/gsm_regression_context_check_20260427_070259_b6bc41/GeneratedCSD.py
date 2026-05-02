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
        d_2_openSpanToken_: _dafny.Seq
        d_2_openSpanToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_3_minOutsideBeforeOpen_: int
        d_3_minOutsideBeforeOpen_ = 4
        d_4_reserveToFinish_: int
        d_4_reserveToFinish_ = 3
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remaining_: int
                        d_5_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_5_remaining_) <= (d_4_reserveToFinish_):
                            d_6_chunkGenerated_: _dafny.Seq
                            d_7_stoppedOnOpen_: bool
                            d_8_stoppedOnEos_: bool
                            d_9_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_remaining_, d_2_openSpanToken_, eosToken)
                            d_6_chunkGenerated_ = out0_
                            d_7_stoppedOnOpen_ = out1_
                            d_8_stoppedOnEos_ = out2_
                            d_9_stepsUsed_ = out3_
                            generated = d_6_chunkGenerated_
                            if (d_9_stepsUsed_) <= (d_5_remaining_):
                                d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                            elif True:
                                raise _dafny.Break("0")
                            if (d_8_stoppedOnEos_) or ((d_9_stepsUsed_) == (0)):
                                raise _dafny.Break("0")
                        elif (len(generated)) < ((len(generatedPrefix)) + (d_3_minOutsideBeforeOpen_)):
                            d_10_chunkLimit_: int
                            d_10_chunkLimit_ = 1
                            d_11_chunkGenerated2_: _dafny.Seq
                            d_12_stoppedOnOpen2_: bool
                            d_13_stoppedOnEos2_: bool
                            d_14_stepsUsed2_: int
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: bool
                            out7_: int
                            out4_, out5_, out6_, out7_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_chunkLimit_, d_2_openSpanToken_, eosToken)
                            d_11_chunkGenerated2_ = out4_
                            d_12_stoppedOnOpen2_ = out5_
                            d_13_stoppedOnEos2_ = out6_
                            d_14_stepsUsed2_ = out7_
                            generated = d_11_chunkGenerated2_
                            if (d_14_stepsUsed2_) <= (d_10_chunkLimit_):
                                d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed2_)
                            elif True:
                                raise _dafny.Break("0")
                            if (d_13_stoppedOnEos2_) or ((d_14_stepsUsed2_) == (0)):
                                raise _dafny.Break("0")
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_2_openSpanToken_]), _dafny.BigRational('8e0'))
                            d_15_top_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_15_top_ = out8_
                            if VerifiedDecoderAgent.default__.Contains(d_15_top_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_16_openedGenerated_: _dafny.Seq
                                d_17_openedInside_: bool
                                d_18_openedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_16_openedGenerated_ = out9_
                                d_17_openedInside_ = out10_
                                d_18_openedCurrent_ = out11_
                                generated = d_16_openedGenerated_
                                insideConstrainedOut = d_17_openedInside_
                                currentConstrainedOut = d_18_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_19_chunkLimit2_: int
                                d_19_chunkLimit2_ = 1
                                d_20_chunkGenerated3_: _dafny.Seq
                                d_21_stoppedOnOpen3_: bool
                                d_22_stoppedOnEos3_: bool
                                d_23_stepsUsed3_: int
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: bool
                                out15_: int
                                out12_, out13_, out14_, out15_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_19_chunkLimit2_, d_2_openSpanToken_, eosToken)
                                d_20_chunkGenerated3_ = out12_
                                d_21_stoppedOnOpen3_ = out13_
                                d_22_stoppedOnEos3_ = out14_
                                d_23_stepsUsed3_ = out15_
                                generated = d_20_chunkGenerated3_
                                if (d_23_stepsUsed3_) <= (d_19_chunkLimit2_):
                                    d_1_steps_ = (d_1_steps_) + (d_23_stepsUsed3_)
                                elif True:
                                    raise _dafny.Break("0")
                                if (d_22_stoppedOnEos3_) or ((d_23_stepsUsed3_) == (0)):
                                    raise _dafny.Break("0")
                    elif True:
                        d_24_isComplete_: bool
                        d_24_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_24_isComplete_:
                            if (d_1_steps_) < (maxSteps):
                                d_25_closedGenerated_: _dafny.Seq
                                d_26_closedInside_: bool
                                d_27_closedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_25_closedGenerated_ = out16_
                                d_26_closedInside_ = out17_
                                d_27_closedCurrent_ = out18_
                                generated = d_25_closedGenerated_
                                insideConstrainedOut = d_26_closedInside_
                                currentConstrainedOut = d_27_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_28_stablePrefix_: _dafny.Seq
                            d_28_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_29_constrainedPrompt_: _dafny.Seq
                            d_29_constrainedPrompt_ = (prompt) + (d_28_stablePrefix_)
                            d_30_remaining2_: int
                            d_30_remaining2_ = (maxSteps) - (d_1_steps_)
                            if (stepTokenBudget) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_31_currentOut_: _dafny.Seq
                                d_32_hitEos_: bool
                                d_33_stepsUsed4_: int
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: int
                                out19_, out20_, out21_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, stepTokenBudget, eosToken)
                                d_31_currentOut_ = out19_
                                d_32_hitEos_ = out20_
                                d_33_stepsUsed4_ = out21_
                                if (d_33_stepsUsed4_) <= (d_30_remaining2_):
                                    d_1_steps_ = (d_1_steps_) + (d_33_stepsUsed4_)
                                elif True:
                                    raise _dafny.Break("0")
                                if (d_32_hitEos_) or ((d_33_stepsUsed4_) == (0)):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (d_28_stablePrefix_) + (d_31_currentOut_)
                                    insideConstrainedOut = True
                                    currentConstrainedOut = d_31_currentOut_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

