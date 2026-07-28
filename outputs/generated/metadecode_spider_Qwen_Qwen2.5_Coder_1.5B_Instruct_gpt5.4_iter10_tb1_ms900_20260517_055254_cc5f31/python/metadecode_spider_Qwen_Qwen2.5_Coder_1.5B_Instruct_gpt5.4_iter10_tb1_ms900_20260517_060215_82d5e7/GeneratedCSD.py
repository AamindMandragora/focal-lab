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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query inside << and >>, with no explanation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_outsideBudget_: int
        d_2_outsideBudget_ = 4
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((len(generated)) == (len(generatedPrefix))) and ((d_2_outsideBudget_) > (0)):
                            d_4_remaining0_: int
                            d_4_remaining0_ = (maxSteps) - (d_1_steps_)
                            d_5_chunkBudget_: int
                            if (d_2_outsideBudget_) > (d_4_remaining0_):
                                d_5_chunkBudget_ = d_4_remaining0_
                            elif True:
                                d_5_chunkBudget_ = d_2_outsideBudget_
                            d_6_chunkedG_: _dafny.Seq
                            d_7_stoppedOpen_: bool
                            d_8_stoppedEos_: bool
                            d_9_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_6_chunkedG_ = out0_
                            d_7_stoppedOpen_ = out1_
                            d_8_stoppedEos_ = out2_
                            d_9_stepsUsed_ = out3_
                            generated = d_6_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                            if d_8_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_7_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif (d_1_steps_) < (maxSteps):
                                d_10_openedGenerated_: _dafny.Seq
                                d_11_openedInside_: bool
                                d_12_openedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_10_openedGenerated_ = out4_
                                d_11_openedInside_ = out5_
                                d_12_openedCurrent_ = out6_
                                generated = d_10_openedGenerated_
                                insideConstrainedOut = d_11_openedInside_
                                currentConstrainedOut = d_12_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_openedGenerated2_: _dafny.Seq
                            d_14_openedInside2_: bool
                            d_15_openedCurrent2_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_13_openedGenerated2_ = out7_
                            d_14_openedInside2_ = out8_
                            d_15_openedCurrent2_ = out9_
                            generated = d_13_openedGenerated2_
                            insideConstrainedOut = d_14_openedInside2_
                            currentConstrainedOut = d_15_openedCurrent2_
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out10_
                        d_17_closedInside_ = out11_
                        d_18_closedCurrent_ = out12_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_19_stablePrefix_: _dafny.Seq
                        d_19_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (d_19_stablePrefix_)
                        d_21_validCount_: int
                        out13_: int
                        out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_21_validCount_ = out13_
                        if (d_21_validCount_) <= (d_3_narrowThreshold_):
                            d_22_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_22_next_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_23_appendedGenerated_: _dafny.Seq
                                d_24_appendedInside_: bool
                                d_25_appendedCurrent_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_23_appendedGenerated_ = out15_
                                d_24_appendedInside_ = out16_
                                d_25_appendedCurrent_ = out17_
                                generated = d_23_appendedGenerated_
                                insideConstrainedOut = d_24_appendedInside_
                                currentConstrainedOut = d_25_appendedCurrent_
                        elif True:
                            d_26_remaining_: int
                            d_26_remaining_ = (maxSteps) - (d_1_steps_)
                            d_27_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_26_remaining_)):
                                d_27_symbolBudget_ = d_26_remaining_
                            elif True:
                                d_27_symbolBudget_ = stepTokenBudget
                            d_28_symbolGenerated_: _dafny.Seq
                            d_29_symbolOut_: _dafny.Seq
                            d_30_hitEos_: bool
                            d_31_stepsUsed2_: int
                            out18_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: int
                            out18_, out19_, out20_, out21_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_20_constrainedPrompt_, generated, currentConstrainedOut, d_27_symbolBudget_, eosToken)
                            d_28_symbolGenerated_ = out18_
                            d_29_symbolOut_ = out19_
                            d_30_hitEos_ = out20_
                            d_31_stepsUsed2_ = out21_
                            generated = d_28_symbolGenerated_
                            currentConstrainedOut = d_29_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_31_stepsUsed2_)
                            if d_30_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        if ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and ((parser).IsCompletePrefix(currentConstrainedOut)):
            d_32_closedGenerated2_: _dafny.Seq
            d_33_closedInside2_: bool
            d_34_closedCurrent2_: _dafny.Seq
            out22_: _dafny.Seq
            out23_: bool
            out24_: _dafny.Seq
            out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_32_closedGenerated2_ = out22_
            d_33_closedInside2_ = out23_
            d_34_closedCurrent2_ = out24_
            generated = d_32_closedGenerated2_
            insideConstrainedOut = d_33_closedInside2_
            currentConstrainedOut = d_34_closedCurrent2_
            d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

