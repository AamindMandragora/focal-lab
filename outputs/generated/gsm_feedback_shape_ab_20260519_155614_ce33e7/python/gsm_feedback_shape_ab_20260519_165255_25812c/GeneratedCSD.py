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
        if not(insideConstrainedOut):
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Every arithmetic computation must appear inside visible delimiters << and >>. Keep each computation short and close >> immediately after the computation.")))
        if (maxSteps) == (0):
            cost = 0
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_needOpenSoon_: bool
        d_2_needOpenSoon_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_3_isComplete_: bool
                        d_3_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_isComplete_:
                            d_4_closedGenerated_: _dafny.Seq
                            d_5_closedInside_: bool
                            d_6_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_closedGenerated_ = out0_
                            d_5_closedInside_ = out1_
                            d_6_closedCurrent_ = out2_
                            generated = d_4_closedGenerated_
                            insideConstrainedOut = d_5_closedInside_
                            currentConstrainedOut = d_6_closedCurrent_
                            d_2_needOpenSoon_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_stablePrefix_: _dafny.Seq
                            d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_8_constrainedPrompt_: _dafny.Seq
                            d_8_constrainedPrompt_ = (prompt) + (d_7_stablePrefix_)
                            d_9_remaining_: int
                            d_9_remaining_ = (maxSteps) - (d_1_steps_)
                            d_10_symbolBudget_: int
                            if (stepTokenBudget) == (0):
                                d_10_symbolBudget_ = 1
                            elif (stepTokenBudget) >= (d_9_remaining_):
                                d_10_symbolBudget_ = d_9_remaining_
                            elif True:
                                d_10_symbolBudget_ = stepTokenBudget
                            d_11_symbolGenerated_: _dafny.Seq
                            d_12_symbolOut_: _dafny.Seq
                            d_13_hitEos_: bool
                            d_14_stepsUsed2_: int
                            out3_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_8_constrainedPrompt_, generated, currentConstrainedOut, d_10_symbolBudget_, eosToken)
                            d_11_symbolGenerated_ = out3_
                            d_12_symbolOut_ = out4_
                            d_13_hitEos_ = out5_
                            d_14_stepsUsed2_ = out6_
                            generated = d_11_symbolGenerated_
                            currentConstrainedOut = d_12_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed2_)
                            if d_13_hitEos_:
                                raise _dafny.Break("0")
                    elif d_2_needOpenSoon_:
                        d_15_openedGenerated_: _dafny.Seq
                        d_16_openedInside_: bool
                        d_17_openedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_15_openedGenerated_ = out7_
                        d_16_openedInside_ = out8_
                        d_17_openedCurrent_ = out9_
                        generated = d_15_openedGenerated_
                        insideConstrainedOut = d_16_openedInside_
                        currentConstrainedOut = d_17_openedCurrent_
                        d_2_needOpenSoon_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_remainingOutside_: int
                        d_18_remainingOutside_ = (maxSteps) - (d_1_steps_)
                        d_19_chunkBudget_: int
                        if (d_18_remainingOutside_) <= (2):
                            d_19_chunkBudget_ = 1
                        elif True:
                            d_19_chunkBudget_ = 2
                        d_20_chunkedG_: _dafny.Seq
                        d_21_stoppedOpen_: bool
                        d_22_stoppedEos_: bool
                        d_23_stepsUsed_: int
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: bool
                        out13_: int
                        out10_, out11_, out12_, out13_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_19_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_20_chunkedG_ = out10_
                        d_21_stoppedOpen_ = out11_
                        d_22_stoppedEos_ = out12_
                        d_23_stepsUsed_ = out13_
                        generated = d_20_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_23_stepsUsed_)
                        if d_22_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_21_stoppedOpen_:
                            d_24_enteredGenerated_: _dafny.Seq
                            d_25_enteredInside_: bool
                            d_26_enteredCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_24_enteredGenerated_ = out14_
                            d_25_enteredInside_ = out15_
                            d_26_enteredCurrent_ = out16_
                            generated = d_24_enteredGenerated_
                            insideConstrainedOut = d_25_enteredInside_
                            currentConstrainedOut = d_26_enteredCurrent_
                            d_2_needOpenSoon_ = False
                        elif True:
                            d_27_sinceOpenCue_: int
                            out17_: int
                            out17_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                            d_27_sinceOpenCue_ = out17_
                            d_28_sincePlus_: int
                            out18_: int
                            out18_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")))
                            d_28_sincePlus_ = out18_
                            d_29_sinceMinus_: int
                            out19_: int
                            out19_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")))
                            d_29_sinceMinus_ = out19_
                            d_30_sinceTimes_: int
                            out20_: int
                            out20_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")))
                            d_30_sinceTimes_ = out20_
                            d_31_sinceDiv_: int
                            out21_: int
                            out21_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))
                            d_31_sinceDiv_ = out21_
                            if (((((d_27_sinceOpenCue_) <= (2)) or ((d_28_sincePlus_) <= (2))) or ((d_29_sinceMinus_) <= (2))) or ((d_30_sinceTimes_) <= (2))) or ((d_31_sinceDiv_) <= (2)):
                                d_2_needOpenSoon_ = True
                            elif True:
                                d_2_needOpenSoon_ = False
                    pass
            pass
        if insideConstrainedOut:
            d_32_canCloseAtEnd_: bool
            d_32_canCloseAtEnd_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if (d_32_canCloseAtEnd_) and ((d_1_steps_) < (maxSteps)):
                d_33_closedGenerated3_: _dafny.Seq
                d_34_closedInside3_: bool
                d_35_closedCurrent3_: _dafny.Seq
                out22_: _dafny.Seq
                out23_: bool
                out24_: _dafny.Seq
                out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_33_closedGenerated3_ = out22_
                d_34_closedInside3_ = out23_
                d_35_closedCurrent3_ = out24_
                generated = d_33_closedGenerated3_
                insideConstrainedOut = d_34_closedInside3_
                currentConstrainedOut = d_35_closedCurrent3_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

