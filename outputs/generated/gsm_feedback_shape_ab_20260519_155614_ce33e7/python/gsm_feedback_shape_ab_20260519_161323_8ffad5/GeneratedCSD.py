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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Every arithmetic computation must appear inside visible delimiters << and >>. Keep each computation short and close >> immediately after the computation.")))
        if (maxSteps) == (0):
            if not(insideConstrainedOut):
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        elif True:
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_needOpenSoon_: bool
            d_2_needOpenSoon_ = False
            with _dafny.label("1_0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("1_0"):
                        if (insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                            d_3_closedGenerated_: _dafny.Seq
                            d_4_closedInside_: bool
                            d_5_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_3_closedGenerated_ = out0_
                            d_4_closedInside_ = out1_
                            d_5_closedCurrent_ = out2_
                            generated = d_3_closedGenerated_
                            insideConstrainedOut = d_4_closedInside_
                            currentConstrainedOut = d_5_closedCurrent_
                            d_2_needOpenSoon_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (not(insideConstrainedOut)) and (d_2_needOpenSoon_):
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out3_
                            d_7_openedInside_ = out4_
                            d_8_openedCurrent_ = out5_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_2_needOpenSoon_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif not(insideConstrainedOut):
                            d_9_remainingOutside_: int
                            d_9_remainingOutside_ = (maxSteps) - (d_1_steps_)
                            d_10_chunkBudget_: int
                            if (d_9_remainingOutside_) <= (2):
                                d_10_chunkBudget_ = 1
                            elif True:
                                d_10_chunkBudget_ = 2
                            d_11_chunkedG_: _dafny.Seq
                            d_12_stoppedOpen_: bool
                            d_13_stoppedEos_: bool
                            d_14_stepsUsed_: int
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: bool
                            out9_: int
                            out6_, out7_, out8_, out9_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_11_chunkedG_ = out6_
                            d_12_stoppedOpen_ = out7_
                            d_13_stoppedEos_ = out8_
                            d_14_stepsUsed_ = out9_
                            generated = d_11_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                            if d_13_stoppedEos_:
                                raise _dafny.Break("1_0")
                            elif d_12_stoppedOpen_:
                                d_15_enteredGenerated_: _dafny.Seq
                                d_16_enteredInside_: bool
                                d_17_enteredCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_15_enteredGenerated_ = out10_
                                d_16_enteredInside_ = out11_
                                d_17_enteredCurrent_ = out12_
                                generated = d_15_enteredGenerated_
                                insideConstrainedOut = d_16_enteredInside_
                                currentConstrainedOut = d_17_enteredCurrent_
                                d_2_needOpenSoon_ = False
                            elif True:
                                d_18_sinceOpenCue_: int
                                out13_: int
                                out13_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_18_sinceOpenCue_ = out13_
                                d_19_sincePlus_: int
                                out14_: int
                                out14_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")))
                                d_19_sincePlus_ = out14_
                                d_20_sinceMinus_: int
                                out15_: int
                                out15_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")))
                                d_20_sinceMinus_ = out15_
                                d_21_sinceTimes_: int
                                out16_: int
                                out16_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")))
                                d_21_sinceTimes_ = out16_
                                d_22_sinceDiv_: int
                                out17_: int
                                out17_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))
                                d_22_sinceDiv_ = out17_
                                if (((((d_18_sinceOpenCue_) <= (2)) or ((d_19_sincePlus_) <= (2))) or ((d_20_sinceMinus_) <= (2))) or ((d_21_sinceTimes_) <= (2))) or ((d_22_sinceDiv_) <= (2)):
                                    d_2_needOpenSoon_ = True
                                elif True:
                                    d_2_needOpenSoon_ = False
                        elif True:
                            d_23_stablePrefix_: _dafny.Seq
                            d_23_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_24_constrainedPrompt_: _dafny.Seq
                            d_24_constrainedPrompt_ = (prompt) + (d_23_stablePrefix_)
                            d_25_remaining_: int
                            d_25_remaining_ = (maxSteps) - (d_1_steps_)
                            if ((d_25_remaining_) == (1)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_26_closedGenerated2_: _dafny.Seq
                                d_27_closedInside2_: bool
                                d_28_closedCurrent2_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_26_closedGenerated2_ = out18_
                                d_27_closedInside2_ = out19_
                                d_28_closedCurrent2_ = out20_
                                generated = d_26_closedGenerated2_
                                insideConstrainedOut = d_27_closedInside2_
                                currentConstrainedOut = d_28_closedCurrent2_
                                d_2_needOpenSoon_ = False
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_29_symbolBudget_: int
                                if (stepTokenBudget) == (0):
                                    d_29_symbolBudget_ = 1
                                elif (stepTokenBudget) >= (d_25_remaining_):
                                    d_29_symbolBudget_ = d_25_remaining_
                                elif True:
                                    d_29_symbolBudget_ = stepTokenBudget
                                d_30_symbolGenerated_: _dafny.Seq
                                d_31_symbolOut_: _dafny.Seq
                                d_32_hitEos_: bool
                                d_33_stepsUsed2_: int
                                out21_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: int
                                out21_, out22_, out23_, out24_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_24_constrainedPrompt_, generated, currentConstrainedOut, d_29_symbolBudget_, eosToken)
                                d_30_symbolGenerated_ = out21_
                                d_31_symbolOut_ = out22_
                                d_32_hitEos_ = out23_
                                d_33_stepsUsed2_ = out24_
                                generated = d_30_symbolGenerated_
                                currentConstrainedOut = d_31_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_33_stepsUsed2_)
                                if d_32_hitEos_:
                                    raise _dafny.Break("1_0")
                        pass
                pass
            if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                d_34_closedGenerated3_: _dafny.Seq
                d_35_closedInside3_: bool
                d_36_closedCurrent3_: _dafny.Seq
                out25_: _dafny.Seq
                out26_: bool
                out27_: _dafny.Seq
                out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_34_closedGenerated3_ = out25_
                d_35_closedInside3_ = out26_
                d_36_closedCurrent3_ = out27_
                generated = d_34_closedGenerated3_
                insideConstrainedOut = d_35_closedInside3_
                currentConstrainedOut = d_36_closedCurrent3_
                d_1_steps_ = (d_1_steps_) + (1)
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

