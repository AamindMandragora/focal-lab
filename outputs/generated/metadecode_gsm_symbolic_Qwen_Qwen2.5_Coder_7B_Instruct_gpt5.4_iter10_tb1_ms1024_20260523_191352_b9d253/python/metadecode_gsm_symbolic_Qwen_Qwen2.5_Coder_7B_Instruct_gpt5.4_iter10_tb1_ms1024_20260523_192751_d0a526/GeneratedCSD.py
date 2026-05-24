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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Put each arithmetic computation inside visible << >>. Make the final reported answer itself appear in the last visible << >> span, and avoid opening a span until you are actually writing that computation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_eqToken_: _dafny.Seq
        d_2_eqToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))
        d_3_colonToken_: _dafny.Seq
        d_3_colonToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":"))
        d_4_isToken_: _dafny.Seq
        d_4_isToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is"))
        d_5_areToken_: _dafny.Seq
        d_5_areToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "are"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_openCount_: int
                        out0_: int
                        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_6_openCount_ = out0_
                        d_7_closeCount_: int
                        out1_: int
                        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                        d_7_closeCount_ = out1_
                        d_8_sinceEq_: int
                        out2_: int
                        out2_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_2_eqToken_)
                        d_8_sinceEq_ = out2_
                        d_9_sinceColon_: int
                        out3_: int
                        out3_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_3_colonToken_)
                        d_9_sinceColon_ = out3_
                        d_10_sinceIs_: int
                        out4_: int
                        out4_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_4_isToken_)
                        d_10_sinceIs_ = out4_
                        d_11_sinceAre_: int
                        out5_: int
                        out5_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_5_areToken_)
                        d_11_sinceAre_ = out5_
                        if ((((d_1_steps_) < (maxSteps)) and ((d_6_openCount_) == (d_7_closeCount_))) and ((len(generated)) >= ((len(generatedPrefix)) + (6)))) and (((((d_8_sinceEq_) <= (1)) or ((d_9_sinceColon_) <= (1))) or ((d_10_sinceIs_) <= (1))) or ((d_11_sinceAre_) <= (1))):
                            d_12_openedGenerated_: _dafny.Seq
                            d_13_openedInside_: bool
                            d_14_openedCurrent_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_12_openedGenerated_ = out6_
                            d_13_openedInside_ = out7_
                            d_14_openedCurrent_ = out8_
                            generated = d_12_openedGenerated_
                            insideConstrainedOut = d_13_openedInside_
                            currentConstrainedOut = d_14_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_15_remainingOutside_: int
                            d_15_remainingOutside_ = (maxSteps) - (d_1_steps_)
                            d_16_chunkBudget_: int
                            if (d_15_remainingOutside_) > (4):
                                d_16_chunkBudget_ = 4
                            elif True:
                                d_16_chunkBudget_ = d_15_remainingOutside_
                            d_17_chunkedGenerated_: _dafny.Seq
                            d_18_stoppedOpen_: bool
                            d_19_stoppedEos_: bool
                            d_20_stepsUsed_: int
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: bool
                            out12_: int
                            out9_, out10_, out11_, out12_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_16_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_17_chunkedGenerated_ = out9_
                            d_18_stoppedOpen_ = out10_
                            d_19_stoppedEos_ = out11_
                            d_20_stepsUsed_ = out12_
                            generated = d_17_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_20_stepsUsed_)
                            if d_19_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_18_stoppedOpen_:
                                d_21_enteredGenerated_: _dafny.Seq
                                d_22_enteredInside_: bool
                                d_23_enteredCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_21_enteredGenerated_ = out13_
                                d_22_enteredInside_ = out14_
                                d_23_enteredCurrent_ = out15_
                                generated = d_21_enteredGenerated_
                                insideConstrainedOut = d_22_enteredInside_
                                currentConstrainedOut = d_23_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_24_closedGenerated_: _dafny.Seq
                        d_25_closedInside_: bool
                        d_26_closedCurrent_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: bool
                        out18_: _dafny.Seq
                        out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_24_closedGenerated_ = out16_
                        d_25_closedInside_ = out17_
                        d_26_closedCurrent_ = out18_
                        generated = d_24_closedGenerated_
                        insideConstrainedOut = d_25_closedInside_
                        currentConstrainedOut = d_26_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_27_constrainedPrompt_: _dafny.Seq
                        d_27_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_28_next_: _dafny.Seq
                        d_28_next_ = eosToken
                        if (len(currentConstrainedOut)) == (0):
                            d_29_nextSoft_: _dafny.Seq
                            d_30_usedFallback_: bool
                            out19_: _dafny.Seq
                            out20_: bool
                            out19_, out20_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_29_nextSoft_ = out19_
                            d_30_usedFallback_ = out20_
                            d_28_next_ = d_29_nextSoft_
                        elif (len(currentConstrainedOut)) < (2):
                            out21_: _dafny.Seq
                            out21_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'), eosToken)
                            d_28_next_ = out21_
                        elif True:
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_28_next_ = out22_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_28_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_31_appendedGenerated_: _dafny.Seq
                            d_32_appendedInside_: bool
                            d_33_appendedCurrent_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: bool
                            out25_: _dafny.Seq
                            out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                            d_31_appendedGenerated_ = out23_
                            d_32_appendedInside_ = out24_
                            d_33_appendedCurrent_ = out25_
                            generated = d_31_appendedGenerated_
                            insideConstrainedOut = d_32_appendedInside_
                            currentConstrainedOut = d_33_appendedCurrent_
                    pass
            pass
        if ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and ((parser).IsCompletePrefix(currentConstrainedOut)):
            d_34_closedGenerated2_: _dafny.Seq
            d_35_closedInside2_: bool
            d_36_closedCurrent2_: _dafny.Seq
            out26_: _dafny.Seq
            out27_: bool
            out28_: _dafny.Seq
            out26_, out27_, out28_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_34_closedGenerated2_ = out26_
            d_35_closedInside2_ = out27_
            d_36_closedCurrent2_ = out28_
            generated = d_34_closedGenerated2_
            insideConstrainedOut = d_35_closedInside2_
            currentConstrainedOut = d_36_closedCurrent2_
            d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

