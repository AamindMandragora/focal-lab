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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Every arithmetic computation must appear inside a complete visible << >> span. Make the final reported answer appear in the last visible << >> span, and close each span immediately after the computation.")))
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
        d_6_plusToken_: _dafny.Seq
        d_6_plusToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+"))
        d_7_minusToken_: _dafny.Seq
        d_7_minusToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_openCount_: int
                        out0_: int
                        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_8_openCount_ = out0_
                        d_9_closeCount_: int
                        out1_: int
                        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                        d_9_closeCount_ = out1_
                        d_10_sinceEq_: int
                        out2_: int
                        out2_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_2_eqToken_)
                        d_10_sinceEq_ = out2_
                        d_11_sinceColon_: int
                        out3_: int
                        out3_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_3_colonToken_)
                        d_11_sinceColon_ = out3_
                        d_12_sinceIs_: int
                        out4_: int
                        out4_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_4_isToken_)
                        d_12_sinceIs_ = out4_
                        d_13_sinceAre_: int
                        out5_: int
                        out5_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_5_areToken_)
                        d_13_sinceAre_ = out5_
                        d_14_sincePlus_: int
                        out6_: int
                        out6_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_6_plusToken_)
                        d_14_sincePlus_ = out6_
                        d_15_sinceMinus_: int
                        out7_: int
                        out7_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_7_minusToken_)
                        d_15_sinceMinus_ = out7_
                        if ((((d_1_steps_) < (maxSteps)) and ((d_8_openCount_) == (d_9_closeCount_))) and ((len(generated)) > (len(generatedPrefix)))) and (((((((d_10_sinceEq_) <= (2)) or ((d_11_sinceColon_) <= (2))) or ((d_12_sinceIs_) <= (2))) or ((d_13_sinceAre_) <= (2))) or ((d_14_sincePlus_) <= (1))) or ((d_15_sinceMinus_) <= (1))):
                            d_16_openedGenerated_: _dafny.Seq
                            d_17_openedInside_: bool
                            d_18_openedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_16_openedGenerated_ = out8_
                            d_17_openedInside_ = out9_
                            d_18_openedCurrent_ = out10_
                            generated = d_16_openedGenerated_
                            insideConstrainedOut = d_17_openedInside_
                            currentConstrainedOut = d_18_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_19_remainingOutside_: int
                            d_19_remainingOutside_ = (maxSteps) - (d_1_steps_)
                            d_20_chunkBudget_: int
                            if (d_19_remainingOutside_) > (3):
                                d_20_chunkBudget_ = 3
                            elif True:
                                d_20_chunkBudget_ = d_19_remainingOutside_
                            d_21_chunkedGenerated_: _dafny.Seq
                            d_22_stoppedOpen_: bool
                            d_23_stoppedEos_: bool
                            d_24_stepsUsed_: int
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: bool
                            out14_: int
                            out11_, out12_, out13_, out14_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_20_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_21_chunkedGenerated_ = out11_
                            d_22_stoppedOpen_ = out12_
                            d_23_stoppedEos_ = out13_
                            d_24_stepsUsed_ = out14_
                            generated = d_21_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_24_stepsUsed_)
                            if d_23_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_22_stoppedOpen_:
                                d_25_enteredGenerated_: _dafny.Seq
                                d_26_enteredInside_: bool
                                d_27_enteredCurrent_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_25_enteredGenerated_ = out15_
                                d_26_enteredInside_ = out16_
                                d_27_enteredCurrent_ = out17_
                                generated = d_25_enteredGenerated_
                                insideConstrainedOut = d_26_enteredInside_
                                currentConstrainedOut = d_27_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_28_closedGenerated_: _dafny.Seq
                        d_29_closedInside_: bool
                        d_30_closedCurrent_: _dafny.Seq
                        out18_: _dafny.Seq
                        out19_: bool
                        out20_: _dafny.Seq
                        out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_28_closedGenerated_ = out18_
                        d_29_closedInside_ = out19_
                        d_30_closedCurrent_ = out20_
                        generated = d_28_closedGenerated_
                        insideConstrainedOut = d_29_closedInside_
                        currentConstrainedOut = d_30_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_31_constrainedPrompt_: _dafny.Seq
                        d_31_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_32_next_: _dafny.Seq
                        d_32_next_ = eosToken
                        if (len(currentConstrainedOut)) == (0):
                            d_33_nextSoft_: _dafny.Seq
                            d_34_usedFallback_: bool
                            out21_: _dafny.Seq
                            out22_: bool
                            out21_, out22_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_31_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_33_nextSoft_ = out21_
                            d_34_usedFallback_ = out22_
                            d_32_next_ = d_33_nextSoft_
                        elif (len(currentConstrainedOut)) < (2):
                            out23_: _dafny.Seq
                            out23_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_31_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'), eosToken)
                            d_32_next_ = out23_
                        elif True:
                            out24_: _dafny.Seq
                            out24_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_31_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_32_next_ = out24_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_32_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_35_appendedGenerated_: _dafny.Seq
                            d_36_appendedInside_: bool
                            d_37_appendedCurrent_: _dafny.Seq
                            out25_: _dafny.Seq
                            out26_: bool
                            out27_: _dafny.Seq
                            out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next_)
                            d_35_appendedGenerated_ = out25_
                            d_36_appendedInside_ = out26_
                            d_37_appendedCurrent_ = out27_
                            generated = d_35_appendedGenerated_
                            insideConstrainedOut = d_36_appendedInside_
                            currentConstrainedOut = d_37_appendedCurrent_
                    pass
            pass
        if ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and ((parser).IsCompletePrefix(currentConstrainedOut)):
            d_38_closedGenerated2_: _dafny.Seq
            d_39_closedInside2_: bool
            d_40_closedCurrent2_: _dafny.Seq
            out28_: _dafny.Seq
            out29_: bool
            out30_: _dafny.Seq
            out28_, out29_, out30_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_38_closedGenerated2_ = out28_
            d_39_closedInside2_ = out29_
            d_40_closedCurrent2_ = out30_
            generated = d_38_closedGenerated2_
            insideConstrainedOut = d_39_closedInside2_
            currentConstrainedOut = d_40_closedCurrent2_
            d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

