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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Every arithmetic computation must appear inside a complete visible << >> span. Open << only when you are writing the computation itself, close >> immediately after it, and make the final reported answer appear in the last visible << >> span.")))
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
        d_8_closeTok_: _dafny.Seq
        d_8_closeTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_9_openCount_: int
                        out0_: int
                        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_9_openCount_ = out0_
                        d_10_closeCount_: int
                        out1_: int
                        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                        d_10_closeCount_ = out1_
                        d_11_sinceEq_: int
                        out2_: int
                        out2_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_2_eqToken_)
                        d_11_sinceEq_ = out2_
                        d_12_sinceColon_: int
                        out3_: int
                        out3_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_3_colonToken_)
                        d_12_sinceColon_ = out3_
                        d_13_sinceIs_: int
                        out4_: int
                        out4_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_4_isToken_)
                        d_13_sinceIs_ = out4_
                        d_14_sinceAre_: int
                        out5_: int
                        out5_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_5_areToken_)
                        d_14_sinceAre_ = out5_
                        d_15_sincePlus_: int
                        out6_: int
                        out6_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_6_plusToken_)
                        d_15_sincePlus_ = out6_
                        d_16_sinceMinus_: int
                        out7_: int
                        out7_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_7_minusToken_)
                        d_16_sinceMinus_ = out7_
                        d_17_sinceClose_: int
                        out8_: int
                        out8_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_8_closeTok_)
                        d_17_sinceClose_ = out8_
                        if (((((d_1_steps_) < (maxSteps)) and ((d_9_openCount_) == (d_10_closeCount_))) and ((len(generated)) > (len(generatedPrefix)))) and ((d_17_sinceClose_) > (2))) and (((((((d_11_sinceEq_) <= (1)) or ((d_12_sinceColon_) <= (1))) or ((d_13_sinceIs_) <= (1))) or ((d_14_sinceAre_) <= (1))) or ((d_15_sincePlus_) == (0))) or ((d_16_sinceMinus_) == (0))):
                            d_18_openedGenerated_: _dafny.Seq
                            d_19_openedInside_: bool
                            d_20_openedCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_18_openedGenerated_ = out9_
                            d_19_openedInside_ = out10_
                            d_20_openedCurrent_ = out11_
                            generated = d_18_openedGenerated_
                            insideConstrainedOut = d_19_openedInside_
                            currentConstrainedOut = d_20_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_21_remainingOutside_: int
                            d_21_remainingOutside_ = (maxSteps) - (d_1_steps_)
                            d_22_chunkBudget_: int
                            if (d_21_remainingOutside_) > (3):
                                d_22_chunkBudget_ = 3
                            elif True:
                                d_22_chunkBudget_ = d_21_remainingOutside_
                            d_23_chunkedGenerated_: _dafny.Seq
                            d_24_stoppedOpen_: bool
                            d_25_stoppedEos_: bool
                            d_26_stepsUsed_: int
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: bool
                            out15_: int
                            out12_, out13_, out14_, out15_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_22_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_23_chunkedGenerated_ = out12_
                            d_24_stoppedOpen_ = out13_
                            d_25_stoppedEos_ = out14_
                            d_26_stepsUsed_ = out15_
                            generated = d_23_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_26_stepsUsed_)
                            if d_25_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_24_stoppedOpen_:
                                d_27_enteredGenerated_: _dafny.Seq
                                d_28_enteredInside_: bool
                                d_29_enteredCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_27_enteredGenerated_ = out16_
                                d_28_enteredInside_ = out17_
                                d_29_enteredCurrent_ = out18_
                                generated = d_27_enteredGenerated_
                                insideConstrainedOut = d_28_enteredInside_
                                currentConstrainedOut = d_29_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_30_closedGenerated_: _dafny.Seq
                        d_31_closedInside_: bool
                        d_32_closedCurrent_: _dafny.Seq
                        out19_: _dafny.Seq
                        out20_: bool
                        out21_: _dafny.Seq
                        out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_30_closedGenerated_ = out19_
                        d_31_closedInside_ = out20_
                        d_32_closedCurrent_ = out21_
                        generated = d_30_closedGenerated_
                        insideConstrainedOut = d_31_closedInside_
                        currentConstrainedOut = d_32_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_33_constrainedPrompt_: _dafny.Seq
                        d_33_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_34_next_: _dafny.Seq
                        d_34_next_ = eosToken
                        if (len(currentConstrainedOut)) < (2):
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_33_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'), eosToken)
                            d_34_next_ = out22_
                        elif True:
                            out23_: _dafny.Seq
                            out23_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_33_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_34_next_ = out23_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_34_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_35_appendedGenerated_: _dafny.Seq
                            d_36_appendedInside_: bool
                            d_37_appendedCurrent_: _dafny.Seq
                            out24_: _dafny.Seq
                            out25_: bool
                            out26_: _dafny.Seq
                            out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_34_next_)
                            d_35_appendedGenerated_ = out24_
                            d_36_appendedInside_ = out25_
                            d_37_appendedCurrent_ = out26_
                            generated = d_35_appendedGenerated_
                            insideConstrainedOut = d_36_appendedInside_
                            currentConstrainedOut = d_37_appendedCurrent_
                    pass
            pass
        if ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and ((parser).IsCompletePrefix(currentConstrainedOut)):
            d_38_closedGenerated2_: _dafny.Seq
            d_39_closedInside2_: bool
            d_40_closedCurrent2_: _dafny.Seq
            out27_: _dafny.Seq
            out28_: bool
            out29_: _dafny.Seq
            out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_38_closedGenerated2_ = out27_
            d_39_closedInside2_ = out28_
            d_40_closedCurrent2_ = out29_
            generated = d_38_closedGenerated2_
            insideConstrainedOut = d_39_closedInside2_
            currentConstrainedOut = d_40_closedCurrent2_
            d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

