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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Keep the reasoning outside delimiters. When you give the final answer, put only the final numeric answer inside exactly one << >> span.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedAnswerSpan_: bool
        d_2_openedAnswerSpan_ = insideConstrained
        d_3_chunkSize_: int
        d_3_chunkSize_ = 24
        d_4_answerCueSeen_: bool
        d_4_answerCueSeen_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remaining_: int
                        d_5_remaining_ = (maxSteps) - (d_1_steps_)
                        d_6_sinceOpen_: int
                        out0_: int
                        out0_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_6_sinceOpen_ = out0_
                        d_7_shouldForceOpen_: bool
                        d_7_shouldForceOpen_ = False
                        if not(d_2_openedAnswerSpan_):
                            if (d_4_answerCueSeen_) and ((d_5_remaining_) > (1)):
                                d_7_shouldForceOpen_ = True
                            elif ((len(generated)) > ((len(generatedPrefix)) + (80))) and ((d_5_remaining_) > (1)):
                                d_7_shouldForceOpen_ = True
                            elif (d_5_remaining_) <= (3):
                                d_7_shouldForceOpen_ = True
                        if d_7_shouldForceOpen_:
                            d_8_openedGenerated_: _dafny.Seq
                            d_9_openedInside_: bool
                            d_10_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_openedGenerated_ = out1_
                            d_9_openedInside_ = out2_
                            d_10_openedCurrent_ = out3_
                            generated = d_8_openedGenerated_
                            insideConstrainedOut = d_9_openedInside_
                            currentConstrainedOut = d_10_openedCurrent_
                            d_2_openedAnswerSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_chunkBudget_: int
                            if (d_5_remaining_) < (d_3_chunkSize_):
                                d_11_chunkBudget_ = d_5_remaining_
                            elif True:
                                d_11_chunkBudget_ = d_3_chunkSize_
                            d_12_chunkedGenerated_: _dafny.Seq
                            d_13_stoppedOnOpenSpan_: bool
                            d_14_stoppedOnEos_: bool
                            d_15_stepsUsed_: int
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: bool
                            out7_: int
                            out4_, out5_, out6_, out7_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_11_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_12_chunkedGenerated_ = out4_
                            d_13_stoppedOnOpenSpan_ = out5_
                            d_14_stoppedOnEos_ = out6_
                            d_15_stepsUsed_ = out7_
                            generated = d_12_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_15_stepsUsed_)
                            if d_14_stoppedOnEos_:
                                if (not(d_2_openedAnswerSpan_)) and ((d_1_steps_) < (maxSteps)):
                                    d_16_openedGenerated2_: _dafny.Seq
                                    d_17_openedInside2_: bool
                                    d_18_openedCurrent2_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_16_openedGenerated2_ = out8_
                                    d_17_openedInside2_ = out9_
                                    d_18_openedCurrent2_ = out10_
                                    generated = d_16_openedGenerated2_
                                    insideConstrainedOut = d_17_openedInside2_
                                    currentConstrainedOut = d_18_openedCurrent2_
                                    d_2_openedAnswerSpan_ = True
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif d_13_stoppedOnOpenSpan_:
                                d_19_enteredGenerated_: _dafny.Seq
                                d_20_enteredInside_: bool
                                d_21_enteredCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_19_enteredGenerated_ = out11_
                                d_20_enteredInside_ = out12_
                                d_21_enteredCurrent_ = out13_
                                generated = d_19_enteredGenerated_
                                insideConstrainedOut = d_20_enteredInside_
                                currentConstrainedOut = d_21_enteredCurrent_
                                d_2_openedAnswerSpan_ = True
                            elif True:
                                d_22_eqCount_: int
                                out14_: int
                                out14_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_22_eqCount_ = out14_
                                d_23_colonCount_: int
                                out15_: int
                                out15_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))
                                d_23_colonCount_ = out15_
                                if (((d_22_eqCount_) > (0)) or ((d_23_colonCount_) > (0))) or ((d_6_sinceOpen_) > (60)):
                                    d_4_answerCueSeen_ = True
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
                        raise _dafny.Break("0")
                    elif True:
                        d_27_stablePrefix_: _dafny.Seq
                        d_27_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_28_constrainedPrompt_: _dafny.Seq
                        d_28_constrainedPrompt_ = (prompt) + (d_27_stablePrefix_)
                        d_29_validCount_: int
                        out19_: int
                        out19_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_29_validCount_ = out19_
                        d_30_nextIn_: _dafny.Seq
                        d_30_nextIn_ = eosToken
                        if (len(currentConstrainedOut)) == (0):
                            if (len(validTokenGroups)) > (0):
                                d_31_nextAdaptive0_: _dafny.Seq
                                out20_: _dafny.Seq
                                out20_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_31_nextAdaptive0_ = out20_
                                d_30_nextIn_ = d_31_nextAdaptive0_
                            elif True:
                                d_32_nextHard0_: _dafny.Seq
                                out21_: _dafny.Seq
                                out21_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_32_nextHard0_ = out21_
                                d_30_nextIn_ = d_32_nextHard0_
                        elif (d_29_validCount_) <= (8):
                            d_33_nextHard_: _dafny.Seq
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_33_nextHard_ = out22_
                            d_30_nextIn_ = d_33_nextHard_
                        elif (len(validTokenGroups)) > (0):
                            d_34_nextAdaptive_: _dafny.Seq
                            out23_: _dafny.Seq
                            out23_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_34_nextAdaptive_ = out23_
                            d_30_nextIn_ = d_34_nextAdaptive_
                        elif True:
                            d_35_nextRep_: _dafny.Seq
                            out24_: _dafny.Seq
                            out24_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_35_nextRep_ = out24_
                            d_30_nextIn_ = d_35_nextRep_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_30_nextIn_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_36_appendedGenerated_: _dafny.Seq
                            d_37_appendedInside_: bool
                            d_38_appendedCurrent_: _dafny.Seq
                            out25_: _dafny.Seq
                            out26_: bool
                            out27_: _dafny.Seq
                            out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_nextIn_)
                            d_36_appendedGenerated_ = out25_
                            d_37_appendedInside_ = out26_
                            d_38_appendedCurrent_ = out27_
                            generated = d_36_appendedGenerated_
                            insideConstrainedOut = d_37_appendedInside_
                            currentConstrainedOut = d_38_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

