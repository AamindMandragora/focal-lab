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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Every arithmetic computation must appear inside << >> delimiters, and the final computation should also be inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openArmed_: bool
        d_2_openArmed_ = False
        d_3_completedSpans_: int
        d_3_completedSpans_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (not(insideConstrainedOut)) and (d_2_openArmed_):
                        d_4_openedGenerated_: _dafny.Seq
                        d_5_openedInside_: bool
                        d_6_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_4_openedGenerated_ = out0_
                        d_5_openedInside_ = out1_
                        d_6_openedCurrent_ = out2_
                        generated = d_4_openedGenerated_
                        insideConstrainedOut = d_5_openedInside_
                        currentConstrainedOut = d_6_openedCurrent_
                        d_2_openArmed_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif not(insideConstrainedOut):
                        d_7_remaining_: int
                        d_7_remaining_ = (maxSteps) - (d_1_steps_)
                        d_8_chunkBudget_: int
                        if (d_7_remaining_) <= (4):
                            d_8_chunkBudget_ = d_7_remaining_
                        elif True:
                            d_8_chunkBudget_ = 4
                        d_9_chunkedG_: _dafny.Seq
                        d_10_stoppedOpen_: bool
                        d_11_stoppedEos_: bool
                        d_12_stepsUsed_: int
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: bool
                        out6_: int
                        out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_9_chunkedG_ = out3_
                        d_10_stoppedOpen_ = out4_
                        d_11_stoppedEos_ = out5_
                        d_12_stepsUsed_ = out6_
                        generated = d_9_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                        if d_11_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_10_stoppedOpen_:
                            d_13_enteredGenerated_: _dafny.Seq
                            d_14_enteredInside_: bool
                            d_15_enteredCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_13_enteredGenerated_ = out7_
                            d_14_enteredInside_ = out8_
                            d_15_enteredCurrent_ = out9_
                            generated = d_13_enteredGenerated_
                            insideConstrainedOut = d_14_enteredInside_
                            currentConstrainedOut = d_15_enteredCurrent_
                            d_2_openArmed_ = False
                        elif True:
                            d_16_sinceEq_: int
                            out10_: int
                            out10_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                            d_16_sinceEq_ = out10_
                            d_17_sincePlus_: int
                            out11_: int
                            out11_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")))
                            d_17_sincePlus_ = out11_
                            d_18_sinceMinus_: int
                            out12_: int
                            out12_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")))
                            d_18_sinceMinus_ = out12_
                            d_19_sinceStar_: int
                            out13_: int
                            out13_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")))
                            d_19_sinceStar_ = out13_
                            d_20_sinceSlash_: int
                            out14_: int
                            out14_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))
                            d_20_sinceSlash_ = out14_
                            d_21_sinceIs_: int
                            out15_: int
                            out15_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is")))
                            d_21_sinceIs_ = out15_
                            d_22_sinceAre_: int
                            out16_: int
                            out16_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "are")))
                            d_22_sinceAre_ = out16_
                            d_23_sinceTotal_: int
                            out17_: int
                            out17_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")))
                            d_23_sinceTotal_ = out17_
                            d_24_sinceLeft_: int
                            out18_: int
                            out18_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "left")))
                            d_24_sinceLeft_ = out18_
                            d_25_sinceCost_: int
                            out19_: int
                            out19_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "cost")))
                            d_25_sinceCost_ = out19_
                            d_26_sinceEach_: int
                            out20_: int
                            out20_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "each")))
                            d_26_sinceEach_ = out20_
                            if (((d_3_completedSpans_) == (0)) and ((len(generated)) >= ((len(generatedPrefix)) + (20)))) and ((((((((((((d_16_sinceEq_) <= (2)) or ((d_17_sincePlus_) <= (2))) or ((d_18_sinceMinus_) <= (2))) or ((d_19_sinceStar_) <= (2))) or ((d_20_sinceSlash_) <= (2))) or ((d_21_sinceIs_) <= (2))) or ((d_22_sinceAre_) <= (2))) or ((d_23_sinceTotal_) <= (2))) or ((d_24_sinceLeft_) <= (2))) or ((d_25_sinceCost_) <= (2))) or ((d_26_sinceEach_) <= (2))):
                                d_2_openArmed_ = True
                            elif ((d_3_completedSpans_) == (0)) and ((len(generated)) >= ((len(generatedPrefix)) + (36))):
                                d_2_openArmed_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_27_closedGenerated_: _dafny.Seq
                        d_28_closedInside_: bool
                        d_29_closedCurrent_: _dafny.Seq
                        out21_: _dafny.Seq
                        out22_: bool
                        out23_: _dafny.Seq
                        out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_27_closedGenerated_ = out21_
                        d_28_closedInside_ = out22_
                        d_29_closedCurrent_ = out23_
                        generated = d_27_closedGenerated_
                        insideConstrainedOut = d_28_closedInside_
                        currentConstrainedOut = d_29_closedCurrent_
                        d_3_completedSpans_ = (d_3_completedSpans_) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_30_stablePrefix_: _dafny.Seq
                        d_30_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_31_constrainedPrompt_: _dafny.Seq
                        d_31_constrainedPrompt_ = (prompt) + (d_30_stablePrefix_)
                        d_32_nextInside_: _dafny.Seq
                        d_32_nextInside_ = eosToken
                        if (len(currentConstrainedOut)) < (2):
                            out24_: _dafny.Seq
                            out24_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_31_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('8e0'), eosToken)
                            d_32_nextInside_ = out24_
                        elif True:
                            d_33_gatedNext_: _dafny.Seq
                            d_34_wasConstrained_: bool
                            out25_: _dafny.Seq
                            out26_: bool
                            out25_, out26_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_31_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_33_gatedNext_ = out25_
                            d_34_wasConstrained_ = out26_
                            d_32_nextInside_ = d_33_gatedNext_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_32_nextInside_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_35_appendedGenerated_: _dafny.Seq
                            d_36_appendedInside_: bool
                            d_37_appendedCurrent_: _dafny.Seq
                            out27_: _dafny.Seq
                            out28_: bool
                            out29_: _dafny.Seq
                            out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_nextInside_)
                            d_35_appendedGenerated_ = out27_
                            d_36_appendedInside_ = out28_
                            d_37_appendedCurrent_ = out29_
                            generated = d_35_appendedGenerated_
                            insideConstrainedOut = d_36_appendedInside_
                            currentConstrainedOut = d_37_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

