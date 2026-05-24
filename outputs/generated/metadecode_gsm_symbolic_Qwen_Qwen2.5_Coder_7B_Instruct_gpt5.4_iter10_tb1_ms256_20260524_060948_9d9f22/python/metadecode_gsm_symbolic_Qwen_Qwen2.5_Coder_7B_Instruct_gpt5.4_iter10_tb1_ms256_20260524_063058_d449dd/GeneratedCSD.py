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
        d_4_rollbackLimit_: int
        d_4_rollbackLimit_ = 24
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (not(insideConstrainedOut)) and (d_2_openArmed_):
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
                        d_2_openArmed_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif not(insideConstrainedOut):
                        d_8_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_8_next_ = out3_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_8_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                            if ((d_3_completedSpans_) == (0)) and ((len(generated)) >= ((len(generatedPrefix)) + (16))):
                                d_2_openArmed_ = True
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                            d_9_sinceEq_: int
                            out4_: int
                            out4_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                            d_9_sinceEq_ = out4_
                            d_10_sincePlus_: int
                            out5_: int
                            out5_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")))
                            d_10_sincePlus_ = out5_
                            d_11_sinceMinus_: int
                            out6_: int
                            out6_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")))
                            d_11_sinceMinus_ = out6_
                            d_12_sinceStar_: int
                            out7_: int
                            out7_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")))
                            d_12_sinceStar_ = out7_
                            d_13_sinceSlash_: int
                            out8_: int
                            out8_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))
                            d_13_sinceSlash_ = out8_
                            d_14_sinceIs_: int
                            out9_: int
                            out9_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is")))
                            d_14_sinceIs_ = out9_
                            d_15_sinceAre_: int
                            out10_: int
                            out10_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "are")))
                            d_15_sinceAre_ = out10_
                            d_16_sinceTotal_: int
                            out11_: int
                            out11_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")))
                            d_16_sinceTotal_ = out11_
                            d_17_sinceLeft_: int
                            out12_: int
                            out12_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "left")))
                            d_17_sinceLeft_ = out12_
                            d_18_sinceCost_: int
                            out13_: int
                            out13_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "cost")))
                            d_18_sinceCost_ = out13_
                            d_19_sinceEach_: int
                            out14_: int
                            out14_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "each")))
                            d_19_sinceEach_ = out14_
                            if ((d_3_completedSpans_) == (0)) and (((((((((((((len(generated)) >= ((len(generatedPrefix)) + (28))) or ((d_9_sinceEq_) <= (2))) or ((d_10_sincePlus_) <= (2))) or ((d_11_sinceMinus_) <= (2))) or ((d_12_sinceStar_) <= (2))) or ((d_13_sinceSlash_) <= (2))) or ((d_14_sinceIs_) <= (2))) or ((d_15_sinceAre_) <= (2))) or ((d_16_sinceTotal_) <= (2))) or ((d_17_sinceLeft_) <= (2))) or ((d_18_sinceCost_) <= (2))) or ((d_19_sinceEach_) <= (2))):
                                d_2_openArmed_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_20_closedGenerated_: _dafny.Seq
                        d_21_closedInside_: bool
                        d_22_closedCurrent_: _dafny.Seq
                        out15_: _dafny.Seq
                        out16_: bool
                        out17_: _dafny.Seq
                        out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_20_closedGenerated_ = out15_
                        d_21_closedInside_ = out16_
                        d_22_closedCurrent_ = out17_
                        generated = d_20_closedGenerated_
                        insideConstrainedOut = d_21_closedInside_
                        currentConstrainedOut = d_22_closedCurrent_
                        d_3_completedSpans_ = (d_3_completedSpans_) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_4_rollbackLimit_):
                        d_23_rolledGenerated_: _dafny.Seq
                        d_24_rolledCurrent_: _dafny.Seq
                        out18_: _dafny.Seq
                        out19_: _dafny.Seq
                        out18_, out19_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_23_rolledGenerated_ = out18_
                        d_24_rolledCurrent_ = out19_
                        generated = d_23_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_24_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_25_stablePrefix_: _dafny.Seq
                        d_25_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_26_constrainedPrompt_: _dafny.Seq
                        d_26_constrainedPrompt_ = (prompt) + (d_25_stablePrefix_)
                        d_27_validCount_: int
                        out20_: int
                        out20_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_27_validCount_ = out20_
                        d_28_nextInside_: _dafny.Seq
                        d_28_nextInside_ = eosToken
                        if (len(currentConstrainedOut)) < (2):
                            out21_: _dafny.Seq
                            out21_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('8e0'), eosToken)
                            d_28_nextInside_ = out21_
                        elif (d_27_validCount_) <= (8):
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('4e0'), 12, eosToken)
                            d_28_nextInside_ = out22_
                        elif (d_27_validCount_) <= (20):
                            out23_: _dafny.Seq
                            out23_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_28_nextInside_ = out23_
                        elif True:
                            out24_: _dafny.Seq
                            out24_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_28_nextInside_ = out24_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_28_nextInside_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_29_appendedGenerated_: _dafny.Seq
                            d_30_appendedInside_: bool
                            d_31_appendedCurrent_: _dafny.Seq
                            out25_: _dafny.Seq
                            out26_: bool
                            out27_: _dafny.Seq
                            out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_nextInside_)
                            d_29_appendedGenerated_ = out25_
                            d_30_appendedInside_ = out26_
                            d_31_appendedCurrent_ = out27_
                            generated = d_29_appendedGenerated_
                            insideConstrainedOut = d_30_appendedInside_
                            currentConstrainedOut = d_31_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

