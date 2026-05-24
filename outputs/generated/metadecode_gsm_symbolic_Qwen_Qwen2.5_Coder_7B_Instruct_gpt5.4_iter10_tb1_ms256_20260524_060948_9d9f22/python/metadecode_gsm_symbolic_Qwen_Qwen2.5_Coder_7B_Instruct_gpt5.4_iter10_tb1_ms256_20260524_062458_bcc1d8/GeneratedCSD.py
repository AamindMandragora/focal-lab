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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Put every arithmetic computation inside << >>, and ensure the final computation is also inside << >>.")))
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
                        d_7_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_7_next_ = out3_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_7_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                            if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_8_enteredGenerated_: _dafny.Seq
                                d_9_enteredInside_: bool
                                d_10_enteredCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_8_enteredGenerated_ = out4_
                                d_9_enteredInside_ = out5_
                                d_10_enteredCurrent_ = out6_
                                generated = d_8_enteredGenerated_
                                insideConstrainedOut = d_9_enteredInside_
                                currentConstrainedOut = d_10_enteredCurrent_
                                d_2_openArmed_ = False
                            elif True:
                                d_11_sinceEq_: int
                                out7_: int
                                out7_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_11_sinceEq_ = out7_
                                d_12_sincePlus_: int
                                out8_: int
                                out8_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")))
                                d_12_sincePlus_ = out8_
                                d_13_sinceMinus_: int
                                out9_: int
                                out9_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")))
                                d_13_sinceMinus_ = out9_
                                d_14_sinceStar_: int
                                out10_: int
                                out10_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")))
                                d_14_sinceStar_ = out10_
                                d_15_sinceSlash_: int
                                out11_: int
                                out11_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))
                                d_15_sinceSlash_ = out11_
                                d_16_sinceIs_: int
                                out12_: int
                                out12_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is")))
                                d_16_sinceIs_ = out12_
                                d_17_sinceAre_: int
                                out13_: int
                                out13_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "are")))
                                d_17_sinceAre_ = out13_
                                d_18_sinceTotal_: int
                                out14_: int
                                out14_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")))
                                d_18_sinceTotal_ = out14_
                                d_19_sinceLeft_: int
                                out15_: int
                                out15_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "left")))
                                d_19_sinceLeft_ = out15_
                                d_20_sinceCost_: int
                                out16_: int
                                out16_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "cost")))
                                d_20_sinceCost_ = out16_
                                d_21_sinceEach_: int
                                out17_: int
                                out17_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "each")))
                                d_21_sinceEach_ = out17_
                                if (((((((((((d_11_sinceEq_) <= (2)) or ((d_12_sincePlus_) <= (2))) or ((d_13_sinceMinus_) <= (2))) or ((d_14_sinceStar_) <= (2))) or ((d_15_sinceSlash_) <= (2))) or ((d_16_sinceIs_) <= (2))) or ((d_17_sinceAre_) <= (2))) or ((d_18_sinceTotal_) <= (2))) or ((d_19_sinceLeft_) <= (2))) or ((d_20_sinceCost_) <= (2))) or ((d_21_sinceEach_) <= (2)):
                                    d_2_openArmed_ = True
                                elif ((d_3_completedSpans_) == (0)) and ((len(generated)) >= ((len(generatedPrefix)) + (24))):
                                    d_2_openArmed_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_22_closedGenerated_: _dafny.Seq
                        d_23_closedInside_: bool
                        d_24_closedCurrent_: _dafny.Seq
                        out18_: _dafny.Seq
                        out19_: bool
                        out20_: _dafny.Seq
                        out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_22_closedGenerated_ = out18_
                        d_23_closedInside_ = out19_
                        d_24_closedCurrent_ = out20_
                        generated = d_22_closedGenerated_
                        insideConstrainedOut = d_23_closedInside_
                        currentConstrainedOut = d_24_closedCurrent_
                        d_3_completedSpans_ = (d_3_completedSpans_) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_25_stablePrefix_: _dafny.Seq
                        d_25_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_26_constrainedPrompt_: _dafny.Seq
                        d_26_constrainedPrompt_ = (prompt) + (d_25_stablePrefix_)
                        d_27_nextInside_: _dafny.Seq
                        d_27_nextInside_ = eosToken
                        if (len(currentConstrainedOut)) < (2):
                            out21_: _dafny.Seq
                            out21_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('8e0'), eosToken)
                            d_27_nextInside_ = out21_
                        elif True:
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_27_nextInside_ = out22_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_27_nextInside_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_28_appendedGenerated_: _dafny.Seq
                            d_29_appendedInside_: bool
                            d_30_appendedCurrent_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: bool
                            out25_: _dafny.Seq
                            out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_nextInside_)
                            d_28_appendedGenerated_ = out23_
                            d_29_appendedInside_ = out24_
                            d_30_appendedCurrent_ = out25_
                            generated = d_28_appendedGenerated_
                            insideConstrainedOut = d_29_appendedInside_
                            currentConstrainedOut = d_30_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

