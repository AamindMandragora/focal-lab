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
                        d_7_stableGenerated_: _dafny.Seq
                        d_7_stableGenerated_ = generated
                        (lm).GenerateLogits((prompt) + (d_7_stableGenerated_))
                        (d_0_helpers_).SafePenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('12e0'))
                        d_8_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (lm).ChooseNextTokenUnconstrained()
                        d_8_next_ = out3_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_8_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                            if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_9_enteredGenerated_: _dafny.Seq
                                d_10_enteredInside_: bool
                                d_11_enteredCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_9_enteredGenerated_ = out4_
                                d_10_enteredInside_ = out5_
                                d_11_enteredCurrent_ = out6_
                                generated = d_9_enteredGenerated_
                                insideConstrainedOut = d_10_enteredInside_
                                currentConstrainedOut = d_11_enteredCurrent_
                                d_2_openArmed_ = False
                            elif True:
                                d_12_sinceEq_: int
                                out7_: int
                                out7_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_12_sinceEq_ = out7_
                                d_13_sincePlus_: int
                                out8_: int
                                out8_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")))
                                d_13_sincePlus_ = out8_
                                d_14_sinceMinus_: int
                                out9_: int
                                out9_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")))
                                d_14_sinceMinus_ = out9_
                                d_15_sinceStar_: int
                                out10_: int
                                out10_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")))
                                d_15_sinceStar_ = out10_
                                d_16_sinceSlash_: int
                                out11_: int
                                out11_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")))
                                d_16_sinceSlash_ = out11_
                                d_17_sinceTimes_: int
                                out12_: int
                                out12_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "times")))
                                d_17_sinceTimes_ = out12_
                                d_18_sinceEach_: int
                                out13_: int
                                out13_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "each")))
                                d_18_sinceEach_ = out13_
                                d_19_sinceTotal_: int
                                out14_: int
                                out14_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")))
                                d_19_sinceTotal_ = out14_
                                d_20_sinceLeft_: int
                                out15_: int
                                out15_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "left")))
                                d_20_sinceLeft_ = out15_
                                d_21_sinceNow_: int
                                out16_: int
                                out16_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "now")))
                                d_21_sinceNow_ = out16_
                                if ((((((((((d_12_sinceEq_) <= (1)) or ((d_13_sincePlus_) <= (1))) or ((d_14_sinceMinus_) <= (1))) or ((d_15_sinceStar_) <= (1))) or ((d_16_sinceSlash_) <= (1))) or ((d_17_sinceTimes_) <= (2))) or ((d_18_sinceEach_) <= (2))) or ((d_19_sinceTotal_) <= (2))) or ((d_20_sinceLeft_) <= (2))) or ((d_21_sinceNow_) <= (2)):
                                    d_2_openArmed_ = True
                                elif ((d_3_completedSpans_) == (0)) and ((len(generated)) >= ((len(generatedPrefix)) + (24))):
                                    d_2_openArmed_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_22_closedGenerated_: _dafny.Seq
                        d_23_closedInside_: bool
                        d_24_closedCurrent_: _dafny.Seq
                        out17_: _dafny.Seq
                        out18_: bool
                        out19_: _dafny.Seq
                        out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_22_closedGenerated_ = out17_
                        d_23_closedInside_ = out18_
                        d_24_closedCurrent_ = out19_
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
                        d_27_validCount_: int
                        out20_: int
                        out20_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_27_validCount_ = out20_
                        d_28_nextConstrained_: _dafny.Seq
                        d_28_nextConstrained_ = eosToken
                        if (d_27_validCount_) <= (10):
                            out21_: _dafny.Seq
                            out21_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_28_nextConstrained_ = out21_
                        elif True:
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), eosToken)
                            d_28_nextConstrained_ = out22_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_28_nextConstrained_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_29_appendedGenerated_: _dafny.Seq
                            d_30_appendedInside_: bool
                            d_31_appendedCurrent_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: bool
                            out25_: _dafny.Seq
                            out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_nextConstrained_)
                            d_29_appendedGenerated_ = out23_
                            d_30_appendedInside_ = out24_
                            d_31_appendedCurrent_ = out25_
                            generated = d_29_appendedGenerated_
                            insideConstrainedOut = d_30_appendedInside_
                            currentConstrainedOut = d_31_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

