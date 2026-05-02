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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedAnySpan_: bool
        d_2_openedAnySpan_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_openedAnySpan_:
                            raise _dafny.Break("0")
                        elif True:
                            d_3_openedGenerated_: _dafny.Seq
                            d_4_openedInside_: bool
                            d_5_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_3_openedGenerated_ = out0_
                            d_4_openedInside_ = out1_
                            d_5_openedCurrent_ = out2_
                            generated = d_3_openedGenerated_
                            insideConstrainedOut = d_4_openedInside_
                            currentConstrainedOut = d_5_openedCurrent_
                            d_2_openedAnySpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_6_complete_: bool
                        d_6_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_complete_:
                            d_7_closedGenerated_: _dafny.Seq
                            d_8_closedInside_: bool
                            d_9_closedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_closedGenerated_ = out3_
                            d_8_closedInside_ = out4_
                            d_9_closedCurrent_ = out5_
                            generated = d_7_closedGenerated_
                            insideConstrainedOut = d_8_closedInside_
                            currentConstrainedOut = d_9_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_10_narrow_: bool
                            out6_: bool
                            out6_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_10_narrow_ = out6_
                            d_11_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_11_validCount_ = out7_
                            d_12_stablePrefix_: _dafny.Seq
                            d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                            if (d_10_narrow_) or ((d_11_validCount_) <= (4)):
                                (lm).GenerateLogits((d_13_constrainedPrompt_) + (currentConstrainedOut))
                                d_14_candidates_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                                d_14_candidates_ = out8_
                                (d_0_helpers_).BoostTokenLogits(lm, d_14_candidates_, _dafny.BigRational('12e0'))
                                d_15_chosen_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).GetHighestLogitToken(lm)
                                d_15_chosen_ = out9_
                                d_16_chosenValid_: bool
                                out10_: bool
                                out10_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_15_chosen_)
                                d_16_chosenValid_ = out10_
                                if d_16_chosenValid_:
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    if (d_15_chosen_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_17_appendedGenerated1_: _dafny.Seq
                                        d_18_appendedInside1_: bool
                                        d_19_appendedCurrent1_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: bool
                                        out13_: _dafny.Seq
                                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_chosen_)
                                        d_17_appendedGenerated1_ = out11_
                                        d_18_appendedInside1_ = out12_
                                        d_19_appendedCurrent1_ = out13_
                                        generated = d_17_appendedGenerated1_
                                        insideConstrainedOut = d_18_appendedInside1_
                                        currentConstrainedOut = d_19_appendedCurrent1_
                                elif True:
                                    d_20_next1_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_20_next1_ = out14_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_20_next1_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_21_appendedGenerated1b_: _dafny.Seq
                                        d_22_appendedInside1b_: bool
                                        d_23_appendedCurrent1b_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next1_)
                                        d_21_appendedGenerated1b_ = out15_
                                        d_22_appendedInside1b_ = out16_
                                        d_23_appendedCurrent1b_ = out17_
                                        generated = d_21_appendedGenerated1b_
                                        insideConstrainedOut = d_22_appendedInside1b_
                                        currentConstrainedOut = d_23_appendedCurrent1b_
                            elif True:
                                (lm).GenerateLogits((d_13_constrainedPrompt_) + (currentConstrainedOut))
                                d_24_candidates2_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 6, eosToken)
                                d_24_candidates2_ = out18_
                                (d_0_helpers_).BoostTokenLogits(lm, d_24_candidates2_, _dafny.BigRational('6e0'))
                                d_25_topBiased_: _dafny.Seq
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).GetHighestLogitToken(lm)
                                d_25_topBiased_ = out19_
                                d_26_topValidBiased_: bool
                                out20_: bool
                                out20_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_25_topBiased_)
                                d_26_topValidBiased_ = out20_
                                if d_26_topValidBiased_:
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    if (d_25_topBiased_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_27_appendedGenerated2_: _dafny.Seq
                                        d_28_appendedInside2_: bool
                                        d_29_appendedCurrent2_: _dafny.Seq
                                        out21_: _dafny.Seq
                                        out22_: bool
                                        out23_: _dafny.Seq
                                        out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_topBiased_)
                                        d_27_appendedGenerated2_ = out21_
                                        d_28_appendedInside2_ = out22_
                                        d_29_appendedCurrent2_ = out23_
                                        generated = d_27_appendedGenerated2_
                                        insideConstrainedOut = d_28_appendedInside2_
                                        currentConstrainedOut = d_29_appendedCurrent2_
                                elif True:
                                    d_30_next_: _dafny.Seq
                                    out24_: _dafny.Seq
                                    out24_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_30_next_ = out24_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_30_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_31_appendedGenerated3_: _dafny.Seq
                                        d_32_appendedInside3_: bool
                                        d_33_appendedCurrent3_: _dafny.Seq
                                        out25_: _dafny.Seq
                                        out26_: bool
                                        out27_: _dafny.Seq
                                        out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                                        d_31_appendedGenerated3_ = out25_
                                        d_32_appendedInside3_ = out26_
                                        d_33_appendedCurrent3_ = out27_
                                        generated = d_31_appendedGenerated3_
                                        insideConstrainedOut = d_32_appendedInside3_
                                        currentConstrainedOut = d_33_appendedCurrent3_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

