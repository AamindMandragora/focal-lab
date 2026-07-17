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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_guidance_: _dafny.Seq
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. Use plain variable names from the problem (no curly braces, no currency prefix). After your reasoning, write the final numeric expression inside << >> using only plain variable names and +, -, *, /. Example: <<n * price - discount>>. No curly braces inside << >>."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_reservedBudget_: int
            if (maxSteps) >= (80):
                d_3_reservedBudget_ = 80
            elif True:
                d_3_reservedBudget_ = maxSteps
            d_4_freeReasoningBudget_: int
            d_4_freeReasoningBudget_ = (maxSteps) - (d_3_reservedBudget_)
            with _dafny.label("1_0"):
                while ((d_2_steps_) < (d_4_freeReasoningBudget_)) and (not(insideConstrainedOut)):
                    with _dafny.c_label("1_0"):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        pass
                pass
            if not(insideConstrainedOut):
                d_6_extraBudget_: int
                if (d_3_reservedBudget_) >= (30):
                    d_6_extraBudget_ = (d_3_reservedBudget_) - (30)
                elif True:
                    d_6_extraBudget_ = 0
                d_7_extraSteps_: int
                d_7_extraSteps_ = 0
                with _dafny.label("1_1_0"):
                    while (((d_7_extraSteps_) < (d_6_extraBudget_)) and (not(insideConstrainedOut))) and ((d_2_steps_) < (maxSteps)):
                        with _dafny.c_label("1_1_0"):
                            d_8_next_: _dafny.Seq
                            out1_: _dafny.Seq
                            out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out1_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_7_extraSteps_ = (d_7_extraSteps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("1_1_0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                            if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            pass
                    pass
            if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                d_9_og_: _dafny.Seq
                d_10_oi_: bool
                d_11_oc_: _dafny.Seq
                out2_: _dafny.Seq
                out3_: bool
                out4_: _dafny.Seq
                out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_9_og_ = out2_
                d_10_oi_ = out3_
                d_11_oc_ = out4_
                generated = d_9_og_
                insideConstrainedOut = d_10_oi_
                currentConstrainedOut = d_11_oc_
                d_2_steps_ = (d_2_steps_) + (1)
            with _dafny.label("1_1"):
                while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                    with _dafny.c_label("1_1"):
                        d_12_remainingBudget_: int
                        d_12_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        if (d_12_remainingBudget_) <= (5):
                            d_13_cg_: _dafny.Seq
                            d_14_ci_: bool
                            d_15_cc_: _dafny.Seq
                            d_16_closed_: bool
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out8_: bool
                            out5_, out6_, out7_, out8_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_13_cg_ = out5_
                            d_14_ci_ = out6_
                            d_15_cc_ = out7_
                            d_16_closed_ = out8_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_16_closed_:
                                generated = d_13_cg_
                                insideConstrainedOut = d_14_ci_
                                currentConstrainedOut = d_15_cc_
                            elif True:
                                raise _dafny.Break("1_1")
                        elif True:
                            d_17_currentLen_: int
                            d_17_currentLen_ = len(currentConstrainedOut)
                            if (d_17_currentLen_) >= (3):
                                d_18_cg_: _dafny.Seq
                                d_19_ci_: bool
                                d_20_cc_: _dafny.Seq
                                d_21_closed_: bool
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out12_: bool
                                out9_, out10_, out11_, out12_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_18_cg_ = out9_
                                d_19_ci_ = out10_
                                d_20_cc_ = out11_
                                d_21_closed_ = out12_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if d_21_closed_:
                                    generated = d_18_cg_
                                    insideConstrainedOut = d_19_ci_
                                    currentConstrainedOut = d_20_cc_
                                elif True:
                                    d_22_constrainedPrompt_: _dafny.Seq
                                    d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_23_next_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                                    d_23_next_ = out13_
                                    if (d_23_next_) == (eosToken):
                                        raise _dafny.Break("1_1")
                                    elif True:
                                        d_24_appendedGenerated_: _dafny.Seq
                                        d_25_appendedInside_: bool
                                        d_26_appendedCurrent_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                        d_24_appendedGenerated_ = out14_
                                        d_25_appendedInside_ = out15_
                                        d_26_appendedCurrent_ = out16_
                                        generated = d_24_appendedGenerated_
                                        insideConstrainedOut = d_25_appendedInside_
                                        currentConstrainedOut = d_26_appendedCurrent_
                            elif True:
                                d_27_constrainedPrompt_: _dafny.Seq
                                d_27_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_28_next_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_28_next_ = out17_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_28_next_) == (eosToken):
                                    raise _dafny.Break("1_1")
                                elif True:
                                    d_29_appendedGenerated_: _dafny.Seq
                                    d_30_appendedInside_: bool
                                    d_31_appendedCurrent_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                                    d_29_appendedGenerated_ = out18_
                                    d_30_appendedInside_ = out19_
                                    d_31_appendedCurrent_ = out20_
                                    generated = d_29_appendedGenerated_
                                    insideConstrainedOut = d_30_appendedInside_
                                    currentConstrainedOut = d_31_appendedCurrent_
                        pass
                pass
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_32_closeBudget_: int
                d_32_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_33_cg_: _dafny.Seq
                d_34_ci_: bool
                d_35_cc_: _dafny.Seq
                out21_: _dafny.Seq
                out22_: bool
                out23_: _dafny.Seq
                out21_, out22_, out23_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_closeBudget_)
                d_33_cg_ = out21_
                d_34_ci_ = out22_
                d_35_cc_ = out23_
                generated = d_33_cg_
                insideConstrainedOut = d_34_ci_
                currentConstrainedOut = d_35_cc_
                d_2_steps_ = maxSteps
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

