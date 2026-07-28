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
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. At the end, write your final arithmetic expression inside << and >>. Use only plain variable names (no curly braces), numbers, +, -, *, /, //, %, (, ), int(). Example: <<n * price + 5>>. Write exactly one << >> expression and stop after >>."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_spanCompleted_: bool
            d_3_spanCompleted_ = False
            d_4_prefixBudget_: int
            d_4_prefixBudget_ = _dafny.euclidian_division((maxSteps) * (80), 100)
            if (d_4_prefixBudget_) == (0):
                d_4_prefixBudget_ = 1
            if (d_4_prefixBudget_) >= (maxSteps):
                d_4_prefixBudget_ = (maxSteps) - (1)
            with _dafny.label("1_0"):
                while ((d_2_steps_) < (d_4_prefixBudget_)) and (not(d_3_spanCompleted_)):
                    with _dafny.c_label("1_0"):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if ((d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (not(insideConstrainedOut)):
                                d_6_g2_: _dafny.Seq
                                d_7_ic2_: bool
                                d_8_cc2_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_6_g2_ = out1_
                                d_7_ic2_ = out2_
                                d_8_cc2_ = out3_
                                generated = d_6_g2_
                                insideConstrainedOut = d_7_ic2_
                                currentConstrainedOut = d_8_cc2_
                            elif ((d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) and (insideConstrainedOut):
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_spanCompleted_ = True
                        pass
                pass
            if not(d_3_spanCompleted_):
                if insideConstrainedOut:
                    if (d_2_steps_) < (maxSteps):
                        d_9_closeBudget_: int
                        d_9_closeBudget_ = (maxSteps) - (d_2_steps_)
                        d_10_cg_: _dafny.Seq
                        d_11_ci_: bool
                        d_12_cc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_closeBudget_)
                        d_10_cg_ = out4_
                        d_11_ci_ = out5_
                        d_12_cc_ = out6_
                        generated = d_10_cg_
                        insideConstrainedOut = d_11_ci_
                        currentConstrainedOut = d_12_cc_
                        d_2_steps_ = maxSteps
                elif True:
                    if (d_2_steps_) < (maxSteps):
                        d_13_g2_: _dafny.Seq
                        d_14_ic2_: bool
                        d_15_cc2_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_13_g2_ = out7_
                        d_14_ic2_ = out8_
                        d_15_cc2_ = out9_
                        generated = d_13_g2_
                        insideConstrainedOut = d_14_ic2_
                        currentConstrainedOut = d_15_cc2_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_16_reservedClose_: int
                        d_16_reservedClose_ = _dafny.euclidian_division(maxSteps, 10)
                        if (d_16_reservedClose_) < (10):
                            d_16_reservedClose_ = 10
                        if (d_16_reservedClose_) >= (maxSteps):
                            d_16_reservedClose_ = _dafny.euclidian_division(maxSteps, 2)
                        if (insideConstrainedOut) and ((maxSteps) > ((d_2_steps_) + (d_16_reservedClose_))):
                            d_17_innerBudget_: int
                            d_17_innerBudget_ = ((maxSteps) - (d_2_steps_)) - (d_16_reservedClose_)
                            d_18_innerSteps_: int
                            d_18_innerSteps_ = 0
                            with _dafny.label("1_3_1_0_2_0"):
                                while ((d_18_innerSteps_) < (d_17_innerBudget_)) and (insideConstrainedOut):
                                    with _dafny.c_label("1_3_1_0_2_0"):
                                        d_19_constrainedPrompt_: _dafny.Seq
                                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                        d_20_next_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                        d_20_next_ = out10_
                                        d_18_innerSteps_ = (d_18_innerSteps_) + (1)
                                        if (d_20_next_) == (eosToken):
                                            raise _dafny.Break("1_3_1_0_2_0")
                                        elif True:
                                            d_21_ag_: _dafny.Seq
                                            d_22_ai_: bool
                                            d_23_ac_: _dafny.Seq
                                            out11_: _dafny.Seq
                                            out12_: bool
                                            out13_: _dafny.Seq
                                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                            d_21_ag_ = out11_
                                            d_22_ai_ = out12_
                                            d_23_ac_ = out13_
                                            generated = d_21_ag_
                                            insideConstrainedOut = d_22_ai_
                                            currentConstrainedOut = d_23_ac_
                                        pass
                                pass
                            d_2_steps_ = (d_2_steps_) + (d_18_innerSteps_)
                        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                            d_24_closeBudget_: int
                            d_24_closeBudget_ = (maxSteps) - (d_2_steps_)
                            d_25_cg_: _dafny.Seq
                            d_26_ci_: bool
                            d_27_cc_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_closeBudget_)
                            d_25_cg_ = out14_
                            d_26_ci_ = out15_
                            d_27_cc_ = out16_
                            generated = d_25_cg_
                            insideConstrainedOut = d_26_ci_
                            currentConstrainedOut = d_27_cc_
                            d_2_steps_ = maxSteps
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

