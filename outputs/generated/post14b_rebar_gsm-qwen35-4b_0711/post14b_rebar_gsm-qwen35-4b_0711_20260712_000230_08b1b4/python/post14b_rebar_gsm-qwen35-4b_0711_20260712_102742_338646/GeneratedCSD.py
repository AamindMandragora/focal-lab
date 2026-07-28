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
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the given variable names directly (no curly braces, no LaTeX, no $, no **). Write exactly ONE final arithmetic expression enclosed in << >>. Stop immediately after >>. The expression must use only: plain variable names (like n, price, count), numbers, +, -, *, /, //, %, (, ), int(). Example: <<n * price + 5>>."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_prefixBudget_: int
            d_3_prefixBudget_ = _dafny.euclidian_division((maxSteps) * (55), 100)
            if (d_3_prefixBudget_) == (0):
                d_3_prefixBudget_ = 1
            if (d_3_prefixBudget_) >= (maxSteps):
                d_3_prefixBudget_ = (maxSteps) - (1)
            with _dafny.label("1_0"):
                while ((d_2_steps_) < (d_3_prefixBudget_)) and (not(insideConstrainedOut)):
                    with _dafny.c_label("1_0"):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_5_g2_: _dafny.Seq
                                d_6_ic2_: bool
                                d_7_cc2_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_5_g2_ = out1_
                                d_6_ic2_ = out2_
                                d_7_cc2_ = out3_
                                generated = d_5_g2_
                                insideConstrainedOut = d_6_ic2_
                                currentConstrainedOut = d_7_cc2_
                        pass
                pass
            if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                d_8_g2_: _dafny.Seq
                d_9_ic2_: bool
                d_10_cc2_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_8_g2_ = out4_
                d_9_ic2_ = out5_
                d_10_cc2_ = out6_
                generated = d_8_g2_
                insideConstrainedOut = d_9_ic2_
                currentConstrainedOut = d_10_cc2_
                d_2_steps_ = (d_2_steps_) + (1)
            d_11_reservedClose_: int
            d_11_reservedClose_ = _dafny.euclidian_division(maxSteps, 20)
            if (d_11_reservedClose_) < (15):
                d_11_reservedClose_ = 15
            if (d_11_reservedClose_) >= (maxSteps):
                d_11_reservedClose_ = _dafny.euclidian_division(maxSteps, 2)
            if (insideConstrainedOut) and ((maxSteps) > ((d_2_steps_) + (d_11_reservedClose_))):
                d_12_innerBudget_: int
                d_12_innerBudget_ = ((maxSteps) - (d_2_steps_)) - (d_11_reservedClose_)
                d_13_innerSteps_: int
                d_13_innerSteps_ = 0
                with _dafny.label("1_6_0"):
                    while ((d_13_innerSteps_) < (d_12_innerBudget_)) and (insideConstrainedOut):
                        with _dafny.c_label("1_6_0"):
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_15_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('7e-1'), eosToken)
                            d_15_next_ = out7_
                            d_13_innerSteps_ = (d_13_innerSteps_) + (1)
                            if (d_15_next_) == (eosToken):
                                raise _dafny.Break("1_6_0")
                            elif True:
                                d_16_ag_: _dafny.Seq
                                d_17_ai_: bool
                                d_18_ac_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                d_16_ag_ = out8_
                                d_17_ai_ = out9_
                                d_18_ac_ = out10_
                                generated = d_16_ag_
                                insideConstrainedOut = d_17_ai_
                                currentConstrainedOut = d_18_ac_
                            pass
                    pass
                d_2_steps_ = (d_2_steps_) + (d_13_innerSteps_)
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_19_closeBudget_: int
                d_19_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_20_cg_: _dafny.Seq
                d_21_ci_: bool
                d_22_cc_: _dafny.Seq
                out11_: _dafny.Seq
                out12_: bool
                out13_: _dafny.Seq
                out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget_)
                d_20_cg_ = out11_
                d_21_ci_ = out12_
                d_22_cc_ = out13_
                generated = d_20_cg_
                insideConstrainedOut = d_21_ci_
                currentConstrainedOut = d_22_cc_
                d_2_steps_ = maxSteps
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

