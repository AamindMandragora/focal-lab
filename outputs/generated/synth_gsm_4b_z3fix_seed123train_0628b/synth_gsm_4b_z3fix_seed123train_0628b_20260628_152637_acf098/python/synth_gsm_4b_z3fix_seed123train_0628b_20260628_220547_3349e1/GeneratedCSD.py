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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the variable names from the problem (no curly braces). At the very end, write the final answer expression inside << >>. Use ONLY: variable names without braces, integers, +, -, *, /, //, %, int(), and parentheses. No LaTeX, no {braces}, no ** exponents, no text inside << >>. Write << >> exactly once at the end. Example: The answer is <<count * (n1 + n2 + n3)>>")))
        d_2_unconstrainedBudget_: int
        d_2_unconstrainedBudget_ = _dafny.euclidian_division((maxSteps) * (75), 100)
        if ((maxSteps) >= (45)) and ((d_2_unconstrainedBudget_) > ((maxSteps) - (45))):
            d_2_unconstrainedBudget_ = (maxSteps) - (45)
        if (d_2_unconstrainedBudget_) > (maxSteps):
            d_2_unconstrainedBudget_ = maxSteps
        d_3_gotGoodSpan_: bool
        d_3_gotGoodSpan_ = False
        with _dafny.label("0"):
            while (((d_1_steps_) < (d_2_unconstrainedBudget_)) and (not(insideConstrainedOut))) and (not(d_3_gotGoodSpan_)):
                with _dafny.c_label("0"):
                    d_4_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_4_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_4_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out1_
                            insideConstrainedOut = out2_
                            currentConstrainedOut = out3_
                            d_5_spanFillBudget_: int
                            d_5_spanFillBudget_ = (maxSteps) - (d_1_steps_)
                            if (d_5_spanFillBudget_) > (35):
                                d_5_spanFillBudget_ = 35
                            d_6_spanSteps_: int
                            d_6_spanSteps_ = 0
                            with _dafny.label("2_1_0_0"):
                                while (((d_6_spanSteps_) < (d_5_spanFillBudget_)) and (insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    with _dafny.c_label("2_1_0_0"):
                                        d_7_cg_: _dafny.Seq
                                        d_8_ci_: bool
                                        d_9_cc_: _dafny.Seq
                                        d_10_closed_: bool
                                        out4_: _dafny.Seq
                                        out5_: bool
                                        out6_: _dafny.Seq
                                        out7_: bool
                                        out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                        d_7_cg_ = out4_
                                        d_8_ci_ = out5_
                                        d_9_cc_ = out6_
                                        d_10_closed_ = out7_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_6_spanSteps_ = (d_6_spanSteps_) + (1)
                                        if d_10_closed_:
                                            generated = d_7_cg_
                                            insideConstrainedOut = d_8_ci_
                                            currentConstrainedOut = d_9_cc_
                                        elif True:
                                            d_11_constrainedPrompt_: _dafny.Seq
                                            d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                            d_12_nextC_: _dafny.Seq
                                            d_13_usedFallback_: bool
                                            out8_: _dafny.Seq
                                            out9_: bool
                                            out8_, out9_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                                            d_12_nextC_ = out8_
                                            d_13_usedFallback_ = out9_
                                            if (d_12_nextC_) == (eosToken):
                                                raise _dafny.Break("2_1_0_0")
                                            elif True:
                                                d_14_ag_: _dafny.Seq
                                                d_15_ai_: bool
                                                d_16_ac_: _dafny.Seq
                                                out10_: _dafny.Seq
                                                out11_: bool
                                                out12_: _dafny.Seq
                                                out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_nextC_)
                                                d_14_ag_ = out10_
                                                d_15_ai_ = out11_
                                                d_16_ac_ = out12_
                                                generated = d_14_ag_
                                                insideConstrainedOut = d_15_ai_
                                                currentConstrainedOut = d_16_ac_
                                        pass
                                pass
                            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                                d_17_closeBudget1_: int
                                d_17_closeBudget1_ = (maxSteps) - (d_1_steps_)
                                if (d_17_closeBudget1_) > (20):
                                    d_17_closeBudget1_ = 20
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget1_)
                                generated = out13_
                                insideConstrainedOut = out14_
                                currentConstrainedOut = out15_
                                d_1_steps_ = (d_1_steps_) + (d_17_closeBudget1_)
                                if (d_1_steps_) > (maxSteps):
                                    d_1_steps_ = maxSteps
                            if not(insideConstrainedOut):
                                d_3_gotGoodSpan_ = True
                            raise _dafny.Break("0")
                    pass
            pass
        if ((not(insideConstrainedOut)) and (not(d_3_gotGoodSpan_))) and ((d_1_steps_) < (maxSteps)):
            d_18_remaining_: int
            d_18_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_18_remaining_) >= (5):
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                generated = out16_
                insideConstrainedOut = out17_
                currentConstrainedOut = out18_
                d_1_steps_ = (d_1_steps_) + (1)
                d_19_fillBudget_: int
                d_19_fillBudget_ = (maxSteps) - (d_1_steps_)
                if (d_19_fillBudget_) > (40):
                    d_19_fillBudget_ = 40
                d_20_fillSteps_: int
                d_20_fillSteps_ = 0
                with _dafny.label("3_0_0"):
                    while (((d_20_fillSteps_) < (d_19_fillBudget_)) and (insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                        with _dafny.c_label("3_0_0"):
                            d_21_cg3_: _dafny.Seq
                            d_22_ci3_: bool
                            d_23_cc3_: _dafny.Seq
                            d_24_closed3_: bool
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out22_: bool
                            out19_, out20_, out21_, out22_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_21_cg3_ = out19_
                            d_22_ci3_ = out20_
                            d_23_cc3_ = out21_
                            d_24_closed3_ = out22_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_20_fillSteps_ = (d_20_fillSteps_) + (1)
                            if d_24_closed3_:
                                generated = d_21_cg3_
                                insideConstrainedOut = d_22_ci3_
                                currentConstrainedOut = d_23_cc3_
                            elif True:
                                d_25_constrainedPrompt3_: _dafny.Seq
                                d_25_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_26_next3_: _dafny.Seq
                                d_27_usedFallback3_: bool
                                out23_: _dafny.Seq
                                out24_: bool
                                out23_, out24_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_25_constrainedPrompt3_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                                d_26_next3_ = out23_
                                d_27_usedFallback3_ = out24_
                                if (d_26_next3_) == (eosToken):
                                    raise _dafny.Break("3_0_0")
                                elif True:
                                    d_28_ag3_: _dafny.Seq
                                    d_29_ai3_: bool
                                    d_30_ac3_: _dafny.Seq
                                    out25_: _dafny.Seq
                                    out26_: bool
                                    out27_: _dafny.Seq
                                    out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next3_)
                                    d_28_ag3_ = out25_
                                    d_29_ai3_ = out26_
                                    d_30_ac3_ = out27_
                                    generated = d_28_ag3_
                                    insideConstrainedOut = d_29_ai3_
                                    currentConstrainedOut = d_30_ac3_
                            pass
                    pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_31_closeBudget_: int
            d_31_closeBudget_ = (maxSteps) - (d_1_steps_)
            out28_: _dafny.Seq
            out29_: bool
            out30_: _dafny.Seq
            out28_, out29_, out30_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_31_closeBudget_)
            generated = out28_
            insideConstrainedOut = out29_
            currentConstrainedOut = out30_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

