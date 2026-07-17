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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the very end, write the final answer as a single expression inside << >>. Use only: variable names (no braces), integers, +, -, *, /, //, %, int(), parentheses. No LaTeX, no {braces}, no ** exponents, no words inside << >>. Open << >> only once for the final answer. Example: <<int(n * price + base)>> or <<total - n1 * c1>>")))
        d_2_unconstrainedBudget_: int
        d_2_unconstrainedBudget_ = _dafny.euclidian_division((maxSteps) * (82), 100)
        if ((maxSteps) >= (50)) and ((d_2_unconstrainedBudget_) > ((maxSteps) - (50))):
            d_2_unconstrainedBudget_ = (maxSteps) - (50)
        if (d_2_unconstrainedBudget_) > (maxSteps):
            d_2_unconstrainedBudget_ = maxSteps
        d_3_spanClosed_: bool
        d_3_spanClosed_ = False
        with _dafny.label("0"):
            while (((d_1_steps_) < (d_2_unconstrainedBudget_)) and (not(insideConstrainedOut))) and (not(d_3_spanClosed_)):
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
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_5_minTokensInSpan_: int
            d_5_minTokensInSpan_ = 3
            d_6_tokensFilled_: int
            d_6_tokensFilled_ = 0
            d_7_spanFillBudget_: int
            d_7_spanFillBudget_ = (maxSteps) - (d_1_steps_)
            if (d_7_spanFillBudget_) > (35):
                d_7_spanFillBudget_ = 35
            d_8_fillSteps_: int
            d_8_fillSteps_ = 0
            with _dafny.label("3_0"):
                while (((d_8_fillSteps_) < (d_7_spanFillBudget_)) and (insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                    with _dafny.c_label("3_0"):
                        if (d_6_tokensFilled_) >= (d_5_minTokensInSpan_):
                            d_9_cg_: _dafny.Seq
                            d_10_ci_: bool
                            d_11_cc_: _dafny.Seq
                            d_12_closed_: bool
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out7_: bool
                            out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_9_cg_ = out4_
                            d_10_ci_ = out5_
                            d_11_cc_ = out6_
                            d_12_closed_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_8_fillSteps_ = (d_8_fillSteps_) + (1)
                            if d_12_closed_:
                                generated = d_9_cg_
                                insideConstrainedOut = d_10_ci_
                                currentConstrainedOut = d_11_cc_
                                d_3_spanClosed_ = True
                            elif True:
                                if (insideConstrainedOut) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                    d_13_constrainedPrompt_: _dafny.Seq
                                    d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_14_next_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_14_next_ = out8_
                                    if (d_14_next_) == (eosToken):
                                        raise _dafny.Break("3_0")
                                    elif True:
                                        d_15_ag_: _dafny.Seq
                                        d_16_ai_: bool
                                        d_17_ac_: _dafny.Seq
                                        out9_: _dafny.Seq
                                        out10_: bool
                                        out11_: _dafny.Seq
                                        out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                        d_15_ag_ = out9_
                                        d_16_ai_ = out10_
                                        d_17_ac_ = out11_
                                        generated = d_15_ag_
                                        insideConstrainedOut = d_16_ai_
                                        currentConstrainedOut = d_17_ac_
                                        d_6_tokensFilled_ = (d_6_tokensFilled_) + (1)
                        elif True:
                            if (insideConstrainedOut) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                d_18_constrainedPrompt_: _dafny.Seq
                                d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_19_next_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_19_next_ = out12_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_8_fillSteps_ = (d_8_fillSteps_) + (1)
                                if (d_19_next_) == (eosToken):
                                    raise _dafny.Break("3_0")
                                elif True:
                                    d_20_ag_: _dafny.Seq
                                    d_21_ai_: bool
                                    d_22_ac_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                    d_20_ag_ = out13_
                                    d_21_ai_ = out14_
                                    d_22_ac_ = out15_
                                    generated = d_20_ag_
                                    insideConstrainedOut = d_21_ai_
                                    currentConstrainedOut = d_22_ac_
                                    d_6_tokensFilled_ = (d_6_tokensFilled_) + (1)
                            elif True:
                                d_23_cg2_: _dafny.Seq
                                d_24_ci2_: bool
                                d_25_cc2_: _dafny.Seq
                                d_26_closed2_: bool
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out19_: bool
                                out16_, out17_, out18_, out19_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_23_cg2_ = out16_
                                d_24_ci2_ = out17_
                                d_25_cc2_ = out18_
                                d_26_closed2_ = out19_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_8_fillSteps_ = (d_8_fillSteps_) + (1)
                                if d_26_closed2_:
                                    generated = d_23_cg2_
                                    insideConstrainedOut = d_24_ci2_
                                    currentConstrainedOut = d_25_cc2_
                                    d_3_spanClosed_ = True
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_27_closeBudget1_: int
                d_27_closeBudget1_ = (maxSteps) - (d_1_steps_)
                if (d_27_closeBudget1_) > (20):
                    d_27_closeBudget1_ = 20
                out20_: _dafny.Seq
                out21_: bool
                out22_: _dafny.Seq
                out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudget1_)
                generated = out20_
                insideConstrainedOut = out21_
                currentConstrainedOut = out22_
                d_1_steps_ = (d_1_steps_) + (d_27_closeBudget1_)
                if (d_1_steps_) > (maxSteps):
                    d_1_steps_ = maxSteps
                if not(insideConstrainedOut):
                    d_3_spanClosed_ = True
        if ((not(insideConstrainedOut)) and (not(d_3_spanClosed_))) and ((d_1_steps_) < (maxSteps)):
            d_28_remaining_: int
            d_28_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_28_remaining_) >= (5):
                out23_: _dafny.Seq
                out24_: bool
                out25_: _dafny.Seq
                out23_, out24_, out25_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                generated = out23_
                insideConstrainedOut = out24_
                currentConstrainedOut = out25_
                d_1_steps_ = (d_1_steps_) + (1)
                d_29_minTokensB_: int
                d_29_minTokensB_ = 3
                d_30_tokensFilledB_: int
                d_30_tokensFilledB_ = 0
                d_31_fillBudgetB_: int
                d_31_fillBudgetB_ = (maxSteps) - (d_1_steps_)
                if (d_31_fillBudgetB_) > (40):
                    d_31_fillBudgetB_ = 40
                d_32_fillStepsB_: int
                d_32_fillStepsB_ = 0
                with _dafny.label("4_0_0"):
                    while (((d_32_fillStepsB_) < (d_31_fillBudgetB_)) and (insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                        with _dafny.c_label("4_0_0"):
                            if (d_30_tokensFilledB_) >= (d_29_minTokensB_):
                                d_33_cg3_: _dafny.Seq
                                d_34_ci3_: bool
                                d_35_cc3_: _dafny.Seq
                                d_36_closed3_: bool
                                out26_: _dafny.Seq
                                out27_: bool
                                out28_: _dafny.Seq
                                out29_: bool
                                out26_, out27_, out28_, out29_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_33_cg3_ = out26_
                                d_34_ci3_ = out27_
                                d_35_cc3_ = out28_
                                d_36_closed3_ = out29_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_32_fillStepsB_ = (d_32_fillStepsB_) + (1)
                                if d_36_closed3_:
                                    generated = d_33_cg3_
                                    insideConstrainedOut = d_34_ci3_
                                    currentConstrainedOut = d_35_cc3_
                                elif True:
                                    if (insideConstrainedOut) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                        d_37_constrainedPrompt3_: _dafny.Seq
                                        d_37_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                        d_38_next3_: _dafny.Seq
                                        out30_: _dafny.Seq
                                        out30_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_37_constrainedPrompt3_, currentConstrainedOut, eosToken)
                                        d_38_next3_ = out30_
                                        if (d_38_next3_) == (eosToken):
                                            raise _dafny.Break("4_0_0")
                                        elif True:
                                            d_39_ag3_: _dafny.Seq
                                            d_40_ai3_: bool
                                            d_41_ac3_: _dafny.Seq
                                            out31_: _dafny.Seq
                                            out32_: bool
                                            out33_: _dafny.Seq
                                            out31_, out32_, out33_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_38_next3_)
                                            d_39_ag3_ = out31_
                                            d_40_ai3_ = out32_
                                            d_41_ac3_ = out33_
                                            generated = d_39_ag3_
                                            insideConstrainedOut = d_40_ai3_
                                            currentConstrainedOut = d_41_ac3_
                                            d_30_tokensFilledB_ = (d_30_tokensFilledB_) + (1)
                            elif True:
                                if (insideConstrainedOut) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                    d_42_constrainedPrompt3_: _dafny.Seq
                                    d_42_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_43_next3_: _dafny.Seq
                                    out34_: _dafny.Seq
                                    out34_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_42_constrainedPrompt3_, currentConstrainedOut, eosToken)
                                    d_43_next3_ = out34_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_32_fillStepsB_ = (d_32_fillStepsB_) + (1)
                                    if (d_43_next3_) == (eosToken):
                                        raise _dafny.Break("4_0_0")
                                    elif True:
                                        d_44_ag3_: _dafny.Seq
                                        d_45_ai3_: bool
                                        d_46_ac3_: _dafny.Seq
                                        out35_: _dafny.Seq
                                        out36_: bool
                                        out37_: _dafny.Seq
                                        out35_, out36_, out37_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_43_next3_)
                                        d_44_ag3_ = out35_
                                        d_45_ai3_ = out36_
                                        d_46_ac3_ = out37_
                                        generated = d_44_ag3_
                                        insideConstrainedOut = d_45_ai3_
                                        currentConstrainedOut = d_46_ac3_
                                        d_30_tokensFilledB_ = (d_30_tokensFilledB_) + (1)
                                elif True:
                                    d_47_cg4_: _dafny.Seq
                                    d_48_ci4_: bool
                                    d_49_cc4_: _dafny.Seq
                                    d_50_closed4_: bool
                                    out38_: _dafny.Seq
                                    out39_: bool
                                    out40_: _dafny.Seq
                                    out41_: bool
                                    out38_, out39_, out40_, out41_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                    d_47_cg4_ = out38_
                                    d_48_ci4_ = out39_
                                    d_49_cc4_ = out40_
                                    d_50_closed4_ = out41_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_32_fillStepsB_ = (d_32_fillStepsB_) + (1)
                                    if d_50_closed4_:
                                        generated = d_47_cg4_
                                        insideConstrainedOut = d_48_ci4_
                                        currentConstrainedOut = d_49_cc4_
                            pass
                    pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_51_closeBudget_: int
            d_51_closeBudget_ = (maxSteps) - (d_1_steps_)
            out42_: _dafny.Seq
            out43_: bool
            out44_: _dafny.Seq
            out42_, out43_, out44_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_51_closeBudget_)
            generated = out42_
            insideConstrainedOut = out43_
            currentConstrainedOut = out44_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

