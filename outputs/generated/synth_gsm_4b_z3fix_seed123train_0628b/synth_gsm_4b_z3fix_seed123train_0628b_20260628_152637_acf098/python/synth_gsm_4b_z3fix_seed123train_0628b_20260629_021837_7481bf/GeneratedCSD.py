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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using symbolic variable names (without braces). At the very END of your response, write the final answer as a single complete arithmetic expression inside << >>. The expression must use variable names, numbers, +, -, *, /, //, %, int(), and parentheses only. No ** operator. No {braces}. No words. Write << >> exactly ONCE at the very end. Example: <<int(n1 * d1 + n2 * d2) // 12>>")))
        d_2_unconstrainedBudget_: int
        d_2_unconstrainedBudget_ = maxSteps
        if (maxSteps) > (80):
            d_2_unconstrainedBudget_ = (maxSteps) - (80)
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_2_unconstrainedBudget_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                        if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
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
            d_4_spanBudget_: int
            d_4_spanBudget_ = (maxSteps) - (d_1_steps_)
            if (d_4_spanBudget_) > (40):
                d_4_spanBudget_ = 40
            d_5_spanSteps_: int
            d_5_spanSteps_ = 0
            with _dafny.label("2_0"):
                while (((d_5_spanSteps_) < (d_4_spanBudget_)) and (insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                    with _dafny.c_label("2_0"):
                        d_6_cg_: _dafny.Seq
                        d_7_ci_: bool
                        d_8_cc_: _dafny.Seq
                        d_9_closed_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_6_cg_ = out4_
                        d_7_ci_ = out5_
                        d_8_cc_ = out6_
                        d_9_closed_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_5_spanSteps_ = (d_5_spanSteps_) + (1)
                        if d_9_closed_:
                            generated = d_6_cg_
                            insideConstrainedOut = d_7_ci_
                            currentConstrainedOut = d_8_cc_
                        elif True:
                            d_10_constrainedPrompt_: _dafny.Seq
                            d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_11_next_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_11_next_ = out8_
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("2_0")
                            elif True:
                                d_12_ag_: _dafny.Seq
                                d_13_ai_: bool
                                d_14_ac_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                d_12_ag_ = out9_
                                d_13_ai_ = out10_
                                d_14_ac_ = out11_
                                generated = d_12_ag_
                                insideConstrainedOut = d_13_ai_
                                currentConstrainedOut = d_14_ac_
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_15_cb1_: int
                d_15_cb1_ = (maxSteps) - (d_1_steps_)
                if (d_15_cb1_) > (30):
                    d_15_cb1_ = 30
                out12_: _dafny.Seq
                out13_: bool
                out14_: _dafny.Seq
                out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_cb1_)
                generated = out12_
                insideConstrainedOut = out13_
                currentConstrainedOut = out14_
                d_1_steps_ = (d_1_steps_) + (d_15_cb1_)
                if (d_1_steps_) > (maxSteps):
                    d_1_steps_ = maxSteps
        with _dafny.label("1"):
            while (((d_1_steps_) < (d_2_unconstrainedBudget_)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                with _dafny.c_label("1"):
                    d_16_next_: _dafny.Seq
                    out15_: _dafny.Seq
                    out15_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_16_next_ = out15_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_16_next_) == (eosToken):
                        raise _dafny.Break("1")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_next_]))
                        if (d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out16_
                            insideConstrainedOut = out17_
                            currentConstrainedOut = out18_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_17_spanBudget2_: int
            d_17_spanBudget2_ = (maxSteps) - (d_1_steps_)
            if (d_17_spanBudget2_) > (40):
                d_17_spanBudget2_ = 40
            d_18_spanSteps2_: int
            d_18_spanSteps2_ = 0
            with _dafny.label("4_0"):
                while (((d_18_spanSteps2_) < (d_17_spanBudget2_)) and (insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                    with _dafny.c_label("4_0"):
                        d_19_cg2_: _dafny.Seq
                        d_20_ci2_: bool
                        d_21_cc2_: _dafny.Seq
                        d_22_closed2_: bool
                        out19_: _dafny.Seq
                        out20_: bool
                        out21_: _dafny.Seq
                        out22_: bool
                        out19_, out20_, out21_, out22_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_19_cg2_ = out19_
                        d_20_ci2_ = out20_
                        d_21_cc2_ = out21_
                        d_22_closed2_ = out22_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_18_spanSteps2_ = (d_18_spanSteps2_) + (1)
                        if d_22_closed2_:
                            generated = d_19_cg2_
                            insideConstrainedOut = d_20_ci2_
                            currentConstrainedOut = d_21_cc2_
                        elif True:
                            d_23_constrainedPrompt2_: _dafny.Seq
                            d_23_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_24_next2_: _dafny.Seq
                            out23_: _dafny.Seq
                            out23_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_23_constrainedPrompt2_, currentConstrainedOut, eosToken)
                            d_24_next2_ = out23_
                            if (d_24_next2_) == (eosToken):
                                raise _dafny.Break("4_0")
                            elif True:
                                d_25_ag2_: _dafny.Seq
                                d_26_ai2_: bool
                                d_27_ac2_: _dafny.Seq
                                out24_: _dafny.Seq
                                out25_: bool
                                out26_: _dafny.Seq
                                out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next2_)
                                d_25_ag2_ = out24_
                                d_26_ai2_ = out25_
                                d_27_ac2_ = out26_
                                generated = d_25_ag2_
                                insideConstrainedOut = d_26_ai2_
                                currentConstrainedOut = d_27_ac2_
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_28_cb2_: int
                d_28_cb2_ = (maxSteps) - (d_1_steps_)
                if (d_28_cb2_) > (30):
                    d_28_cb2_ = 30
                out27_: _dafny.Seq
                out28_: bool
                out29_: _dafny.Seq
                out27_, out28_, out29_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_28_cb2_)
                generated = out27_
                insideConstrainedOut = out28_
                currentConstrainedOut = out29_
                d_1_steps_ = (d_1_steps_) + (d_28_cb2_)
                if (d_1_steps_) > (maxSteps):
                    d_1_steps_ = maxSteps
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_29_remainingForSpan_: int
            d_29_remainingForSpan_ = (maxSteps) - (d_1_steps_)
            if (d_29_remainingForSpan_) >= (5):
                out30_: _dafny.Seq
                out31_: bool
                out32_: _dafny.Seq
                out30_, out31_, out32_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                generated = out30_
                insideConstrainedOut = out31_
                currentConstrainedOut = out32_
                d_1_steps_ = (d_1_steps_) + (1)
                d_30_fillBudget_: int
                d_30_fillBudget_ = (maxSteps) - (d_1_steps_)
                if (d_30_fillBudget_) > (10):
                    d_30_fillBudget_ = (d_30_fillBudget_) - (10)
                elif True:
                    d_30_fillBudget_ = 0
                if (d_30_fillBudget_) > (60):
                    d_30_fillBudget_ = 60
                d_31_fillSteps_: int
                d_31_fillSteps_ = 0
                with _dafny.label("5_0_0"):
                    while (((d_31_fillSteps_) < (d_30_fillBudget_)) and (insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                        with _dafny.c_label("5_0_0"):
                            d_32_cg3_: _dafny.Seq
                            d_33_ci3_: bool
                            d_34_cc3_: _dafny.Seq
                            d_35_closed3_: bool
                            out33_: _dafny.Seq
                            out34_: bool
                            out35_: _dafny.Seq
                            out36_: bool
                            out33_, out34_, out35_, out36_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_32_cg3_ = out33_
                            d_33_ci3_ = out34_
                            d_34_cc3_ = out35_
                            d_35_closed3_ = out36_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_31_fillSteps_ = (d_31_fillSteps_) + (1)
                            if d_35_closed3_:
                                generated = d_32_cg3_
                                insideConstrainedOut = d_33_ci3_
                                currentConstrainedOut = d_34_cc3_
                            elif True:
                                d_36_constrainedPrompt3_: _dafny.Seq
                                d_36_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_37_next3_: _dafny.Seq
                                out37_: _dafny.Seq
                                out37_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_36_constrainedPrompt3_, currentConstrainedOut, eosToken)
                                d_37_next3_ = out37_
                                if (d_37_next3_) == (eosToken):
                                    raise _dafny.Break("5_0_0")
                                elif True:
                                    d_38_ag3_: _dafny.Seq
                                    d_39_ai3_: bool
                                    d_40_ac3_: _dafny.Seq
                                    out38_: _dafny.Seq
                                    out39_: bool
                                    out40_: _dafny.Seq
                                    out38_, out39_, out40_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_37_next3_)
                                    d_38_ag3_ = out38_
                                    d_39_ai3_ = out39_
                                    d_40_ac3_ = out40_
                                    generated = d_38_ag3_
                                    insideConstrainedOut = d_39_ai3_
                                    currentConstrainedOut = d_40_ac3_
                            pass
                    pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_41_closeBudget_: int
            d_41_closeBudget_ = (maxSteps) - (d_1_steps_)
            out41_: _dafny.Seq
            out42_: bool
            out43_: _dafny.Seq
            out41_, out42_, out43_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_41_closeBudget_)
            generated = out41_
            insideConstrainedOut = out42_
            currentConstrainedOut = out43_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

