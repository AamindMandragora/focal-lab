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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step using plain text reasoning. At the very end, write the final answer as a single arithmetic expression inside << >>. Inside << >>, use ONLY: variable names without braces (write x not {x}), integers, +, -, *, /, //, %, int(), and parentheses. No LaTeX, no {braces}, no ** exponents, no text. Example: <<int(n * price + base)>> or <<x * k * (12 // n)>>")))
        d_2_unconstrainedBudget_: int
        d_2_unconstrainedBudget_ = maxSteps
        if (maxSteps) > (60):
            d_2_unconstrainedBudget_ = (maxSteps) - (60)
        d_3_spanSuccessfullyClosed_: bool
        d_3_spanSuccessfullyClosed_ = False
        with _dafny.label("0"):
            while (((d_1_steps_) < (d_2_unconstrainedBudget_)) and (not(insideConstrainedOut))) and (not(d_3_spanSuccessfullyClosed_)):
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
                            raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_5_fillSteps_: int
            d_5_fillSteps_ = 0
            d_6_maxFill_: int
            d_6_maxFill_ = 40
            with _dafny.label("2_0"):
                while (((d_5_fillSteps_) < (d_6_maxFill_)) and (insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                    with _dafny.c_label("2_0"):
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_7_cg_: _dafny.Seq
                            d_8_ci_: bool
                            d_9_cc_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_cg_ = out4_
                            d_8_ci_ = out5_
                            d_9_cc_ = out6_
                            generated = d_7_cg_
                            insideConstrainedOut = d_8_ci_
                            currentConstrainedOut = d_9_cc_
                            d_3_spanSuccessfullyClosed_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_5_fillSteps_ = (d_5_fillSteps_) + (1)
                        elif True:
                            d_10_constrainedPrompt_: _dafny.Seq
                            d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_11_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_11_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_5_fillSteps_ = (d_5_fillSteps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("2_0")
                            elif True:
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_12_ag_: _dafny.Seq
                                    d_13_ai_: bool
                                    d_14_ac_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                    d_12_ag_ = out8_
                                    d_13_ai_ = out9_
                                    d_14_ac_ = out10_
                                    generated = d_12_ag_
                                    insideConstrainedOut = d_13_ai_
                                    currentConstrainedOut = d_14_ac_
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_15_closeBudget1_: int
                d_15_closeBudget1_ = (maxSteps) - (d_1_steps_)
                if (d_15_closeBudget1_) > (35):
                    d_15_closeBudget1_ = 35
                out11_: _dafny.Seq
                out12_: bool
                out13_: _dafny.Seq
                out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget1_)
                generated = out11_
                insideConstrainedOut = out12_
                currentConstrainedOut = out13_
                if not(insideConstrainedOut):
                    d_3_spanSuccessfullyClosed_ = True
                if ((d_1_steps_) + (d_15_closeBudget1_)) <= (maxSteps):
                    d_1_steps_ = (d_1_steps_) + (d_15_closeBudget1_)
                elif True:
                    d_1_steps_ = maxSteps
        if ((not(insideConstrainedOut)) and (not(d_3_spanSuccessfullyClosed_))) and ((d_1_steps_) < (d_2_unconstrainedBudget_)):
            with _dafny.label("3_0"):
                while (((d_1_steps_) < (d_2_unconstrainedBudget_)) and (not(insideConstrainedOut))) and (not(d_3_spanSuccessfullyClosed_)):
                    with _dafny.c_label("3_0"):
                        d_16_next2_: _dafny.Seq
                        out14_: _dafny.Seq
                        out14_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_16_next2_ = out14_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_16_next2_) == (eosToken):
                            raise _dafny.Break("3_0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_next2_]))
                            if (d_16_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                generated = out15_
                                insideConstrainedOut = out16_
                                currentConstrainedOut = out17_
                                raise _dafny.Break("3_0")
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_17_fill2_: int
                d_17_fill2_ = 0
                d_18_maxFill2_: int
                d_18_maxFill2_ = 40
                with _dafny.label("3_1_0"):
                    while (((d_17_fill2_) < (d_18_maxFill2_)) and (insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                        with _dafny.c_label("3_1_0"):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_19_cg2_: _dafny.Seq
                                d_20_ci2_: bool
                                d_21_cc2_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_19_cg2_ = out18_
                                d_20_ci2_ = out19_
                                d_21_cc2_ = out20_
                                generated = d_19_cg2_
                                insideConstrainedOut = d_20_ci2_
                                currentConstrainedOut = d_21_cc2_
                                d_3_spanSuccessfullyClosed_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_17_fill2_ = (d_17_fill2_) + (1)
                            elif True:
                                d_22_constrainedPrompt2_: _dafny.Seq
                                d_22_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_23_next3_: _dafny.Seq
                                out21_: _dafny.Seq
                                out21_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt2_, currentConstrainedOut, eosToken)
                                d_23_next3_ = out21_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_17_fill2_ = (d_17_fill2_) + (1)
                                if (d_23_next3_) == (eosToken):
                                    raise _dafny.Break("3_1_0")
                                elif True:
                                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                        d_24_ag2_: _dafny.Seq
                                        d_25_ai2_: bool
                                        d_26_ac2_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out23_: bool
                                        out24_: _dafny.Seq
                                        out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next3_)
                                        d_24_ag2_ = out22_
                                        d_25_ai2_ = out23_
                                        d_26_ac2_ = out24_
                                        generated = d_24_ag2_
                                        insideConstrainedOut = d_25_ai2_
                                        currentConstrainedOut = d_26_ac2_
                            pass
                    pass
                if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                    d_27_closeBudget2_: int
                    d_27_closeBudget2_ = (maxSteps) - (d_1_steps_)
                    if (d_27_closeBudget2_) > (35):
                        d_27_closeBudget2_ = 35
                    out25_: _dafny.Seq
                    out26_: bool
                    out27_: _dafny.Seq
                    out25_, out26_, out27_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudget2_)
                    generated = out25_
                    insideConstrainedOut = out26_
                    currentConstrainedOut = out27_
                    if not(insideConstrainedOut):
                        d_3_spanSuccessfullyClosed_ = True
                    if ((d_1_steps_) + (d_27_closeBudget2_)) <= (maxSteps):
                        d_1_steps_ = (d_1_steps_) + (d_27_closeBudget2_)
                    elif True:
                        d_1_steps_ = maxSteps
        if ((not(insideConstrainedOut)) and (not(d_3_spanSuccessfullyClosed_))) and ((d_1_steps_) < (maxSteps)):
            d_28_remaining_: int
            d_28_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_28_remaining_) >= (5):
                out28_: _dafny.Seq
                out29_: bool
                out30_: _dafny.Seq
                out28_, out29_, out30_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                generated = out28_
                insideConstrainedOut = out29_
                currentConstrainedOut = out30_
                d_1_steps_ = (d_1_steps_) + (1)
                d_29_forceFill_: int
                d_29_forceFill_ = 0
                d_30_forceMax_: int
                d_30_forceMax_ = 35
                with _dafny.label("4_0_0"):
                    while (((d_29_forceFill_) < (d_30_forceMax_)) and (insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                        with _dafny.c_label("4_0_0"):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_31_cg3_: _dafny.Seq
                                d_32_ci3_: bool
                                d_33_cc3_: _dafny.Seq
                                out31_: _dafny.Seq
                                out32_: bool
                                out33_: _dafny.Seq
                                out31_, out32_, out33_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_31_cg3_ = out31_
                                d_32_ci3_ = out32_
                                d_33_cc3_ = out33_
                                generated = d_31_cg3_
                                insideConstrainedOut = d_32_ci3_
                                currentConstrainedOut = d_33_cc3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_29_forceFill_ = (d_29_forceFill_) + (1)
                            elif True:
                                d_34_constrainedPrompt3_: _dafny.Seq
                                d_34_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_35_next5_: _dafny.Seq
                                out34_: _dafny.Seq
                                out34_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_34_constrainedPrompt3_, currentConstrainedOut, eosToken)
                                d_35_next5_ = out34_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_29_forceFill_ = (d_29_forceFill_) + (1)
                                if (d_35_next5_) == (eosToken):
                                    raise _dafny.Break("4_0_0")
                                elif True:
                                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                        d_36_ag4_: _dafny.Seq
                                        d_37_ai4_: bool
                                        d_38_ac4_: _dafny.Seq
                                        out35_: _dafny.Seq
                                        out36_: bool
                                        out37_: _dafny.Seq
                                        out35_, out36_, out37_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_35_next5_)
                                        d_36_ag4_ = out35_
                                        d_37_ai4_ = out36_
                                        d_38_ac4_ = out37_
                                        generated = d_36_ag4_
                                        insideConstrainedOut = d_37_ai4_
                                        currentConstrainedOut = d_38_ac4_
                            pass
                    pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_39_closeBudget_: int
            d_39_closeBudget_ = (maxSteps) - (d_1_steps_)
            out38_: _dafny.Seq
            out39_: bool
            out40_: _dafny.Seq
            out38_, out39_, out40_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_39_closeBudget_)
            generated = out38_
            insideConstrainedOut = out39_
            currentConstrainedOut = out40_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

