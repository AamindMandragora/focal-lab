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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. At the very end, write the final answer as a single arithmetic expression inside << >>. Use ONLY: variable names (no braces), integers, +, -, *, /, //, %, int(), and parentheses. No {braces}, no **, no text inside << >>. Open << >> exactly ONCE for the final answer only. Example: The answer is <<n * (mult + 1)>>")))
        d_2_unconstrainedBudget_: int
        d_2_unconstrainedBudget_ = _dafny.euclidian_division((maxSteps) * (80), 100)
        if ((maxSteps) >= (50)) and ((d_2_unconstrainedBudget_) > ((maxSteps) - (50))):
            d_2_unconstrainedBudget_ = (maxSteps) - (50)
        if (d_2_unconstrainedBudget_) > (maxSteps):
            d_2_unconstrainedBudget_ = maxSteps
        d_3_spanDone_: bool
        d_3_spanDone_ = False
        with _dafny.label("0"):
            while (((d_1_steps_) < (d_2_unconstrainedBudget_)) and (not(insideConstrainedOut))) and (not(d_3_spanDone_)):
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
                            if (d_5_spanFillBudget_) > (40):
                                d_5_spanFillBudget_ = 40
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
                                            out8_: _dafny.Seq
                                            out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                                            d_12_nextC_ = out8_
                                            if (d_12_nextC_) == (eosToken):
                                                raise _dafny.Break("2_1_0_0")
                                            elif True:
                                                d_13_ag_: _dafny.Seq
                                                d_14_ai_: bool
                                                d_15_ac_: _dafny.Seq
                                                out9_: _dafny.Seq
                                                out10_: bool
                                                out11_: _dafny.Seq
                                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_nextC_)
                                                d_13_ag_ = out9_
                                                d_14_ai_ = out10_
                                                d_15_ac_ = out11_
                                                generated = d_13_ag_
                                                insideConstrainedOut = d_14_ai_
                                                currentConstrainedOut = d_15_ac_
                                        pass
                                pass
                            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                                d_16_closeBudget1_: int
                                d_16_closeBudget1_ = (maxSteps) - (d_1_steps_)
                                if (d_16_closeBudget1_) > (25):
                                    d_16_closeBudget1_ = 25
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_closeBudget1_)
                                generated = out12_
                                insideConstrainedOut = out13_
                                currentConstrainedOut = out14_
                                d_1_steps_ = (d_1_steps_) + (d_16_closeBudget1_)
                                if (d_1_steps_) > (maxSteps):
                                    d_1_steps_ = maxSteps
                            if not(insideConstrainedOut):
                                d_3_spanDone_ = True
                            raise _dafny.Break("0")
                    pass
            pass
        if ((not(d_3_spanDone_)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_17_remaining_: int
            d_17_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_17_remaining_) >= (5):
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                generated = out15_
                insideConstrainedOut = out16_
                currentConstrainedOut = out17_
                d_1_steps_ = (d_1_steps_) + (1)
                d_18_fillBudget_: int
                d_18_fillBudget_ = (maxSteps) - (d_1_steps_)
                if (d_18_fillBudget_) > (40):
                    d_18_fillBudget_ = 40
                d_19_fillSteps_: int
                d_19_fillSteps_ = 0
                with _dafny.label("3_0_0"):
                    while (((d_19_fillSteps_) < (d_18_fillBudget_)) and (insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                        with _dafny.c_label("3_0_0"):
                            d_20_cg3_: _dafny.Seq
                            d_21_ci3_: bool
                            d_22_cc3_: _dafny.Seq
                            d_23_closed3_: bool
                            out18_: _dafny.Seq
                            out19_: bool
                            out20_: _dafny.Seq
                            out21_: bool
                            out18_, out19_, out20_, out21_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_20_cg3_ = out18_
                            d_21_ci3_ = out19_
                            d_22_cc3_ = out20_
                            d_23_closed3_ = out21_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_19_fillSteps_ = (d_19_fillSteps_) + (1)
                            if d_23_closed3_:
                                generated = d_20_cg3_
                                insideConstrainedOut = d_21_ci3_
                                currentConstrainedOut = d_22_cc3_
                            elif True:
                                d_24_constrainedPrompt3_: _dafny.Seq
                                d_24_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_25_next3_: _dafny.Seq
                                out22_: _dafny.Seq
                                out22_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_24_constrainedPrompt3_, currentConstrainedOut, eosToken)
                                d_25_next3_ = out22_
                                if (d_25_next3_) == (eosToken):
                                    raise _dafny.Break("3_0_0")
                                elif True:
                                    d_26_ag3_: _dafny.Seq
                                    d_27_ai3_: bool
                                    d_28_ac3_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next3_)
                                    d_26_ag3_ = out23_
                                    d_27_ai3_ = out24_
                                    d_28_ac3_ = out25_
                                    generated = d_26_ag3_
                                    insideConstrainedOut = d_27_ai3_
                                    currentConstrainedOut = d_28_ac3_
                            pass
                    pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_29_closeBudget_: int
            d_29_closeBudget_ = (maxSteps) - (d_1_steps_)
            out26_: _dafny.Seq
            out27_: bool
            out28_: _dafny.Seq
            out26_, out27_, out28_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudget_)
            generated = out26_
            insideConstrainedOut = out27_
            currentConstrainedOut = out28_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

