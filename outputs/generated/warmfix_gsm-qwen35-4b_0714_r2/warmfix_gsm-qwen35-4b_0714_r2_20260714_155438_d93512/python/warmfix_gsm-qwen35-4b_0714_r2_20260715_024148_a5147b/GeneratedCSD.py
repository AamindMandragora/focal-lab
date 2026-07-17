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
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the very end, write your final arithmetic answer as a single expression inside << >> using only variable names, numbers, +, -, *, /, //, %, (, ), int(). No LaTeX, no {}, no **. Write exactly one <<expression>> at the end as your final answer."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_prefixBudget_: int
            d_2_prefixBudget_ = _dafny.euclidian_division((maxSteps) * (80), 100)
            if (d_2_prefixBudget_) >= (maxSteps):
                d_2_prefixBudget_ = (maxSteps) - (1)
            d_3_steps_: int
            d_3_steps_ = 0
            d_4_answerWritten_: bool
            d_4_answerWritten_ = False
            with _dafny.label("1_0"):
                while ((d_3_steps_) < (d_2_prefixBudget_)) and (not(d_4_answerWritten_)):
                    with _dafny.c_label("1_0"):
                        if not(insideConstrainedOut):
                            d_5_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_5_next_ = out0_
                            d_3_steps_ = (d_3_steps_) + (1)
                            if (d_5_next_) == (eosToken):
                                raise _dafny.Break("1_0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
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
                        elif True:
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
                            d_3_steps_ = (d_3_steps_) + (1)
                            if d_12_closed_:
                                generated = d_9_cg_
                                insideConstrainedOut = d_10_ci_
                                currentConstrainedOut = d_11_cc_
                                d_4_answerWritten_ = True
                                raise _dafny.Break("1_0")
                            elif (d_3_steps_) < (d_2_prefixBudget_):
                                d_13_constrainedPrompt_: _dafny.Seq
                                d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_14_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_14_next_ = out8_
                                d_3_steps_ = (d_3_steps_) + (1)
                                if (d_14_next_) == (eosToken):
                                    raise _dafny.Break("1_0")
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
                        pass
                pass
            if (insideConstrainedOut) and ((d_3_steps_) < (maxSteps)):
                d_18_closeBudget_: int
                d_18_closeBudget_ = (maxSteps) - (d_3_steps_)
                d_19_cg_: _dafny.Seq
                d_20_ci_: bool
                d_21_cc_: _dafny.Seq
                out12_: _dafny.Seq
                out13_: bool
                out14_: _dafny.Seq
                out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget_)
                d_19_cg_ = out12_
                d_20_ci_ = out13_
                d_21_cc_ = out14_
                generated = d_19_cg_
                insideConstrainedOut = d_20_ci_
                currentConstrainedOut = d_21_cc_
                d_3_steps_ = maxSteps
            elif ((not(insideConstrainedOut)) and (not(d_4_answerWritten_))) and ((d_3_steps_) < (maxSteps)):
                d_22_g2_: _dafny.Seq
                d_23_ic2_: bool
                d_24_cc2_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_22_g2_ = out15_
                d_23_ic2_ = out16_
                d_24_cc2_ = out17_
                generated = d_22_g2_
                insideConstrainedOut = d_23_ic2_
                currentConstrainedOut = d_24_cc2_
                d_3_steps_ = (d_3_steps_) + (1)
                if (d_3_steps_) < (maxSteps):
                    d_25_closeBudget2_: int
                    d_25_closeBudget2_ = (maxSteps) - (d_3_steps_)
                    d_26_cg2_: _dafny.Seq
                    d_27_ci2_: bool
                    d_28_cc2b_: _dafny.Seq
                    out18_: _dafny.Seq
                    out19_: bool
                    out20_: _dafny.Seq
                    out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_closeBudget2_)
                    d_26_cg2_ = out18_
                    d_27_ci2_ = out19_
                    d_28_cc2b_ = out20_
                    generated = d_26_cg2_
                    insideConstrainedOut = d_27_ci2_
                    currentConstrainedOut = d_28_cc2b_
                    d_3_steps_ = maxSteps
            cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

