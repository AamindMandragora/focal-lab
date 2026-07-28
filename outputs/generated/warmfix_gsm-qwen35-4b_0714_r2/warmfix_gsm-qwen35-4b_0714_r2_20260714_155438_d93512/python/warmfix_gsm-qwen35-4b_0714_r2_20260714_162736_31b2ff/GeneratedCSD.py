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
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Reason in plain text. At the very end, write your FINAL arithmetic expression inside << >> using only variable names, numbers, +, -, *, /, //, %, (, ), int(). No LaTeX, no dollar signs, no curly braces, no backticks inside << >>. Write exactly ONE << >> block at the end, then stop."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_prefixBudget_: int
            d_3_prefixBudget_ = _dafny.euclidian_division((maxSteps) * (4), 5)
            if (d_3_prefixBudget_) >= (maxSteps):
                d_3_prefixBudget_ = (maxSteps) - (1)
            d_4_minSpanBudget_: int
            d_4_minSpanBudget_ = 10
            if ((d_3_prefixBudget_) + (d_4_minSpanBudget_)) > (maxSteps):
                if (maxSteps) >= (d_4_minSpanBudget_):
                    d_3_prefixBudget_ = (maxSteps) - (d_4_minSpanBudget_)
                elif True:
                    d_3_prefixBudget_ = 0
            with _dafny.label("1_0"):
                while ((d_2_steps_) < (d_3_prefixBudget_)) and (not(insideConstrainedOut)):
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
                        pass
                pass
            if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                d_6_g2_: _dafny.Seq
                d_7_ic2_: bool
                d_8_cc2_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_6_g2_ = out1_
                d_7_ic2_ = out2_
                d_8_cc2_ = out3_
                generated = d_6_g2_
                insideConstrainedOut = d_7_ic2_
                currentConstrainedOut = d_8_cc2_
                d_2_steps_ = (d_2_steps_) + (1)
            d_9_spanTokenCount_: int
            d_9_spanTokenCount_ = 0
            d_10_minSpanTokens_: int
            d_10_minSpanTokens_ = 3
            with _dafny.label("1_1"):
                while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                    with _dafny.c_label("1_1"):
                        if (d_9_spanTokenCount_) >= (d_10_minSpanTokens_):
                            d_11_cg_: _dafny.Seq
                            d_12_ci_: bool
                            d_13_cc_: _dafny.Seq
                            d_14_closed_: bool
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out7_: bool
                            out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_11_cg_ = out4_
                            d_12_ci_ = out5_
                            d_13_cc_ = out6_
                            d_14_closed_ = out7_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_14_closed_:
                                generated = d_11_cg_
                                insideConstrainedOut = d_12_ci_
                                currentConstrainedOut = d_13_cc_
                                raise _dafny.Break("1_1")
                            if ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                                d_15_constrainedPrompt_: _dafny.Seq
                                d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_16_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                                d_16_next_ = out8_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_16_next_) == (eosToken):
                                    raise _dafny.Break("1_1")
                                elif True:
                                    d_17_ag_: _dafny.Seq
                                    d_18_ai_: bool
                                    d_19_ac_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                    d_17_ag_ = out9_
                                    d_18_ai_ = out10_
                                    d_19_ac_ = out11_
                                    generated = d_17_ag_
                                    insideConstrainedOut = d_18_ai_
                                    currentConstrainedOut = d_19_ac_
                                    d_9_spanTokenCount_ = (d_9_spanTokenCount_) + (1)
                        elif True:
                            d_20_constrainedPrompt_: _dafny.Seq
                            d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_21_next_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_21_next_ = out12_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_21_next_) == (eosToken):
                                raise _dafny.Break("1_1")
                            elif True:
                                d_22_ag_: _dafny.Seq
                                d_23_ai_: bool
                                d_24_ac_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                d_22_ag_ = out13_
                                d_23_ai_ = out14_
                                d_24_ac_ = out15_
                                generated = d_22_ag_
                                insideConstrainedOut = d_23_ai_
                                currentConstrainedOut = d_24_ac_
                                d_9_spanTokenCount_ = (d_9_spanTokenCount_) + (1)
                        pass
                pass
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_25_closeBudget_: int
                d_25_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_26_cg_: _dafny.Seq
                d_27_ci_: bool
                d_28_cc_: _dafny.Seq
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_closeBudget_)
                d_26_cg_ = out16_
                d_27_ci_ = out17_
                d_28_cc_ = out18_
                generated = d_26_cg_
                insideConstrainedOut = d_27_ci_
                currentConstrainedOut = d_28_cc_
                d_2_steps_ = maxSteps
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

