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
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Work through the problem step by step in plain text. After ALL reasoning is complete, write the final arithmetic expression inside << >>. Use only: variable names, numbers, +, -, *, /, //, %, (, ), int(). No LaTeX, no $, no {}, no **, no backticks inside << >>. Open << exactly once at the very end. Do not open another << after closing >>."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_steps_: int
            d_2_steps_ = 0
            d_3_prefixBudget_: int
            d_3_prefixBudget_ = _dafny.euclidian_division((maxSteps) * (4), 5)
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
                        pass
                pass
            if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                d_5_g2_: _dafny.Seq
                d_6_ic2_: bool
                d_7_cc2_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_5_g2_ = out1_
                d_6_ic2_ = out2_
                d_7_cc2_ = out3_
                generated = d_5_g2_
                insideConstrainedOut = d_6_ic2_
                currentConstrainedOut = d_7_cc2_
                d_2_steps_ = (d_2_steps_) + (1)
            with _dafny.label("1_1"):
                while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                    with _dafny.c_label("1_1"):
                        d_8_cg_: _dafny.Seq
                        d_9_ci_: bool
                        d_10_cc_: _dafny.Seq
                        d_11_closed_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_8_cg_ = out4_
                        d_9_ci_ = out5_
                        d_10_cc_ = out6_
                        d_11_closed_ = out7_
                        d_2_steps_ = (d_2_steps_) + (1)
                        generated = d_8_cg_
                        insideConstrainedOut = d_9_ci_
                        currentConstrainedOut = d_10_cc_
                        if d_11_closed_:
                            raise _dafny.Break("1_1")
                        elif True:
                            if (d_2_steps_) < (maxSteps):
                                d_12_constrainedPrompt_: _dafny.Seq
                                d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_13_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                                d_13_next_ = out8_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_13_next_) == (eosToken):
                                    raise _dafny.Break("1_1")
                                elif True:
                                    d_14_ag_: _dafny.Seq
                                    d_15_ai_: bool
                                    d_16_ac_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                    d_14_ag_ = out9_
                                    d_15_ai_ = out10_
                                    d_16_ac_ = out11_
                                    generated = d_14_ag_
                                    insideConstrainedOut = d_15_ai_
                                    currentConstrainedOut = d_16_ac_
                        pass
                pass
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_17_closeBudget_: int
                d_17_closeBudget_ = (maxSteps) - (d_2_steps_)
                d_18_cg_: _dafny.Seq
                d_19_ci_: bool
                d_20_cc_: _dafny.Seq
                out12_: _dafny.Seq
                out13_: bool
                out14_: _dafny.Seq
                out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
                d_18_cg_ = out12_
                d_19_ci_ = out13_
                d_20_cc_ = out14_
                generated = d_18_cg_
                insideConstrainedOut = d_19_ci_
                currentConstrainedOut = d_20_cc_
                d_2_steps_ = maxSteps
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

