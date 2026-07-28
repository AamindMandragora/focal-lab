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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this symbolic math word problem step by step. The placeholders like {n}, {mult}, {name}, {frac} are symbolic variables - do NOT substitute numbers for them. After your reasoning, write the final answer as a Python arithmetic expression using exactly those variable names inside << >> delimiters. For example: if the answer is n times (mult plus 1), write <<n * (mult + 1)>>. The expression inside << >> must use the variable names from the problem.")))
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_preambleBudget_: int
            if (maxSteps) > (150):
                d_2_preambleBudget_ = (maxSteps) - (120)
            elif True:
                d_2_preambleBudget_ = _dafny.euclidian_division((maxSteps) * (2), 3)
            with _dafny.label("1_0"):
                while ((d_1_steps_) < (d_2_preambleBudget_)) and (not(insideConstrainedOut)):
                    with _dafny.c_label("1_0"):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                        if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        pass
                pass
            if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_4_og_: _dafny.Seq
                d_5_oi_: bool
                d_6_oc_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_4_og_ = out1_
                d_5_oi_ = out2_
                d_6_oc_ = out3_
                generated = d_4_og_
                insideConstrainedOut = d_5_oi_
                currentConstrainedOut = d_6_oc_
                d_1_steps_ = (d_1_steps_) + (1)
            with _dafny.label("1_1"):
                while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                    with _dafny.c_label("1_1"):
                        d_7_remainingBudget_: int
                        d_7_remainingBudget_ = (maxSteps) - (d_1_steps_)
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
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_11_closed_:
                            generated = d_8_cg_
                            insideConstrainedOut = d_9_ci_
                            currentConstrainedOut = d_10_cc_
                        elif True:
                            d_12_constrainedPrompt_: _dafny.Seq
                            d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_13_next_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_13_next_ = out8_
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
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_17_remainBudget_: int
                d_17_remainBudget_ = (maxSteps) - (d_1_steps_)
                d_18_cg_: _dafny.Seq
                d_19_ci_: bool
                d_20_cc_: _dafny.Seq
                out12_: _dafny.Seq
                out13_: bool
                out14_: _dafny.Seq
                out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_remainBudget_)
                d_18_cg_ = out12_
                d_19_ci_ = out13_
                d_20_cc_ = out14_
                generated = d_18_cg_
                insideConstrainedOut = d_19_ci_
                currentConstrainedOut = d_20_cc_
                d_1_steps_ = maxSteps
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

