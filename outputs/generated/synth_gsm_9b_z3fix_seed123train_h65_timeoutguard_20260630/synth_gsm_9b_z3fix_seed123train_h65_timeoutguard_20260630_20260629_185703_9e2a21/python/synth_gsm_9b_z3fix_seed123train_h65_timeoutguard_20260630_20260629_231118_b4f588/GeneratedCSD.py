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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. Use the symbolic variable names from the problem (no curly braces). At the end of your reasoning, write the final symbolic expression for the answer. Use only: variable names, integers, +, -, *, /, //, %, int(). The final answer expression will be extracted automatically.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_reservedBudget_: int
        d_2_reservedBudget_ = 150
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_remaining_: int
                    d_3_remaining_ = (maxSteps) - (d_1_steps_)
                    if (d_3_remaining_) <= (d_2_reservedBudget_):
                        raise _dafny.Break("0")
                    d_4_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_4_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_4_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_5_remaining2_: int
            d_5_remaining2_ = (maxSteps) - (d_1_steps_)
            if (d_5_remaining2_) >= (3):
                d_6_og_: _dafny.Seq
                d_7_oi_: bool
                d_8_oc_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_6_og_ = out1_
                d_7_oi_ = out2_
                d_8_oc_ = out3_
                generated = d_6_og_
                insideConstrainedOut = d_7_oi_
                currentConstrainedOut = d_8_oc_
                d_1_steps_ = (d_1_steps_) + (1)
        d_9_spanSteps_: int
        d_9_spanSteps_ = 0
        d_10_spanBudget_: int
        d_10_spanBudget_ = 60
        with _dafny.label("1"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_11_remaining3_: int
                    d_11_remaining3_ = (maxSteps) - (d_1_steps_)
                    if ((d_9_spanSteps_) >= (d_10_spanBudget_)) or ((d_11_remaining3_) <= (5)):
                        d_12_closeBudget_: int
                        if (d_11_remaining3_) < (20):
                            d_12_closeBudget_ = d_11_remaining3_
                        elif True:
                            d_12_closeBudget_ = 20
                        if (d_12_closeBudget_) == (0):
                            raise _dafny.Break("1")
                        d_13_cg_: _dafny.Seq
                        d_14_ci_: bool
                        d_15_cc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_closeBudget_)
                        d_13_cg_ = out4_
                        d_14_ci_ = out5_
                        d_15_cc_ = out6_
                        generated = d_13_cg_
                        insideConstrainedOut = d_14_ci_
                        currentConstrainedOut = d_15_cc_
                        d_1_steps_ = (d_1_steps_) + (d_12_closeBudget_)
                        raise _dafny.Break("1")
                    d_16_cg2_: _dafny.Seq
                    d_17_ci2_: bool
                    d_18_cc2_: _dafny.Seq
                    d_19_closed_: bool
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out10_: bool
                    out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_16_cg2_ = out7_
                    d_17_ci2_ = out8_
                    d_18_cc2_ = out9_
                    d_19_closed_ = out10_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_9_spanSteps_ = (d_9_spanSteps_) + (1)
                    if d_19_closed_:
                        generated = d_16_cg2_
                        insideConstrainedOut = d_17_ci2_
                        currentConstrainedOut = d_18_cc2_
                        raise _dafny.Break("1")
                    if (d_1_steps_) < (maxSteps):
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_next_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_21_next_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_9_spanSteps_ = (d_9_spanSteps_) + (1)
                        if (d_21_next_) == (eosToken):
                            d_22_remaining4_: int
                            d_22_remaining4_ = (maxSteps) - (d_1_steps_)
                            if (d_22_remaining4_) == (0):
                                raise _dafny.Break("1")
                            d_23_closeBudget2_: int
                            if (d_22_remaining4_) < (25):
                                d_23_closeBudget2_ = d_22_remaining4_
                            elif True:
                                d_23_closeBudget2_ = 25
                            d_24_cg3_: _dafny.Seq
                            d_25_ci3_: bool
                            d_26_cc3_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudget2_)
                            d_24_cg3_ = out12_
                            d_25_ci3_ = out13_
                            d_26_cc3_ = out14_
                            generated = d_24_cg3_
                            insideConstrainedOut = d_25_ci3_
                            currentConstrainedOut = d_26_cc3_
                            d_1_steps_ = (d_1_steps_) + (d_23_closeBudget2_)
                            raise _dafny.Break("1")
                        elif True:
                            d_27_ag_: _dafny.Seq
                            d_28_ai_: bool
                            d_29_ac_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_27_ag_ = out15_
                            d_28_ai_ = out16_
                            d_29_ac_ = out17_
                            generated = d_27_ag_
                            insideConstrainedOut = d_28_ai_
                            currentConstrainedOut = d_29_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

