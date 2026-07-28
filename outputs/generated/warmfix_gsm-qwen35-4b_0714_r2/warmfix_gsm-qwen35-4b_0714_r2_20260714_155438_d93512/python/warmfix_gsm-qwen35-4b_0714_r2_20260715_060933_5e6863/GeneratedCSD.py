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
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the very end, write your final arithmetic answer as a single expression inside << >> using only variable names (no curly braces), numbers, +, -, *, /, //, %, (, ), int(). No LaTeX. No ** operator. Write exactly one <<expression>> at the end."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_reservedBudget_: int
            d_2_reservedBudget_ = 120
            d_3_freePhaseLimit_: int = int(0)
            if (maxSteps) > (d_2_reservedBudget_):
                d_3_freePhaseLimit_ = (maxSteps) - (d_2_reservedBudget_)
            elif True:
                d_3_freePhaseLimit_ = 0
            d_4_steps_: int
            d_4_steps_ = 0
            with _dafny.label("1_0"):
                while (d_4_steps_) < (d_3_freePhaseLimit_):
                    with _dafny.c_label("1_0"):
                        if not(insideConstrainedOut):
                            d_5_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_5_next_ = out0_
                            d_4_steps_ = (d_4_steps_) + (1)
                            if (d_5_next_) == (eosToken):
                                raise _dafny.Break("1_0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                                if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_6_cg_: _dafny.Seq
                            d_7_ci_: bool
                            d_8_cc_: _dafny.Seq
                            d_9_closed_: bool
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out4_: bool
                            out1_, out2_, out3_, out4_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_6_cg_ = out1_
                            d_7_ci_ = out2_
                            d_8_cc_ = out3_
                            d_9_closed_ = out4_
                            d_4_steps_ = (d_4_steps_) + (1)
                            if d_9_closed_:
                                generated = d_6_cg_
                                insideConstrainedOut = d_7_ci_
                                currentConstrainedOut = d_8_cc_
                            elif True:
                                if (d_4_steps_) < (d_3_freePhaseLimit_):
                                    d_10_constrainedPrompt_: _dafny.Seq
                                    d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_11_next_: _dafny.Seq
                                    out5_: _dafny.Seq
                                    out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_11_next_ = out5_
                                    d_4_steps_ = (d_4_steps_) + (1)
                                    if (d_11_next_) == (eosToken):
                                        raise _dafny.Break("1_0")
                                    elif True:
                                        d_12_valid_: bool
                                        out6_: bool
                                        out6_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_11_next_)
                                        d_12_valid_ = out6_
                                        if d_12_valid_:
                                            d_13_ag_: _dafny.Seq
                                            d_14_ai_: bool
                                            d_15_ac_: _dafny.Seq
                                            out7_: _dafny.Seq
                                            out8_: bool
                                            out9_: _dafny.Seq
                                            out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                            d_13_ag_ = out7_
                                            d_14_ai_ = out8_
                                            d_15_ac_ = out9_
                                            generated = d_13_ag_
                                            insideConstrainedOut = d_14_ai_
                                            currentConstrainedOut = d_15_ac_
                        pass
                pass
            if (d_4_steps_) < (maxSteps):
                d_16_remainingBudget_: int
                d_16_remainingBudget_ = (maxSteps) - (d_4_steps_)
                if insideConstrainedOut:
                    d_17_cg_: _dafny.Seq
                    d_18_ci_: bool
                    d_19_cc_: _dafny.Seq
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_remainingBudget_)
                    d_17_cg_ = out10_
                    d_18_ci_ = out11_
                    d_19_cc_ = out12_
                    generated = d_17_cg_
                    insideConstrainedOut = d_18_ci_
                    currentConstrainedOut = d_19_cc_
                    d_4_steps_ = maxSteps
                elif True:
                    d_20_closeCount_: int
                    out13_: int
                    out13_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                    d_20_closeCount_ = out13_
                    if ((d_20_closeCount_) == (0)) and ((d_16_remainingBudget_) >= (2)):
                        d_21_og_: _dafny.Seq
                        d_22_oi_: bool
                        d_23_oc_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out14_, out15_, out16_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_21_og_ = out14_
                        d_22_oi_ = out15_
                        d_23_oc_ = out16_
                        generated = d_21_og_
                        insideConstrainedOut = d_22_oi_
                        currentConstrainedOut = d_23_oc_
                        d_4_steps_ = (d_4_steps_) + (1)
                        d_24_spanBudget_: int
                        d_24_spanBudget_ = (maxSteps) - (d_4_steps_)
                        if (d_24_spanBudget_) > (0):
                            d_25_cg_: _dafny.Seq
                            d_26_ci_: bool
                            d_27_cc_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_spanBudget_)
                            d_25_cg_ = out17_
                            d_26_ci_ = out18_
                            d_27_cc_ = out19_
                            generated = d_25_cg_
                            insideConstrainedOut = d_26_ci_
                            currentConstrainedOut = d_27_cc_
                            d_4_steps_ = maxSteps
            cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

