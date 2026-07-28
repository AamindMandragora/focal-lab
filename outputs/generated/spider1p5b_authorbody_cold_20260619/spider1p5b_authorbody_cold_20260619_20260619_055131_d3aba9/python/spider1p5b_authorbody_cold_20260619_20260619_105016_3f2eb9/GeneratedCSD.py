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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SQL query. Output format: SQL: <<YOUR QUERY>>. Use only tables and columns from the schema provided. No markdown, no explanation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_closeBudgetReserve_: int
        d_2_closeBudgetReserve_ = 200
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
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
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_4_remainingBudget_: int
                        d_4_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_4_remainingBudget_) <= (d_2_closeBudgetReserve_):
                            if (d_4_remainingBudget_) > (0):
                                d_5_cg_: _dafny.Seq
                                d_6_ci_: bool
                                d_7_cc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_4_remainingBudget_)
                                d_5_cg_ = out1_
                                d_6_ci_ = out2_
                                d_7_cc_ = out3_
                                generated = d_5_cg_
                                insideConstrainedOut = d_6_ci_
                                currentConstrainedOut = d_7_cc_
                                d_1_steps_ = (d_1_steps_) + (d_4_remainingBudget_)
                            elif True:
                                d_1_steps_ = maxSteps
                            raise _dafny.Break("0")
                        elif True:
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
                            generated = d_8_cg_
                            insideConstrainedOut = d_9_ci_
                            currentConstrainedOut = d_10_cc_
                            if d_11_closed_:
                                raise _dafny.Break("0")
                            elif True:
                                d_12_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, prompt, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_12_next_ = out8_
                                if (d_12_next_) == (eosToken):
                                    d_13_remaining_: int
                                    d_13_remaining_ = (maxSteps) - (d_1_steps_)
                                    if (d_13_remaining_) > (0):
                                        d_14_cg2_: _dafny.Seq
                                        d_15_ci2_: bool
                                        d_16_cc2_: _dafny.Seq
                                        out9_: _dafny.Seq
                                        out10_: bool
                                        out11_: _dafny.Seq
                                        out9_, out10_, out11_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_13_remaining_)
                                        d_14_cg2_ = out9_
                                        d_15_ci2_ = out10_
                                        d_16_cc2_ = out11_
                                        generated = d_14_cg2_
                                        insideConstrainedOut = d_15_ci2_
                                        currentConstrainedOut = d_16_cc2_
                                        d_1_steps_ = (d_1_steps_) + (d_13_remaining_)
                                    elif True:
                                        d_1_steps_ = maxSteps
                                    raise _dafny.Break("0")
                                elif True:
                                    d_17_valid_: bool
                                    out12_: bool
                                    out12_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_12_next_)
                                    d_17_valid_ = out12_
                                    if d_17_valid_:
                                        d_18_ag_: _dafny.Seq
                                        d_19_ai_: bool
                                        d_20_ac_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out14_: bool
                                        out15_: _dafny.Seq
                                        out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                        d_18_ag_ = out13_
                                        d_19_ai_ = out14_
                                        d_20_ac_ = out15_
                                        generated = d_18_ag_
                                        insideConstrainedOut = d_19_ai_
                                        currentConstrainedOut = d_20_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

