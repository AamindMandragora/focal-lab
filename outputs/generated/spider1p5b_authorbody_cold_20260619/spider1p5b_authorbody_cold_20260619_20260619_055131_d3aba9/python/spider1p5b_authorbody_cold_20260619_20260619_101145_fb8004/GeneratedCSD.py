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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are a SQL expert. Generate a single SQL query answering the question. Start your response with 'SQL: ' then the query. Use only tables and columns from the schema provided.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_maxPreamble_: int
        d_2_maxPreamble_ = 8
        d_3_preambleSteps_: int
        d_3_preambleSteps_ = 0
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut))) and ((d_3_preambleSteps_) < (d_2_maxPreamble_)):
                with _dafny.c_label("0"):
                    d_4_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_4_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_3_preambleSteps_ = (d_3_preambleSteps_) + (1)
                    if (d_4_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_5_og_: _dafny.Seq
            d_6_oi_: bool
            d_7_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_5_og_ = out1_
            d_6_oi_ = out2_
            d_7_oc_ = out3_
            d_1_steps_ = (d_1_steps_) + (1)
            generated = d_5_og_
            insideConstrainedOut = d_6_oi_
            currentConstrainedOut = d_7_oc_
        d_8_closeBudgetReserve_: int
        d_8_closeBudgetReserve_ = 200
        with _dafny.label("1"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_9_remainingBudget_: int
                    d_9_remainingBudget_ = (maxSteps) - (d_1_steps_)
                    if (d_9_remainingBudget_) <= (d_8_closeBudgetReserve_):
                        if (d_9_remainingBudget_) > (0):
                            d_10_cg_: _dafny.Seq
                            d_11_ci_: bool
                            d_12_cc_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_remainingBudget_)
                            d_10_cg_ = out4_
                            d_11_ci_ = out5_
                            d_12_cc_ = out6_
                            generated = d_10_cg_
                            insideConstrainedOut = d_11_ci_
                            currentConstrainedOut = d_12_cc_
                            d_1_steps_ = (d_1_steps_) + (d_9_remainingBudget_)
                        elif True:
                            d_1_steps_ = maxSteps
                        raise _dafny.Break("1")
                    elif True:
                        d_13_cg_: _dafny.Seq
                        d_14_ci_: bool
                        d_15_cc_: _dafny.Seq
                        d_16_closed_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_13_cg_ = out7_
                        d_14_ci_ = out8_
                        d_15_cc_ = out9_
                        d_16_closed_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_16_closed_:
                            generated = d_13_cg_
                            insideConstrainedOut = d_14_ci_
                            currentConstrainedOut = d_15_cc_
                            raise _dafny.Break("1")
                        elif True:
                            d_17_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, prompt, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_17_next_ = out11_
                            if (d_17_next_) == (eosToken):
                                d_18_remaining_: int
                                d_18_remaining_ = (maxSteps) - (d_1_steps_)
                                if (d_18_remaining_) > (0):
                                    d_19_cg2_: _dafny.Seq
                                    d_20_ci2_: bool
                                    d_21_cc2_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_remaining_)
                                    d_19_cg2_ = out12_
                                    d_20_ci2_ = out13_
                                    d_21_cc2_ = out14_
                                    generated = d_19_cg2_
                                    insideConstrainedOut = d_20_ci2_
                                    currentConstrainedOut = d_21_cc2_
                                    d_1_steps_ = (d_1_steps_) + (d_18_remaining_)
                                elif True:
                                    d_1_steps_ = maxSteps
                                raise _dafny.Break("1")
                            elif True:
                                d_22_valid_: bool
                                out15_: bool
                                out15_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_17_next_)
                                d_22_valid_ = out15_
                                if d_22_valid_:
                                    d_23_ag_: _dafny.Seq
                                    d_24_ai_: bool
                                    d_25_ac_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                    d_23_ag_ = out16_
                                    d_24_ai_ = out17_
                                    d_25_ac_ = out18_
                                    generated = d_23_ag_
                                    insideConstrainedOut = d_24_ai_
                                    currentConstrainedOut = d_25_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

