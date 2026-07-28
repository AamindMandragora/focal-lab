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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output only a SQL SELECT query. Use exact column and table names from the schema. Keep the query simple and direct. Do not add unnecessary JOINs or subqueries.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if not(insideConstrainedOut):
            if (d_1_steps_) < (maxSteps):
                d_2_og_: _dafny.Seq
                d_3_oi_: bool
                d_4_oc_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_2_og_ = out0_
                d_3_oi_ = out1_
                d_4_oc_ = out2_
                generated = d_2_og_
                insideConstrainedOut = d_3_oi_
                currentConstrainedOut = d_4_oc_
                d_1_steps_ = (d_1_steps_) + (1)
        d_5_closeReserve_: int
        d_5_closeReserve_ = 30
        if (maxSteps) <= (30):
            if (maxSteps) > (1):
                d_5_closeReserve_ = (maxSteps) - (1)
            elif True:
                d_5_closeReserve_ = 0
        with _dafny.label("0"):
            while (insideConstrainedOut) and ((d_1_steps_) < ((maxSteps) - (d_5_closeReserve_))):
                with _dafny.c_label("0"):
                    d_6_cg_: _dafny.Seq
                    d_7_ci_: bool
                    d_8_cc_: _dafny.Seq
                    d_9_closed_: bool
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out6_: bool
                    out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_6_cg_ = out3_
                    d_7_ci_ = out4_
                    d_8_cc_ = out5_
                    d_9_closed_ = out6_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_9_closed_:
                        generated = d_6_cg_
                        insideConstrainedOut = d_7_ci_
                        currentConstrainedOut = d_8_cc_
                    elif True:
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_11_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), 50, eosToken)
                        d_11_next_ = out7_
                        if (d_11_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
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
            d_15_closeBudget_: int
            d_15_closeBudget_ = (maxSteps) - (d_1_steps_)
            out11_: _dafny.Seq
            out12_: bool
            out13_: _dafny.Seq
            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
            generated = out11_
            insideConstrainedOut = out12_
            currentConstrainedOut = out13_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

