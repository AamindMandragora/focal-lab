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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a concise SQL query answering the question using only the schema provided. Use simple SELECT statements with correct table and column names."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if insideConstrainedOut:
            if (maxSteps) > (0):
                d_2_closeBudget_: int
                d_2_closeBudget_ = maxSteps
                d_3_cg_: _dafny.Seq
                d_4_ci_: bool
                d_5_cc_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_2_closeBudget_)
                d_3_cg_ = out0_
                d_4_ci_ = out1_
                d_5_cc_ = out2_
                generated = d_3_cg_
                insideConstrainedOut = d_4_ci_
                currentConstrainedOut = d_5_cc_
                cost = d_2_closeBudget_
        elif True:
            d_6_steps_: int
            d_6_steps_ = 0
            with _dafny.label("1_0"):
                while (d_6_steps_) < (maxSteps):
                    with _dafny.c_label("1_0"):
                        if not(insideConstrainedOut):
                            d_7_og_: _dafny.Seq
                            d_8_oi_: bool
                            d_9_oc_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_og_ = out3_
                            d_8_oi_ = out4_
                            d_9_oc_ = out5_
                            generated = d_7_og_
                            insideConstrainedOut = d_8_oi_
                            currentConstrainedOut = d_9_oc_
                            d_6_steps_ = (d_6_steps_) + (1)
                        elif True:
                            d_10_cg_: _dafny.Seq
                            d_11_ci_: bool
                            d_12_cc_: _dafny.Seq
                            d_13_closed_: bool
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out9_: bool
                            out6_, out7_, out8_, out9_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_10_cg_ = out6_
                            d_11_ci_ = out7_
                            d_12_cc_ = out8_
                            d_13_closed_ = out9_
                            d_6_steps_ = (d_6_steps_) + (1)
                            if d_13_closed_:
                                generated = d_10_cg_
                                insideConstrainedOut = d_11_ci_
                                currentConstrainedOut = d_12_cc_
                                raise _dafny.Break("1_0")
                            elif True:
                                d_14_constrainedPrompt_: _dafny.Seq
                                d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_15_next_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_15_next_ = out10_
                                if (d_15_next_) == (eosToken):
                                    raise _dafny.Break("1_0")
                                elif True:
                                    d_16_ag_: _dafny.Seq
                                    d_17_ai_: bool
                                    d_18_ac_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                    d_16_ag_ = out11_
                                    d_17_ai_ = out12_
                                    d_18_ac_ = out13_
                                    generated = d_16_ag_
                                    insideConstrainedOut = d_17_ai_
                                    currentConstrainedOut = d_18_ac_
                        pass
                pass
            cost = d_6_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

