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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output format: SQL: <<single valid SQL query using only provided schema tables and columns>>. No markdown, no explanation, no extra text. The SQL must be complete and syntactically valid.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 16
        d_3_spanOpened_: bool
        d_3_spanOpened_ = insideConstrained
        d_4_maxFreeSteps_: int
        d_4_maxFreeSteps_ = 8
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    if (d_1_steps_) >= (d_4_maxFreeSteps_):
                        d_5_og_: _dafny.Seq
                        d_6_oi_: bool
                        d_7_oc_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_5_og_ = out0_
                        d_6_oi_ = out1_
                        d_7_oc_ = out2_
                        generated = d_5_og_
                        insideConstrainedOut = d_6_oi_
                        currentConstrainedOut = d_7_oc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanOpened_ = True
                    elif True:
                        d_8_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_8_next_ = out3_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_8_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                            if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_spanOpened_ = True
                    pass
            pass
        with _dafny.label("1"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_9_cg_: _dafny.Seq
                        d_10_ci_: bool
                        d_11_cc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_9_cg_ = out4_
                        d_10_ci_ = out5_
                        d_11_cc_ = out6_
                        generated = d_9_cg_
                        insideConstrainedOut = d_10_ci_
                        currentConstrainedOut = d_11_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("1")
                    elif True:
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), d_2_narrowThreshold_, eosToken)
                        d_13_next_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            if (d_1_steps_) < (maxSteps):
                                d_14_closeBudget_: int
                                d_14_closeBudget_ = (maxSteps) - (d_1_steps_)
                                d_15_cg2_: _dafny.Seq
                                d_16_ci2_: bool
                                d_17_cc2_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
                                d_15_cg2_ = out8_
                                d_16_ci2_ = out9_
                                d_17_cc2_ = out10_
                                generated = d_15_cg2_
                                insideConstrainedOut = d_16_ci2_
                                currentConstrainedOut = d_17_cc2_
                                d_1_steps_ = maxSteps
                            raise _dafny.Break("1")
                        elif True:
                            d_18_ag_: _dafny.Seq
                            d_19_ai_: bool
                            d_20_ac_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_18_ag_ = out11_
                            d_19_ai_ = out12_
                            d_20_ac_ = out13_
                            generated = d_18_ag_
                            insideConstrainedOut = d_19_ai_
                            currentConstrainedOut = d_20_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_21_closeBudget_: int
            d_21_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_22_cg3_: _dafny.Seq
            d_23_ci3_: bool
            d_24_cc3_: _dafny.Seq
            out14_: _dafny.Seq
            out15_: bool
            out16_: _dafny.Seq
            out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_closeBudget_)
            d_22_cg3_ = out14_
            d_23_ci3_ = out15_
            d_24_cc3_ = out16_
            generated = d_22_cg3_
            insideConstrainedOut = d_23_ci3_
            currentConstrainedOut = d_24_cc3_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

