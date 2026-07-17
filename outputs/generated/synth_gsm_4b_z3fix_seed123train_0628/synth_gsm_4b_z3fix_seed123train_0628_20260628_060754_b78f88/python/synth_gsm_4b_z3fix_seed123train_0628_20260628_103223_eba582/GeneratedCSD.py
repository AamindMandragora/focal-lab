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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step-by-step with full arithmetic. Show each calculation explicitly with numbers. At the very end, write the final answer formula inside << >> using simple variable names without curly braces (write n1 not {n1}), standard operators (+, -, *, /, //, %), and parentheses only. Example: <<n * price - cost>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_maxFreeSteps_: int
        d_3_maxFreeSteps_ = 600
        d_4_spanOpened_: bool
        d_4_spanOpened_ = False
        d_5_spanClosed_: bool
        d_5_spanClosed_ = False
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut))) and (not(d_5_spanClosed_)):
                with _dafny.c_label("0"):
                    d_6_remaining_: int
                    d_6_remaining_ = (maxSteps) - (d_2_steps_)
                    if (d_6_remaining_) <= (20):
                        raise _dafny.Break("0")
                    if ((d_2_steps_) >= (d_3_maxFreeSteps_)) and (not(d_4_spanOpened_)):
                        d_7_og_: _dafny.Seq
                        d_8_oi_: bool
                        d_9_oc_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_7_og_ = out0_
                        d_8_oi_ = out1_
                        d_9_oc_ = out2_
                        generated = d_7_og_
                        insideConstrainedOut = d_8_oi_
                        currentConstrainedOut = d_9_oc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_4_spanOpened_ = True
                        raise _dafny.Break("0")
                    d_10_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_10_next_ = out3_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_10_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                        if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_4_spanOpened_ = True
                    pass
            pass
        d_11_maxFillSteps_: int
        d_11_maxFillSteps_ = 50
        d_12_fillSteps_: int
        d_12_fillSteps_ = 0
        with _dafny.label("1"):
            while ((insideConstrainedOut) and ((d_2_steps_) < (maxSteps))) and ((d_12_fillSteps_) < (d_11_maxFillSteps_)):
                with _dafny.c_label("1"):
                    d_13_remaining_: int
                    d_13_remaining_ = (maxSteps) - (d_2_steps_)
                    if (d_13_remaining_) <= (10):
                        raise _dafny.Break("1")
                    d_14_cg_: _dafny.Seq
                    d_15_ci_: bool
                    d_16_cc_: _dafny.Seq
                    d_17_closed_: bool
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out7_: bool
                    out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_14_cg_ = out4_
                    d_15_ci_ = out5_
                    d_16_cc_ = out6_
                    d_17_closed_ = out7_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if d_17_closed_:
                        generated = d_14_cg_
                        insideConstrainedOut = d_15_ci_
                        currentConstrainedOut = d_16_cc_
                        d_5_spanClosed_ = True
                        raise _dafny.Break("1")
                    d_18_constrainedPrompt_: _dafny.Seq
                    d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_19_penaltyTokens_: _dafny.Seq
                    d_19_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "**")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "frac")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "times")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "cdot"))])
                    d_20_next_: _dafny.Seq
                    out8_: _dafny.Seq
                    out8_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_19_penaltyTokens_, _dafny.BigRational('6e0'), 8, eosToken)
                    d_20_next_ = out8_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_12_fillSteps_ = (d_12_fillSteps_) + (1)
                    if (d_20_next_) == (eosToken):
                        raise _dafny.Break("1")
                    elif True:
                        d_21_appendedGenerated_: _dafny.Seq
                        d_22_appendedInside_: bool
                        d_23_appendedCurrent_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                        d_21_appendedGenerated_ = out9_
                        d_22_appendedInside_ = out10_
                        d_23_appendedCurrent_ = out11_
                        generated = d_21_appendedGenerated_
                        insideConstrainedOut = d_22_appendedInside_
                        currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_24_closeBudget_: int
            d_24_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_25_cg2_: _dafny.Seq
            d_26_ci2_: bool
            d_27_cc2_: _dafny.Seq
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_closeBudget_)
            d_25_cg2_ = out12_
            d_26_ci2_ = out13_
            d_27_cc2_ = out14_
            generated = d_25_cg2_
            insideConstrainedOut = d_26_ci2_
            currentConstrainedOut = d_27_cc2_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

