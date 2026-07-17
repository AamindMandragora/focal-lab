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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final answer as: The answer is <<EXPR>> where EXPR uses only variable names, numbers, +, -, *, /, //, %, (, ). No LaTeX, no {}, no **, no backslashes. Keep the expression short and concise."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hasCompletedSpan_: bool
        d_3_hasCompletedSpan_ = False
        d_4_phase1Budget_: int
        d_4_phase1Budget_ = _dafny.euclidian_division((maxSteps) * (85), 100)
        if ((d_4_phase1Budget_) == (0)) and ((maxSteps) > (0)):
            d_4_phase1Budget_ = 1
        with _dafny.label("0"):
            while ((d_2_steps_) < (d_4_phase1Budget_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_5_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_5_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_5_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_6_eg_: _dafny.Seq
                            d_7_ei_: bool
                            d_8_ec_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_6_eg_ = out1_
                            d_7_ei_ = out2_
                            d_8_ec_ = out3_
                            generated = d_6_eg_
                            insideConstrainedOut = d_7_ei_
                            currentConstrainedOut = d_8_ec_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_9_spanBudget_: int
            d_9_spanBudget_ = 40
            d_10_remainAfterPhase1_: int
            d_10_remainAfterPhase1_ = (maxSteps) - (d_2_steps_)
            if (d_9_spanBudget_) > (d_10_remainAfterPhase1_):
                d_9_spanBudget_ = d_10_remainAfterPhase1_
            if (d_9_spanBudget_) > (0):
                d_11_wg_: _dafny.Seq
                d_12_wi_: bool
                d_13_wc_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_spanBudget_)
                d_11_wg_ = out4_
                d_12_wi_ = out5_
                d_13_wc_ = out6_
                generated = d_11_wg_
                insideConstrainedOut = d_12_wi_
                currentConstrainedOut = d_13_wc_
                d_2_steps_ = (d_2_steps_) + (d_9_spanBudget_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        while (((d_2_steps_) < (d_4_phase1Budget_)) and (not(insideConstrainedOut))) and ((not(d_3_hasCompletedSpan_)) == (False)):
            d_2_steps_ = (d_2_steps_) + ((d_4_phase1Budget_) - (d_2_steps_))
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
            d_14_openCount_: int
            out7_: int
            out7_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            d_14_openCount_ = out7_
            d_15_closeCount_: int
            out8_: int
            out8_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
            d_15_closeCount_ = out8_
            if (d_14_openCount_) <= (d_15_closeCount_):
                if ((d_2_steps_) + (2)) <= (maxSteps):
                    d_16_fg_: _dafny.Seq
                    d_17_fi_: bool
                    d_18_fc_: _dafny.Seq
                    out9_: _dafny.Seq
                    out10_: bool
                    out11_: _dafny.Seq
                    out9_, out10_, out11_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_16_fg_ = out9_
                    d_17_fi_ = out10_
                    d_18_fc_ = out11_
                    generated = d_16_fg_
                    insideConstrainedOut = d_17_fi_
                    currentConstrainedOut = d_18_fc_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_19_remainBudget_: int
                    d_19_remainBudget_ = (maxSteps) - (d_2_steps_)
                    if (d_19_remainBudget_) > (40):
                        d_19_remainBudget_ = 40
                    if (d_19_remainBudget_) > (0):
                        d_20_wg_: _dafny.Seq
                        d_21_wi_: bool
                        d_22_wc_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_remainBudget_)
                        d_20_wg_ = out12_
                        d_21_wi_ = out13_
                        d_22_wc_ = out14_
                        generated = d_20_wg_
                        insideConstrainedOut = d_21_wi_
                        currentConstrainedOut = d_22_wc_
                        d_2_steps_ = (d_2_steps_) + (d_19_remainBudget_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_23_finalBudget_: int
            d_23_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_23_finalBudget_) > (0):
                d_24_wg2_: _dafny.Seq
                d_25_wi2_: bool
                d_26_wc2_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_finalBudget_)
                d_24_wg2_ = out15_
                d_25_wi2_ = out16_
                d_26_wc2_ = out17_
                generated = d_24_wg2_
                insideConstrainedOut = d_25_wi2_
                currentConstrainedOut = d_26_wc2_
                d_2_steps_ = (d_2_steps_) + (d_23_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

