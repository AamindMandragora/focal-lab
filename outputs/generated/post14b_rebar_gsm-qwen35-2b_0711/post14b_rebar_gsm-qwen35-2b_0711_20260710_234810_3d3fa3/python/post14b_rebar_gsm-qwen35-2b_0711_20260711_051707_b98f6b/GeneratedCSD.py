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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the very end, write: The answer is <<EXPR>> where EXPR is a concise arithmetic expression using only variable names from the problem, numbers, and operators +, -, *, /, //, %, (, ). No LaTeX, no {}, no **, no $, no backslashes."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hasCompletedSpan_: bool
        d_3_hasCompletedSpan_ = False
        d_4_chunkSize_: int
        d_4_chunkSize_ = 30
        d_5_phase1Limit_: int
        d_5_phase1Limit_ = _dafny.euclidian_division((maxSteps) * (82), 100)
        if ((d_5_phase1Limit_) == (0)) and ((maxSteps) > (0)):
            d_5_phase1Limit_ = 1
        with _dafny.label("0"):
            while (((d_2_steps_) < (d_5_phase1Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("0"):
                    d_6_actualChunk_: int
                    d_6_actualChunk_ = d_4_chunkSize_
                    if ((d_2_steps_) + (d_6_actualChunk_)) > (d_5_phase1Limit_):
                        d_6_actualChunk_ = (d_5_phase1Limit_) - (d_2_steps_)
                    if (d_6_actualChunk_) == (0):
                        raise _dafny.Break("0")
                    d_7_cg_: _dafny.Seq
                    d_8_stoppedOnOpen_: bool
                    d_9_stoppedOnEos_: bool
                    d_10_stepsUsed_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_actualChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_7_cg_ = out0_
                    d_8_stoppedOnOpen_ = out1_
                    d_9_stoppedOnEos_ = out2_
                    d_10_stepsUsed_ = out3_
                    generated = d_7_cg_
                    d_2_steps_ = (d_2_steps_) + (d_10_stepsUsed_)
                    if d_9_stoppedOnEos_:
                        raise _dafny.Break("0")
                    if d_8_stoppedOnOpen_:
                        d_11_eg_: _dafny.Seq
                        d_12_ei_: bool
                        d_13_ec_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_11_eg_ = out4_
                        d_12_ei_ = out5_
                        d_13_ec_ = out6_
                        generated = d_11_eg_
                        insideConstrainedOut = d_12_ei_
                        currentConstrainedOut = d_13_ec_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_14_spanBudget_: int
            d_14_spanBudget_ = 50
            d_15_remaining_: int
            d_15_remaining_ = (maxSteps) - (d_2_steps_)
            if (d_14_spanBudget_) > (d_15_remaining_):
                d_14_spanBudget_ = d_15_remaining_
            if (d_14_spanBudget_) > (0):
                d_16_wg_: _dafny.Seq
                d_17_wi_: bool
                d_18_wc_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_spanBudget_)
                d_16_wg_ = out7_
                d_17_wi_ = out8_
                d_18_wc_ = out9_
                generated = d_16_wg_
                insideConstrainedOut = d_17_wi_
                currentConstrainedOut = d_18_wc_
                d_2_steps_ = (d_2_steps_) + (d_14_spanBudget_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        with _dafny.label("1"):
            while (((d_2_steps_) < (d_5_phase1Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("1"):
                    d_19_next_: _dafny.Seq
                    out10_: _dafny.Seq
                    out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_19_next_ = out10_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_19_next_) == (eosToken):
                        raise _dafny.Break("1")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_19_next_]))
                    if (d_19_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_20_eg_: _dafny.Seq
                        d_21_ei_: bool
                        d_22_ec_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_20_eg_ = out11_
                        d_21_ei_ = out12_
                        d_22_ec_ = out13_
                        generated = d_20_eg_
                        insideConstrainedOut = d_21_ei_
                        currentConstrainedOut = d_22_ec_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_23_spanBudget2_: int
            d_23_spanBudget2_ = 50
            d_24_remaining2_: int
            d_24_remaining2_ = (maxSteps) - (d_2_steps_)
            if (d_23_spanBudget2_) > (d_24_remaining2_):
                d_23_spanBudget2_ = d_24_remaining2_
            if (d_23_spanBudget2_) > (0):
                d_25_wg2_: _dafny.Seq
                d_26_wi2_: bool
                d_27_wc2_: _dafny.Seq
                out14_: _dafny.Seq
                out15_: bool
                out16_: _dafny.Seq
                out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_spanBudget2_)
                d_25_wg2_ = out14_
                d_26_wi2_ = out15_
                d_27_wc2_ = out16_
                generated = d_25_wg2_
                insideConstrainedOut = d_26_wi2_
                currentConstrainedOut = d_27_wc2_
                d_2_steps_ = (d_2_steps_) + (d_23_spanBudget2_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
            d_28_openCount_: int
            out17_: int
            out17_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            d_28_openCount_ = out17_
            d_29_closeCount_: int
            out18_: int
            out18_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
            d_29_closeCount_ = out18_
            if (d_28_openCount_) <= (d_29_closeCount_):
                if ((d_2_steps_) + (2)) <= (maxSteps):
                    d_30_fg_: _dafny.Seq
                    d_31_fi_: bool
                    d_32_fc_: _dafny.Seq
                    out19_: _dafny.Seq
                    out20_: bool
                    out21_: _dafny.Seq
                    out19_, out20_, out21_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_30_fg_ = out19_
                    d_31_fi_ = out20_
                    d_32_fc_ = out21_
                    generated = d_30_fg_
                    insideConstrainedOut = d_31_fi_
                    currentConstrainedOut = d_32_fc_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_33_remainBudget_: int
                    d_33_remainBudget_ = (maxSteps) - (d_2_steps_)
                    if (d_33_remainBudget_) > (50):
                        d_33_remainBudget_ = 50
                    if (d_33_remainBudget_) > (0):
                        d_34_wg3_: _dafny.Seq
                        d_35_wi3_: bool
                        d_36_wc3_: _dafny.Seq
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: _dafny.Seq
                        out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_remainBudget_)
                        d_34_wg3_ = out22_
                        d_35_wi3_ = out23_
                        d_36_wc3_ = out24_
                        generated = d_34_wg3_
                        insideConstrainedOut = d_35_wi3_
                        currentConstrainedOut = d_36_wc3_
                        d_2_steps_ = (d_2_steps_) + (d_33_remainBudget_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_37_finalBudget_: int
            d_37_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_37_finalBudget_) > (0):
                d_38_wg4_: _dafny.Seq
                d_39_wi4_: bool
                d_40_wc4_: _dafny.Seq
                out25_: _dafny.Seq
                out26_: bool
                out27_: _dafny.Seq
                out25_, out26_, out27_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_37_finalBudget_)
                d_38_wg4_ = out25_
                d_39_wi4_ = out26_
                d_40_wc4_ = out27_
                generated = d_38_wg4_
                insideConstrainedOut = d_39_wi4_
                currentConstrainedOut = d_40_wc4_
                d_2_steps_ = (d_2_steps_) + (d_37_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

