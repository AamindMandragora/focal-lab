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
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQL query. Output: SQL: <<your SQL query here>>. Use only schema tables and columns. No markdown, no explanation."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        with _dafny.label("0"):
            while ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_4_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_4_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_4_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        d_5_reserveForClose_: int
        d_5_reserveForClose_ = 50
        d_6_innerBudget_: int
        if (maxSteps) > ((d_2_steps_) + (d_5_reserveForClose_)):
            d_6_innerBudget_ = ((maxSteps) - (d_2_steps_)) - (d_5_reserveForClose_)
        elif True:
            d_6_innerBudget_ = 0
        d_7_innerSteps_: int
        d_7_innerSteps_ = 0
        while (insideConstrainedOut) and ((d_7_innerSteps_) < (d_6_innerBudget_)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_8_cg_: _dafny.Seq
                d_9_ci_: bool
                d_10_cc_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_8_cg_ = out1_
                d_9_ci_ = out2_
                d_10_cc_ = out3_
                generated = d_8_cg_
                insideConstrainedOut = d_9_ci_
                currentConstrainedOut = d_10_cc_
                d_7_innerSteps_ = (d_7_innerSteps_) + (1)
            elif True:
                d_11_constrainedPrompt_: _dafny.Seq
                d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_12_validCount_: int
                out4_: int
                out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                d_12_validCount_ = out4_
                d_13_next_: _dafny.Seq
                d_13_next_ = eosToken
                if (d_12_validCount_) <= (d_3_narrowThreshold_):
                    out5_: _dafny.Seq
                    out5_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), eosToken)
                    d_13_next_ = out5_
                elif True:
                    out6_: _dafny.Seq
                    out6_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                    d_13_next_ = out6_
                d_7_innerSteps_ = (d_7_innerSteps_) + (1)
                if (d_13_next_) == (eosToken):
                    d_7_innerSteps_ = d_6_innerBudget_
                elif True:
                    d_14_ag_: _dafny.Seq
                    d_15_ai_: bool
                    d_16_ac_: _dafny.Seq
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                    d_14_ag_ = out7_
                    d_15_ai_ = out8_
                    d_16_ac_ = out9_
                    generated = d_14_ag_
                    insideConstrainedOut = d_15_ai_
                    currentConstrainedOut = d_16_ac_
        d_2_steps_ = (d_2_steps_) + (d_7_innerSteps_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_17_closeBudget_: int
            d_17_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_18_cg_: _dafny.Seq
            d_19_ci_: bool
            d_20_cc_: _dafny.Seq
            out10_: _dafny.Seq
            out11_: bool
            out12_: _dafny.Seq
            out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
            d_18_cg_ = out10_
            d_19_ci_ = out11_
            d_20_cc_ = out12_
            generated = d_18_cg_
            insideConstrainedOut = d_19_ci_
            currentConstrainedOut = d_20_cc_
            d_2_steps_ = maxSteps
        if ((maxSteps) > (0)) and ((d_2_steps_) == (0)):
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_21_closeBudget2_: int
                d_21_closeBudget2_ = (maxSteps) - (d_2_steps_)
                d_22_cg2_: _dafny.Seq
                d_23_ci2_: bool
                d_24_cc2_: _dafny.Seq
                out13_: _dafny.Seq
                out14_: bool
                out15_: _dafny.Seq
                out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_closeBudget2_)
                d_22_cg2_ = out13_
                d_23_ci2_ = out14_
                d_24_cc2_ = out15_
                generated = d_22_cg2_
                insideConstrainedOut = d_23_ci2_
                currentConstrainedOut = d_24_cc2_
                d_2_steps_ = maxSteps
            elif (d_2_steps_) < (maxSteps):
                d_25_next2_: _dafny.Seq
                out16_: _dafny.Seq
                out16_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_25_next2_ = out16_
                d_2_steps_ = (d_2_steps_) + (1)
                if (d_25_next2_) != (eosToken):
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_25_next2_]))
                    if (d_25_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

