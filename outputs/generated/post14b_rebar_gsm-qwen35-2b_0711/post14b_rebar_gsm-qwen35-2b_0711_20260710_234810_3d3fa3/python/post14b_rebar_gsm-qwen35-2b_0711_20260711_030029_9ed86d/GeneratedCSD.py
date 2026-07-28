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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final answer as: The answer is <<EXPR>> where EXPR is a plain arithmetic expression using only variable names, numbers, +, -, *, /, //, %, (, ). Example: <<(n1 + n2) * 7>>. No LaTeX, no {}, no backslashes, no **."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_freeBudget_: int
        if (maxSteps) > (120):
            d_3_freeBudget_ = (maxSteps) - (100)
        elif True:
            d_3_freeBudget_ = _dafny.euclidian_division(maxSteps, 2)
        if ((d_3_freeBudget_) == (0)) and ((maxSteps) > (0)):
            d_3_freeBudget_ = 1
        with _dafny.label("0"):
            while ((d_2_steps_) < (d_3_freeBudget_)) and (not(insideConstrainedOut)):
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
        d_5_spanLen_: int
        d_5_spanLen_ = 0
        d_6_maxSpanLen_: int
        d_6_maxSpanLen_ = 40
        with _dafny.label("1"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    if ((d_5_spanLen_) >= (d_6_maxSpanLen_)) and ((d_2_steps_) < (maxSteps)):
                        d_7_closeBudget_: int
                        d_7_closeBudget_ = (maxSteps) - (d_2_steps_)
                        if (d_7_closeBudget_) > (0):
                            d_8_wg_: _dafny.Seq
                            d_9_wi_: bool
                            d_10_wc_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_7_closeBudget_)
                            d_8_wg_ = out1_
                            d_9_wi_ = out2_
                            d_10_wc_ = out3_
                            generated = d_8_wg_
                            insideConstrainedOut = d_9_wi_
                            currentConstrainedOut = d_10_wc_
                            d_2_steps_ = maxSteps
                        raise _dafny.Break("1")
                    d_11_cg_: _dafny.Seq
                    d_12_ci_: bool
                    d_13_cc_: _dafny.Seq
                    d_14_closed_: bool
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out7_: bool
                    out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_11_cg_ = out4_
                    d_12_ci_ = out5_
                    d_13_cc_ = out6_
                    d_14_closed_ = out7_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if d_14_closed_:
                        generated = d_11_cg_
                        insideConstrainedOut = d_12_ci_
                        currentConstrainedOut = d_13_cc_
                        d_5_spanLen_ = 0
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_next_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('5e-1'), eosToken)
                        d_16_next_ = out8_
                        if (d_16_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_17_ag_: _dafny.Seq
                            d_18_ai_: bool
                            d_19_ac_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                            d_17_ag_ = out9_
                            d_18_ai_ = out10_
                            d_19_ac_ = out11_
                            generated = d_17_ag_
                            insideConstrainedOut = d_18_ai_
                            currentConstrainedOut = d_19_ac_
                            d_5_spanLen_ = (d_5_spanLen_) + (1)
                    pass
            pass
        d_20_phase3Budget_: int
        if ((d_3_freeBudget_) + (20)) < (maxSteps):
            d_20_phase3Budget_ = (d_3_freeBudget_) + (20)
        elif True:
            d_20_phase3Budget_ = maxSteps
        with _dafny.label("2"):
            while ((d_2_steps_) < (d_20_phase3Budget_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("2"):
                    d_21_next3_: _dafny.Seq
                    out12_: _dafny.Seq
                    out12_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_21_next3_ = out12_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_21_next3_) == (eosToken):
                        raise _dafny.Break("2")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_21_next3_]))
                        if (d_21_next3_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        d_22_spanLen2_: int
        d_22_spanLen2_ = 0
        with _dafny.label("3"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("3"):
                    if ((d_22_spanLen2_) >= (d_6_maxSpanLen_)) and ((d_2_steps_) < (maxSteps)):
                        d_23_closeBudget2_: int
                        d_23_closeBudget2_ = (maxSteps) - (d_2_steps_)
                        if (d_23_closeBudget2_) > (0):
                            d_24_wg2_: _dafny.Seq
                            d_25_wi2_: bool
                            d_26_wc2_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudget2_)
                            d_24_wg2_ = out13_
                            d_25_wi2_ = out14_
                            d_26_wc2_ = out15_
                            generated = d_24_wg2_
                            insideConstrainedOut = d_25_wi2_
                            currentConstrainedOut = d_26_wc2_
                            d_2_steps_ = maxSteps
                        raise _dafny.Break("3")
                    d_27_cg2_: _dafny.Seq
                    d_28_ci2_: bool
                    d_29_cc2_: _dafny.Seq
                    d_30_closed2_: bool
                    out16_: _dafny.Seq
                    out17_: bool
                    out18_: _dafny.Seq
                    out19_: bool
                    out16_, out17_, out18_, out19_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_27_cg2_ = out16_
                    d_28_ci2_ = out17_
                    d_29_cc2_ = out18_
                    d_30_closed2_ = out19_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if d_30_closed2_:
                        generated = d_27_cg2_
                        insideConstrainedOut = d_28_ci2_
                        currentConstrainedOut = d_29_cc2_
                        d_22_spanLen2_ = 0
                    elif True:
                        d_31_constrainedPrompt2_: _dafny.Seq
                        d_31_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_32_next2_: _dafny.Seq
                        out20_: _dafny.Seq
                        out20_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_31_constrainedPrompt2_, currentConstrainedOut, _dafny.BigRational('5e-1'), eosToken)
                        d_32_next2_ = out20_
                        if (d_32_next2_) == (eosToken):
                            raise _dafny.Break("3")
                        elif True:
                            d_33_ag2_: _dafny.Seq
                            d_34_ai2_: bool
                            d_35_ac2_: _dafny.Seq
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: _dafny.Seq
                            out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next2_)
                            d_33_ag2_ = out21_
                            d_34_ai2_ = out22_
                            d_35_ac2_ = out23_
                            generated = d_33_ag2_
                            insideConstrainedOut = d_34_ai2_
                            currentConstrainedOut = d_35_ac2_
                            d_22_spanLen2_ = (d_22_spanLen2_) + (1)
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_36_openCount_: int
            out24_: int
            out24_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            d_36_openCount_ = out24_
            d_37_closeCount_: int
            out25_: int
            out25_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
            d_37_closeCount_ = out25_
            if ((d_36_openCount_) == (0)) or ((d_36_openCount_) > (d_37_closeCount_)):
                if ((d_2_steps_) + (2)) <= (maxSteps):
                    d_38_fg_: _dafny.Seq
                    d_39_fi_: bool
                    d_40_fc_: _dafny.Seq
                    out26_: _dafny.Seq
                    out27_: bool
                    out28_: _dafny.Seq
                    out26_, out27_, out28_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_38_fg_ = out26_
                    d_39_fi_ = out27_
                    d_40_fc_ = out28_
                    generated = d_38_fg_
                    insideConstrainedOut = d_39_fi_
                    currentConstrainedOut = d_40_fc_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_41_remainBudget_: int
                    d_41_remainBudget_ = (maxSteps) - (d_2_steps_)
                    if (d_41_remainBudget_) > (0):
                        d_42_wg5_: _dafny.Seq
                        d_43_wi5_: bool
                        d_44_wc5_: _dafny.Seq
                        out29_: _dafny.Seq
                        out30_: bool
                        out31_: _dafny.Seq
                        out29_, out30_, out31_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_41_remainBudget_)
                        d_42_wg5_ = out29_
                        d_43_wi5_ = out30_
                        d_44_wc5_ = out31_
                        generated = d_42_wg5_
                        insideConstrainedOut = d_43_wi5_
                        currentConstrainedOut = d_44_wc5_
                        d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

