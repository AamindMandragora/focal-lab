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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. End with: The answer is <<EXPR>> where EXPR uses only variable names, numbers, +, -, *, /, //, %, (, ). No LaTeX, no {}, no backslashes, no **."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_freeBudget_: int
        d_3_freeBudget_ = _dafny.euclidian_division(maxSteps, 3)
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
        d_5_maxSpanSteps_: int
        d_5_maxSpanSteps_ = 60
        d_6_spanSteps_: int
        d_6_spanSteps_ = 0
        with _dafny.label("1"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    if ((d_6_spanSteps_) >= (d_5_maxSpanSteps_)) or ((len(currentConstrainedOut)) > (80)):
                        d_7_rb__gen_: _dafny.Seq
                        d_8_rb__cur_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: _dafny.Seq
                        out1_, out2_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_7_rb__gen_ = out1_
                        d_8_rb__cur_ = out2_
                        generated = d_7_rb__gen_
                        currentConstrainedOut = d_8_rb__cur_
                        d_9_closeBudget_: int
                        d_9_closeBudget_ = (maxSteps) - (d_2_steps_)
                        if (d_9_closeBudget_) > (0):
                            d_10_wg_: _dafny.Seq
                            d_11_wi_: bool
                            d_12_wc_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_closeBudget_)
                            d_10_wg_ = out3_
                            d_11_wi_ = out4_
                            d_12_wc_ = out5_
                            generated = d_10_wg_
                            insideConstrainedOut = d_11_wi_
                            currentConstrainedOut = d_12_wc_
                            d_2_steps_ = maxSteps
                        raise _dafny.Break("1")
                    d_13_cg_: _dafny.Seq
                    d_14_ci_: bool
                    d_15_cc_: _dafny.Seq
                    d_16_closed_: bool
                    out6_: _dafny.Seq
                    out7_: bool
                    out8_: _dafny.Seq
                    out9_: bool
                    out6_, out7_, out8_, out9_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_13_cg_ = out6_
                    d_14_ci_ = out7_
                    d_15_cc_ = out8_
                    d_16_closed_ = out9_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_6_spanSteps_ = (d_6_spanSteps_) + (1)
                    if d_16_closed_:
                        generated = d_13_cg_
                        insideConstrainedOut = d_14_ci_
                        currentConstrainedOut = d_15_cc_
                        d_6_spanSteps_ = 0
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_18_next_ = out10_
                        if (d_18_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_19_ag_: _dafny.Seq
                            d_20_ai_: bool
                            d_21_ac_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_19_ag_ = out11_
                            d_20_ai_ = out12_
                            d_21_ac_ = out13_
                            generated = d_19_ag_
                            insideConstrainedOut = d_20_ai_
                            currentConstrainedOut = d_21_ac_
                    pass
            pass
        d_22_phase3End_: int
        d_22_phase3End_ = (d_2_steps_) + (d_3_freeBudget_)
        if (d_22_phase3End_) > (maxSteps):
            d_22_phase3End_ = maxSteps
        with _dafny.label("2"):
            while ((d_2_steps_) < (d_22_phase3End_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("2"):
                    d_23_next3_: _dafny.Seq
                    out14_: _dafny.Seq
                    out14_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_23_next3_ = out14_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_23_next3_) == (eosToken):
                        raise _dafny.Break("2")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_23_next3_]))
                        if (d_23_next3_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        d_24_spanSteps2_: int
        d_24_spanSteps2_ = 0
        with _dafny.label("3"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("3"):
                    if ((d_24_spanSteps2_) >= (d_5_maxSpanSteps_)) or ((len(currentConstrainedOut)) > (80)):
                        d_25_rb2__gen_: _dafny.Seq
                        d_26_rb2__cur_: _dafny.Seq
                        out15_: _dafny.Seq
                        out16_: _dafny.Seq
                        out15_, out16_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_25_rb2__gen_ = out15_
                        d_26_rb2__cur_ = out16_
                        generated = d_25_rb2__gen_
                        currentConstrainedOut = d_26_rb2__cur_
                        d_27_closeBudget2_: int
                        d_27_closeBudget2_ = (maxSteps) - (d_2_steps_)
                        if (d_27_closeBudget2_) > (0):
                            d_28_wg2_: _dafny.Seq
                            d_29_wi2_: bool
                            d_30_wc2_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudget2_)
                            d_28_wg2_ = out17_
                            d_29_wi2_ = out18_
                            d_30_wc2_ = out19_
                            generated = d_28_wg2_
                            insideConstrainedOut = d_29_wi2_
                            currentConstrainedOut = d_30_wc2_
                            d_2_steps_ = maxSteps
                        raise _dafny.Break("3")
                    d_31_cg4_: _dafny.Seq
                    d_32_ci4_: bool
                    d_33_cc4_: _dafny.Seq
                    d_34_closed4_: bool
                    out20_: _dafny.Seq
                    out21_: bool
                    out22_: _dafny.Seq
                    out23_: bool
                    out20_, out21_, out22_, out23_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_31_cg4_ = out20_
                    d_32_ci4_ = out21_
                    d_33_cc4_ = out22_
                    d_34_closed4_ = out23_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_24_spanSteps2_ = (d_24_spanSteps2_) + (1)
                    if d_34_closed4_:
                        generated = d_31_cg4_
                        insideConstrainedOut = d_32_ci4_
                        currentConstrainedOut = d_33_cc4_
                        d_24_spanSteps2_ = 0
                    elif True:
                        d_35_constrainedPrompt4_: _dafny.Seq
                        d_35_constrainedPrompt4_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_36_next4_: _dafny.Seq
                        out24_: _dafny.Seq
                        out24_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_35_constrainedPrompt4_, currentConstrainedOut, eosToken)
                        d_36_next4_ = out24_
                        if (d_36_next4_) == (eosToken):
                            raise _dafny.Break("3")
                        elif True:
                            d_37_ag4_: _dafny.Seq
                            d_38_ai4_: bool
                            d_39_ac4_: _dafny.Seq
                            out25_: _dafny.Seq
                            out26_: bool
                            out27_: _dafny.Seq
                            out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_36_next4_)
                            d_37_ag4_ = out25_
                            d_38_ai4_ = out26_
                            d_39_ac4_ = out27_
                            generated = d_37_ag4_
                            insideConstrainedOut = d_38_ai4_
                            currentConstrainedOut = d_39_ac4_
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_40_openCount_: int
            out28_: int
            out28_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            d_40_openCount_ = out28_
            d_41_closeCount_: int
            out29_: int
            out29_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
            d_41_closeCount_ = out29_
            if ((d_40_openCount_) == (0)) or ((d_40_openCount_) > (d_41_closeCount_)):
                if ((d_2_steps_) + (2)) <= (maxSteps):
                    d_42_fg_: _dafny.Seq
                    d_43_fi_: bool
                    d_44_fc_: _dafny.Seq
                    out30_: _dafny.Seq
                    out31_: bool
                    out32_: _dafny.Seq
                    out30_, out31_, out32_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_42_fg_ = out30_
                    d_43_fi_ = out31_
                    d_44_fc_ = out32_
                    generated = d_42_fg_
                    insideConstrainedOut = d_43_fi_
                    currentConstrainedOut = d_44_fc_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_45_remainBudget_: int
                    d_45_remainBudget_ = (maxSteps) - (d_2_steps_)
                    if (d_45_remainBudget_) > (0):
                        d_46_wg5_: _dafny.Seq
                        d_47_wi5_: bool
                        d_48_wc5_: _dafny.Seq
                        out33_: _dafny.Seq
                        out34_: bool
                        out35_: _dafny.Seq
                        out33_, out34_, out35_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_45_remainBudget_)
                        d_46_wg5_ = out33_
                        d_47_wi5_ = out34_
                        d_48_wc5_ = out35_
                        generated = d_46_wg5_
                        insideConstrainedOut = d_47_wi5_
                        currentConstrainedOut = d_48_wc5_
                        d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

