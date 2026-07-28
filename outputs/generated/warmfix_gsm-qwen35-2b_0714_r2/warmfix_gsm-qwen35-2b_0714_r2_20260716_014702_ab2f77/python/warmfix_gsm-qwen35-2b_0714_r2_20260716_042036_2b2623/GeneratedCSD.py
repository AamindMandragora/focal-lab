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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. When you have the final answer write: The answer is <<EXPR>> where EXPR is an arithmetic expression using the variable names from the problem and operators +, -, *, /, (, ). No LaTeX, no curly braces, no backslashes. Keep the expression short. Example: The answer is <<n * price - discount>>")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_prefixLimit_: int
        d_2_prefixLimit_ = (_dafny.euclidian_division(maxSteps, 4)) * (3)
        if ((d_2_prefixLimit_) == (0)) and ((maxSteps) > (0)):
            d_2_prefixLimit_ = maxSteps
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_2_prefixLimit_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_4_eg_: _dafny.Seq
                        d_5_ei_: bool
                        d_6_ec_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_4_eg_ = out1_
                        d_5_ei_ = out2_
                        d_6_ec_ = out3_
                        generated = d_4_eg_
                        insideConstrainedOut = d_5_ei_
                        currentConstrainedOut = d_6_ec_
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_7_fg_: _dafny.Seq
            d_8_fi_: bool
            d_9_fc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_7_fg_ = out4_
            d_8_fi_ = out5_
            d_9_fc_ = out6_
            generated = d_7_fg_
            insideConstrainedOut = d_8_fi_
            currentConstrainedOut = d_9_fc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_10_minExpr_: int
        d_10_minExpr_ = 5
        d_11_exprCount_: int
        d_11_exprCount_ = 0
        d_12_maxExpr_: int
        d_12_maxExpr_ = 80
        with _dafny.label("1"):
            while ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and ((d_11_exprCount_) < (d_12_maxExpr_)):
                with _dafny.c_label("1"):
                    if (d_11_exprCount_) >= (d_10_minExpr_):
                        d_13_cg2_: _dafny.Seq
                        d_14_ci2_: bool
                        d_15_cc2_: _dafny.Seq
                        d_16_closed2_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_13_cg2_ = out7_
                        d_14_ci2_ = out8_
                        d_15_cc2_ = out9_
                        d_16_closed2_ = out10_
                        if d_16_closed2_:
                            generated = d_13_cg2_
                            insideConstrainedOut = d_14_ci2_
                            currentConstrainedOut = d_15_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("1")
                        d_17_cp_: _dafny.Seq
                        d_17_cp_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_nx_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_17_cp_, currentConstrainedOut, eosToken)
                        d_18_nx_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_11_exprCount_ = (d_11_exprCount_) + (1)
                        if (d_18_nx_) == (eosToken):
                            raise _dafny.Break("1")
                        d_19_ag_: _dafny.Seq
                        d_20_ai_: bool
                        d_21_ac_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_nx_)
                        d_19_ag_ = out12_
                        d_20_ai_ = out13_
                        d_21_ac_ = out14_
                        generated = d_19_ag_
                        insideConstrainedOut = d_20_ai_
                        currentConstrainedOut = d_21_ac_
                    elif True:
                        d_22_cp2_: _dafny.Seq
                        d_22_cp2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_23_nx2_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_cp2_, currentConstrainedOut, eosToken)
                        d_23_nx2_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_11_exprCount_ = (d_11_exprCount_) + (1)
                        if (d_23_nx2_) == (eosToken):
                            raise _dafny.Break("1")
                        d_24_ag2_: _dafny.Seq
                        d_25_ai2_: bool
                        d_26_ac2_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: bool
                        out18_: _dafny.Seq
                        out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nx2_)
                        d_24_ag2_ = out16_
                        d_25_ai2_ = out17_
                        d_26_ac2_ = out18_
                        generated = d_24_ag2_
                        insideConstrainedOut = d_25_ai2_
                        currentConstrainedOut = d_26_ac2_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_27_cb_: int
            d_27_cb_ = (maxSteps) - (d_1_steps_)
            if (d_27_cb_) > (50):
                d_27_cb_ = 50
            d_28_wg_: _dafny.Seq
            d_29_wi_: bool
            d_30_wc_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: bool
            out21_: _dafny.Seq
            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_cb_)
            d_28_wg_ = out19_
            d_29_wi_ = out20_
            d_30_wc_ = out21_
            generated = d_28_wg_
            insideConstrainedOut = d_29_wi_
            currentConstrainedOut = d_30_wc_
            d_1_steps_ = (d_1_steps_) + (d_27_cb_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

