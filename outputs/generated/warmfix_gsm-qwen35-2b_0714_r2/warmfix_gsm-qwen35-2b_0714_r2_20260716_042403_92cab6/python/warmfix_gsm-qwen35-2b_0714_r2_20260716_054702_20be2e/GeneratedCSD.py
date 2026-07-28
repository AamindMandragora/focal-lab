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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the very end write: The answer is <<EXPR>> where EXPR is an arithmetic expression using numbers, variable names from the problem, and operators (+, -, *, /, parentheses). Example: <<(n * price) + tax>>. Always include at least one operator inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_hasCompletedSpan_: bool
        d_2_hasCompletedSpan_ = False
        d_3_freeSteps_: int
        d_3_freeSteps_ = _dafny.euclidian_division((maxSteps) * (3), 4)
        if (d_3_freeSteps_) > (maxSteps):
            d_3_freeSteps_ = maxSteps
        if ((d_3_freeSteps_) == (0)) and ((maxSteps) > (0)):
            d_3_freeSteps_ = 1
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_3_freeSteps_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_4_t1_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_4_t1_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_4_t1_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_t1_]))
                    if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_5_eg1_: _dafny.Seq
                        d_6_ei1_: bool
                        d_7_ec1_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_5_eg1_ = out1_
                        d_6_ei1_ = out2_
                        d_7_ec1_ = out3_
                        generated = d_5_eg1_
                        insideConstrainedOut = d_6_ei1_
                        currentConstrainedOut = d_7_ec1_
                    pass
            pass
        with _dafny.label("1"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_8_cg2_: _dafny.Seq
                    d_9_ci2_: bool
                    d_10_cc2_: _dafny.Seq
                    d_11_cl2_: bool
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out7_: bool
                    out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_8_cg2_ = out4_
                    d_9_ci2_ = out5_
                    d_10_cc2_ = out6_
                    d_11_cl2_ = out7_
                    if d_11_cl2_:
                        generated = d_8_cg2_
                        insideConstrainedOut = d_9_ci2_
                        currentConstrainedOut = d_10_cc2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_hasCompletedSpan_ = True
                    elif True:
                        d_12_cp2_: _dafny.Seq
                        d_12_cp2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_t2_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_cp2_, currentConstrainedOut, eosToken)
                        d_13_t2_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_t2_) == (eosToken):
                            raise _dafny.Break("1")
                        d_14_g2_: _dafny.Seq
                        d_15_i2_: bool
                        d_16_c2_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_t2_)
                        d_14_g2_ = out9_
                        d_15_i2_ = out10_
                        d_16_c2_ = out11_
                        generated = d_14_g2_
                        insideConstrainedOut = d_15_i2_
                        currentConstrainedOut = d_16_c2_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_17_rem3_: int
            d_17_rem3_ = (maxSteps) - (d_1_steps_)
            d_18_wg3_: _dafny.Seq
            d_19_wi3_: bool
            d_20_wc3_: _dafny.Seq
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_rem3_)
            d_18_wg3_ = out12_
            d_19_wi3_ = out13_
            d_20_wc3_ = out14_
            generated = d_18_wg3_
            insideConstrainedOut = d_19_wi3_
            currentConstrainedOut = d_20_wc3_
            d_1_steps_ = (d_1_steps_) + (d_17_rem3_)
            if not(insideConstrainedOut):
                d_2_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_2_hasCompletedSpan_))) and ((d_1_steps_) < (maxSteps)):
            d_21_fg4_: _dafny.Seq
            d_22_fi4_: bool
            d_23_fc4_: _dafny.Seq
            out15_: _dafny.Seq
            out16_: bool
            out17_: _dafny.Seq
            out15_, out16_, out17_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_21_fg4_ = out15_
            d_22_fi4_ = out16_
            d_23_fc4_ = out17_
            generated = d_21_fg4_
            insideConstrainedOut = d_22_fi4_
            currentConstrainedOut = d_23_fc4_
            d_1_steps_ = (d_1_steps_) + (1)
            with _dafny.label("5_0"):
                while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                    with _dafny.c_label("5_0"):
                        d_24_cg4_: _dafny.Seq
                        d_25_ci4_: bool
                        d_26_cc4_: _dafny.Seq
                        d_27_cl4_: bool
                        out18_: _dafny.Seq
                        out19_: bool
                        out20_: _dafny.Seq
                        out21_: bool
                        out18_, out19_, out20_, out21_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_24_cg4_ = out18_
                        d_25_ci4_ = out19_
                        d_26_cc4_ = out20_
                        d_27_cl4_ = out21_
                        if d_27_cl4_:
                            generated = d_24_cg4_
                            insideConstrainedOut = d_25_ci4_
                            currentConstrainedOut = d_26_cc4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_hasCompletedSpan_ = True
                        elif True:
                            d_28_cp4_: _dafny.Seq
                            d_28_cp4_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_29_t4_: _dafny.Seq
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_28_cp4_, currentConstrainedOut, eosToken)
                            d_29_t4_ = out22_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_29_t4_) == (eosToken):
                                raise _dafny.Break("5_0")
                            d_30_g4_: _dafny.Seq
                            d_31_i4_: bool
                            d_32_c4_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: bool
                            out25_: _dafny.Seq
                            out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_t4_)
                            d_30_g4_ = out23_
                            d_31_i4_ = out24_
                            d_32_c4_ = out25_
                            generated = d_30_g4_
                            insideConstrainedOut = d_31_i4_
                            currentConstrainedOut = d_32_c4_
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_33_rem4b_: int
                d_33_rem4b_ = (maxSteps) - (d_1_steps_)
                d_34_wg4_: _dafny.Seq
                d_35_wi4_: bool
                d_36_wc4_: _dafny.Seq
                out26_: _dafny.Seq
                out27_: bool
                out28_: _dafny.Seq
                out26_, out27_, out28_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_rem4b_)
                d_34_wg4_ = out26_
                d_35_wi4_ = out27_
                d_36_wc4_ = out28_
                generated = d_34_wg4_
                insideConstrainedOut = d_35_wi4_
                currentConstrainedOut = d_36_wc4_
                d_1_steps_ = (d_1_steps_) + (d_33_rem4b_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

