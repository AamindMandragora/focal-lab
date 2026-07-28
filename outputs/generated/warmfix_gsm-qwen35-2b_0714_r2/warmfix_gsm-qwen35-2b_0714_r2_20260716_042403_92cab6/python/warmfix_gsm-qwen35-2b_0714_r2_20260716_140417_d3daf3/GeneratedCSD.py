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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the end write exactly: The answer is EXPR. EXPR rules: use exact variable names from the problem without braces, operators +,-,*,/,(,), use int() for integer division or fraction multiplication. Single compact expression only. Example: int(n * frac_1) + base_cost")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((maxSteps) > (0)):
            d_2_p1b_: int
            d_2_p1b_ = _dafny.euclidian_division(maxSteps, 2)
            if (d_2_p1b_) == (0):
                d_2_p1b_ = maxSteps
            d_3_cg1_: _dafny.Seq
            d_4_soo1_: bool
            d_5_soe1_: bool
            d_6_su1_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_p1b_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_3_cg1_ = out0_
            d_4_soo1_ = out1_
            d_5_soe1_ = out2_
            d_6_su1_ = out3_
            generated = d_3_cg1_
            d_1_steps_ = (d_1_steps_) + (d_6_su1_)
            if (d_4_soo1_) and (not(d_5_soe1_)):
                d_7_eg1_: _dafny.Seq
                d_8_ei1_: bool
                d_9_ec1_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_7_eg1_ = out4_
                d_8_ei1_ = out5_
                d_9_ec1_ = out6_
                generated = d_7_eg1_
                insideConstrainedOut = d_8_ei1_
                currentConstrainedOut = d_9_ec1_
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_10_b2_: int
            d_10_b2_ = (maxSteps) - (d_1_steps_)
            d_11_wg2_: _dafny.Seq
            d_12_wi2_: bool
            d_13_wc2_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_b2_)
            d_11_wg2_ = out7_
            d_12_wi2_ = out8_
            d_13_wc2_ = out9_
            generated = d_11_wg2_
            insideConstrainedOut = d_12_wi2_
            currentConstrainedOut = d_13_wc2_
            d_1_steps_ = (d_1_steps_) + (d_10_b2_)
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_14_rem3_: int
            d_14_rem3_ = (maxSteps) - (d_1_steps_)
            d_15_p3b_: int
            d_15_p3b_ = _dafny.euclidian_division(d_14_rem3_, 2)
            if (d_15_p3b_) == (0):
                d_15_p3b_ = d_14_rem3_
            d_16_cg3_: _dafny.Seq
            d_17_soo3_: bool
            d_18_soe3_: bool
            d_19_su3_: int
            out10_: _dafny.Seq
            out11_: bool
            out12_: bool
            out13_: int
            out10_, out11_, out12_, out13_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_15_p3b_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_16_cg3_ = out10_
            d_17_soo3_ = out11_
            d_18_soe3_ = out12_
            d_19_su3_ = out13_
            generated = d_16_cg3_
            d_1_steps_ = (d_1_steps_) + (d_19_su3_)
            if (d_17_soo3_) and (not(d_18_soe3_)):
                d_20_eg3_: _dafny.Seq
                d_21_ei3_: bool
                d_22_ec3_: _dafny.Seq
                out14_: _dafny.Seq
                out15_: bool
                out16_: _dafny.Seq
                out14_, out15_, out16_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_20_eg3_ = out14_
                d_21_ei3_ = out15_
                d_22_ec3_ = out16_
                generated = d_20_eg3_
                insideConstrainedOut = d_21_ei3_
                currentConstrainedOut = d_22_ec3_
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_23_b3_: int
                d_23_b3_ = (maxSteps) - (d_1_steps_)
                d_24_wg3_: _dafny.Seq
                d_25_wi3_: bool
                d_26_wc3_: _dafny.Seq
                out17_: _dafny.Seq
                out18_: bool
                out19_: _dafny.Seq
                out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_b3_)
                d_24_wg3_ = out17_
                d_25_wi3_ = out18_
                d_26_wc3_ = out19_
                generated = d_24_wg3_
                insideConstrainedOut = d_25_wi3_
                currentConstrainedOut = d_26_wc3_
                d_1_steps_ = (d_1_steps_) + (d_23_b3_)
        if (not(insideConstrainedOut)) and (((d_1_steps_) + (1)) < (maxSteps)):
            d_27_fg4_: _dafny.Seq
            d_28_fi4_: bool
            d_29_fc4_: _dafny.Seq
            out20_: _dafny.Seq
            out21_: bool
            out22_: _dafny.Seq
            out20_, out21_, out22_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_27_fg4_ = out20_
            d_28_fi4_ = out21_
            d_29_fc4_ = out22_
            generated = d_27_fg4_
            insideConstrainedOut = d_28_fi4_
            currentConstrainedOut = d_29_fc4_
            d_1_steps_ = (d_1_steps_) + (1)
            d_30_b4_: int
            d_30_b4_ = (maxSteps) - (d_1_steps_)
            d_31_wg4_: _dafny.Seq
            d_32_wi4_: bool
            d_33_wc4_: _dafny.Seq
            out23_: _dafny.Seq
            out24_: bool
            out25_: _dafny.Seq
            out23_, out24_, out25_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_30_b4_)
            d_31_wg4_ = out23_
            d_32_wi4_ = out24_
            d_33_wc4_ = out25_
            generated = d_31_wg4_
            insideConstrainedOut = d_32_wi4_
            currentConstrainedOut = d_33_wc4_
            d_1_steps_ = (d_1_steps_) + (d_30_b4_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_34_b5_: int
            d_34_b5_ = (maxSteps) - (d_1_steps_)
            d_35_wg5_: _dafny.Seq
            d_36_wi5_: bool
            d_37_wc5_: _dafny.Seq
            out26_: _dafny.Seq
            out27_: bool
            out28_: _dafny.Seq
            out26_, out27_, out28_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_34_b5_)
            d_35_wg5_ = out26_
            d_36_wi5_ = out27_
            d_37_wc5_ = out28_
            generated = d_35_wg5_
            insideConstrainedOut = d_36_wi5_
            currentConstrainedOut = d_37_wc5_
            d_1_steps_ = (d_1_steps_) + (d_34_b5_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

