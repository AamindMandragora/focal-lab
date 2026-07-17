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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<YOUR QUERY>> where YOUR QUERY is a valid SQL query for the schema. No explanation, no markdown.")))
        if (d_1_steps_) < (maxSteps):
            if not(insideConstrainedOut):
                d_2_g2_: _dafny.Seq
                d_3_i2_: bool
                d_4_c2_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_2_g2_ = out0_
                d_3_i2_ = out1_
                d_4_c2_ = out2_
                generated = d_2_g2_
                insideConstrainedOut = d_3_i2_
                currentConstrainedOut = d_4_c2_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_5_constrainedPrompt_: _dafny.Seq
                d_5_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                if (parser).IsCompletePrefix(currentConstrainedOut):
                    d_6_cg_: _dafny.Seq
                    d_7_ci_: bool
                    d_8_cc_: _dafny.Seq
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_6_cg_ = out3_
                    d_7_ci_ = out4_
                    d_8_cc_ = out5_
                    generated = d_6_cg_
                    insideConstrainedOut = d_7_ci_
                    currentConstrainedOut = d_8_cc_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_9_next_: _dafny.Seq
                    out6_: _dafny.Seq
                    out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_5_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                    d_9_next_ = out6_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_9_next_) != (eosToken):
                        d_10_ag_: _dafny.Seq
                        d_11_ai_: bool
                        d_12_ac_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                        d_10_ag_ = out7_
                        d_11_ai_ = out8_
                        d_12_ac_ = out9_
                        generated = d_10_ag_
                        insideConstrainedOut = d_11_ai_
                        currentConstrainedOut = d_12_ac_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    d_13_cg_: _dafny.Seq
                    d_14_ci_: bool
                    d_15_cc_: _dafny.Seq
                    d_16_closed_: bool
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out13_: bool
                    out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_13_cg_ = out10_
                    d_14_ci_ = out11_
                    d_15_cc_ = out12_
                    d_16_closed_ = out13_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_16_closed_:
                        generated = d_13_cg_
                        insideConstrainedOut = d_14_ci_
                        currentConstrainedOut = d_15_cc_
                        raise _dafny.Break("0")
                    if ((d_1_steps_) + (2)) >= (maxSteps):
                        d_17_rg_: _dafny.Seq
                        d_18_rc_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: _dafny.Seq
                        out14_, out15_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_17_rg_ = out14_
                        d_18_rc_ = out15_
                        generated = d_17_rg_
                        currentConstrainedOut = d_18_rc_
                        insideConstrainedOut = True
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_19_cg2_: _dafny.Seq
                            d_20_ci2_: bool
                            d_21_cc2_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_cg2_ = out16_
                            d_20_ci2_ = out17_
                            d_21_cc2_ = out18_
                            generated = d_19_cg2_
                            insideConstrainedOut = d_20_ci2_
                            currentConstrainedOut = d_21_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    d_22_constrainedPrompt_: _dafny.Seq
                    d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_23_next_: _dafny.Seq
                    out19_: _dafny.Seq
                    out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                    d_23_next_ = out19_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_23_next_) == (eosToken):
                        d_24_rg_: _dafny.Seq
                        d_25_rc_: _dafny.Seq
                        out20_: _dafny.Seq
                        out21_: _dafny.Seq
                        out20_, out21_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_24_rg_ = out20_
                        d_25_rc_ = out21_
                        generated = d_24_rg_
                        currentConstrainedOut = d_25_rc_
                        insideConstrainedOut = True
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_26_cg2_: _dafny.Seq
                            d_27_ci2_: bool
                            d_28_cc2_: _dafny.Seq
                            out22_: _dafny.Seq
                            out23_: bool
                            out24_: _dafny.Seq
                            out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_26_cg2_ = out22_
                            d_27_ci2_ = out23_
                            d_28_cc2_ = out24_
                            generated = d_26_cg2_
                            insideConstrainedOut = d_27_ci2_
                            currentConstrainedOut = d_28_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_29_ag_: _dafny.Seq
                        d_30_ai_: bool
                        d_31_ac_: _dafny.Seq
                        out25_: _dafny.Seq
                        out26_: bool
                        out27_: _dafny.Seq
                        out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                        d_29_ag_ = out25_
                        d_30_ai_ = out26_
                        d_31_ac_ = out27_
                        generated = d_29_ag_
                        insideConstrainedOut = d_30_ai_
                        currentConstrainedOut = d_31_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_32_rg_: _dafny.Seq
            d_33_rc_: _dafny.Seq
            out28_: _dafny.Seq
            out29_: _dafny.Seq
            out28_, out29_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_32_rg_ = out28_
            d_33_rc_ = out29_
            generated = d_32_rg_
            currentConstrainedOut = d_33_rc_
            insideConstrainedOut = True
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_34_cg_: _dafny.Seq
                d_35_ci_: bool
                d_36_cc_: _dafny.Seq
                out30_: _dafny.Seq
                out31_: bool
                out32_: _dafny.Seq
                out30_, out31_, out32_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_34_cg_ = out30_
                d_35_ci_ = out31_
                d_36_cc_ = out32_
                generated = d_34_cg_
                insideConstrainedOut = d_35_ci_
                currentConstrainedOut = d_36_cc_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

