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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQL query. Output format: SQL: <<YOUR QUERY HERE>>. Use only tables and columns from the schema. Write a single complete SELECT statement."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_prefixSteps_: int
        d_3_prefixSteps_ = 0
        d_4_maxPrefixSteps_: int
        d_4_maxPrefixSteps_ = 10
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_prefixSteps_ = (d_3_prefixSteps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if ((d_3_prefixSteps_) >= (d_4_maxPrefixSteps_)) and ((d_2_steps_) < (maxSteps)):
                                d_6_og_: _dafny.Seq
                                d_7_oi_: bool
                                d_8_oc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_6_og_ = out1_
                                d_7_oi_ = out2_
                                d_8_oc_ = out3_
                                generated = d_6_og_
                                insideConstrainedOut = d_7_oi_
                                currentConstrainedOut = d_8_oc_
                                d_2_steps_ = (d_2_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_9_cg_: _dafny.Seq
                        d_10_ci_: bool
                        d_11_cc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_9_cg_ = out4_
                        d_10_ci_ = out5_
                        d_11_cc_ = out6_
                        generated = d_9_cg_
                        insideConstrainedOut = d_10_ci_
                        currentConstrainedOut = d_11_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_12_remaining_: int
                        d_12_remaining_ = (maxSteps) - (d_2_steps_)
                        if (d_12_remaining_) <= (30):
                            d_13_rg_: _dafny.Seq
                            d_14_rc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: _dafny.Seq
                            out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_13_rg_ = out7_
                            d_14_rc_ = out8_
                            generated = d_13_rg_
                            currentConstrainedOut = d_14_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                d_15_cg_: _dafny.Seq
                                d_16_ci_: bool
                                d_17_cc_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_15_cg_ = out9_
                                d_16_ci_ = out10_
                                d_17_cc_ = out11_
                                generated = d_15_cg_
                                insideConstrainedOut = d_16_ci_
                                currentConstrainedOut = d_17_cc_
                                d_2_steps_ = (d_2_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_18_constrainedPrompt_: _dafny.Seq
                                d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_19_next_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                                d_19_next_ = out12_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_19_next_) == (eosToken):
                                    d_20_rg2_: _dafny.Seq
                                    d_21_rc2_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out13_, out14_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_20_rg2_ = out13_
                                    d_21_rc2_ = out14_
                                    generated = d_20_rg2_
                                    currentConstrainedOut = d_21_rc2_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                        d_22_cg2_: _dafny.Seq
                                        d_23_ci2_: bool
                                        d_24_cc2_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_22_cg2_ = out15_
                                        d_23_ci2_ = out16_
                                        d_24_cc2_ = out17_
                                        generated = d_22_cg2_
                                        insideConstrainedOut = d_23_ci2_
                                        currentConstrainedOut = d_24_cc2_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_25_ag_: _dafny.Seq
                                    d_26_ai_: bool
                                    d_27_ac_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                    d_25_ag_ = out18_
                                    d_26_ai_ = out19_
                                    d_27_ac_ = out20_
                                    generated = d_25_ag_
                                    insideConstrainedOut = d_26_ai_
                                    currentConstrainedOut = d_27_ac_
                        elif True:
                            d_28_constrainedPrompt_: _dafny.Seq
                            d_28_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_29_spanLen_: int
                            d_29_spanLen_ = len(currentConstrainedOut)
                            d_30_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if (d_29_spanLen_) < (5):
                                out21_: _dafny.Seq
                                out21_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_30_next_ = out21_
                            elif True:
                                out22_: _dafny.Seq
                                out22_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_30_next_ = out22_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_30_next_) == (eosToken):
                                d_31_rg_: _dafny.Seq
                                d_32_rc_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: _dafny.Seq
                                out23_, out24_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_31_rg_ = out23_
                                d_32_rc_ = out24_
                                generated = d_31_rg_
                                currentConstrainedOut = d_32_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                    d_33_cg_: _dafny.Seq
                                    d_34_ci_: bool
                                    d_35_cc_: _dafny.Seq
                                    out25_: _dafny.Seq
                                    out26_: bool
                                    out27_: _dafny.Seq
                                    out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_33_cg_ = out25_
                                    d_34_ci_ = out26_
                                    d_35_cc_ = out27_
                                    generated = d_33_cg_
                                    insideConstrainedOut = d_34_ci_
                                    currentConstrainedOut = d_35_cc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_36_ag_: _dafny.Seq
                                d_37_ai_: bool
                                d_38_ac_: _dafny.Seq
                                out28_: _dafny.Seq
                                out29_: bool
                                out30_: _dafny.Seq
                                out28_, out29_, out30_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                                d_36_ag_ = out28_
                                d_37_ai_ = out29_
                                d_38_ac_ = out30_
                                generated = d_36_ag_
                                insideConstrainedOut = d_37_ai_
                                currentConstrainedOut = d_38_ac_
                    pass
            pass
        if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_2_steps_) < (maxSteps)):
            d_39_cg_: _dafny.Seq
            d_40_ci_: bool
            d_41_cc_: _dafny.Seq
            out31_: _dafny.Seq
            out32_: bool
            out33_: _dafny.Seq
            out31_, out32_, out33_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_39_cg_ = out31_
            d_40_ci_ = out32_
            d_41_cc_ = out33_
            generated = d_39_cg_
            insideConstrainedOut = d_40_ci_
            currentConstrainedOut = d_41_cc_
            d_2_steps_ = (d_2_steps_) + (1)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

