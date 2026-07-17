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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate correct SQL for the question. Format: SQL: <<query>>. Write complete SQL: include FROM clause with exact table name, WHERE clause when filtering, GROUP BY + HAVING when aggregating groups, ORDER BY col DESC LIMIT 1 for max/min questions. Use COUNT(*) for total row counts; use COUNT(DISTINCT col) only when the question asks for different/unique/distinct values. Use only exact table and column names from the provided schema. Output the SQL between << and >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and ((d_1_steps_) < (5))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_2_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_2_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_2_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                        if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_3_og_: _dafny.Seq
            d_4_oi_: bool
            d_5_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_3_og_ = out1_
            d_4_oi_ = out2_
            d_5_oc_ = out3_
            generated = d_3_og_
            insideConstrainedOut = d_4_oi_
            currentConstrainedOut = d_5_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("1"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_6_constrainedPrompt_: _dafny.Seq
                    d_6_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_doSpeculate_: bool
                        d_7_doSpeculate_ = ((len(currentConstrainedOut)) <= (20)) and (((d_1_steps_) + (7)) <= (maxSteps))
                        if d_7_doSpeculate_:
                            d_8___v0_: _dafny.Seq
                            d_9___v1_: _dafny.Seq
                            d_10_hitComplete_: bool
                            d_11___v2_: bool
                            d_12_stepsUsed_: int
                            out4_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: bool
                            out8_: int
                            out4_, out5_, out6_, out7_, out8_ = (d_0_helpers_).SpeculativeConstrainedRollout(lm, parser, d_6_constrainedPrompt_, currentConstrainedOut, 5, eosToken)
                            d_8___v0_ = out4_
                            d_9___v1_ = out5_
                            d_10_hitComplete_ = out6_
                            d_11___v2_ = out7_
                            d_12_stepsUsed_ = out8_
                            d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                            if d_10_hitComplete_:
                                d_13_next_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_6_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), eosToken)
                                d_13_next_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_13_next_) == (eosToken):
                                    if (d_1_steps_) < (maxSteps):
                                        d_14_cg_: _dafny.Seq
                                        d_15_ci_: bool
                                        d_16_cc_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_14_cg_ = out10_
                                        d_15_ci_ = out11_
                                        d_16_cc_ = out12_
                                        generated = d_14_cg_
                                        insideConstrainedOut = d_15_ci_
                                        currentConstrainedOut = d_16_cc_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("1")
                                elif True:
                                    d_17_ag_: _dafny.Seq
                                    d_18_ai_: bool
                                    d_19_ac_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                    d_17_ag_ = out13_
                                    d_18_ai_ = out14_
                                    d_19_ac_ = out15_
                                    generated = d_17_ag_
                                    insideConstrainedOut = d_18_ai_
                                    currentConstrainedOut = d_19_ac_
                            elif True:
                                d_20_cg_: _dafny.Seq
                                d_21_ci_: bool
                                d_22_cc_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_20_cg_ = out16_
                                d_21_ci_ = out17_
                                d_22_cc_ = out18_
                                generated = d_20_cg_
                                insideConstrainedOut = d_21_ci_
                                currentConstrainedOut = d_22_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("1")
                        elif True:
                            d_23_cg_: _dafny.Seq
                            d_24_ci_: bool
                            d_25_cc_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_23_cg_ = out19_
                            d_24_ci_ = out20_
                            d_25_cc_ = out21_
                            generated = d_23_cg_
                            insideConstrainedOut = d_24_ci_
                            currentConstrainedOut = d_25_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("1")
                    elif True:
                        d_26_next_: _dafny.Seq
                        out22_: _dafny.Seq
                        out22_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_6_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), eosToken)
                        d_26_next_ = out22_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_26_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_27_cg_: _dafny.Seq
                                d_28_ci_: bool
                                d_29_cc_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: bool
                                out25_: _dafny.Seq
                                out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_27_cg_ = out23_
                                d_28_ci_ = out24_
                                d_29_cc_ = out25_
                                generated = d_27_cg_
                                insideConstrainedOut = d_28_ci_
                                currentConstrainedOut = d_29_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("1")
                        elif True:
                            d_30_ag_: _dafny.Seq
                            d_31_ai_: bool
                            d_32_ac_: _dafny.Seq
                            out26_: _dafny.Seq
                            out27_: bool
                            out28_: _dafny.Seq
                            out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                            d_30_ag_ = out26_
                            d_31_ai_ = out27_
                            d_32_ac_ = out28_
                            generated = d_30_ag_
                            insideConstrainedOut = d_31_ai_
                            currentConstrainedOut = d_32_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_33_remaining_: int
            d_33_remaining_ = (maxSteps) - (d_1_steps_)
            d_34_closeBudget_: int = int(0)
            if (d_33_remaining_) <= (80):
                d_34_closeBudget_ = d_33_remaining_
            elif True:
                d_34_closeBudget_ = 80
            d_35_cg_: _dafny.Seq
            d_36_ci_: bool
            d_37_cc_: _dafny.Seq
            out29_: _dafny.Seq
            out30_: bool
            out31_: _dafny.Seq
            out29_, out30_, out31_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_34_closeBudget_)
            d_35_cg_ = out29_
            d_36_ci_ = out30_
            d_37_cc_ = out31_
            generated = d_35_cg_
            insideConstrainedOut = d_36_ci_
            currentConstrainedOut = d_37_cc_
            d_1_steps_ = (d_1_steps_) + (d_34_closeBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

