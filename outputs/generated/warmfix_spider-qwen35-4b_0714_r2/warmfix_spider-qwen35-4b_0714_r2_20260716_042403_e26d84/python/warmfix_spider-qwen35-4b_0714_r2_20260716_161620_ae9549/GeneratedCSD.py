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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output SQL: <<query>> where query is a complete SQL SELECT statement that always includes a FROM clause. Use ONLY exact table and column names from the schema. Structure: SELECT ... FROM <table> [JOIN ...] [WHERE ...] [GROUP BY ...] [HAVING ...] [ORDER BY ...] [LIMIT ...]. No table aliases unless explicitly defined in the schema. No explanations outside the delimiters.")))
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
                    d_6_fromU_: int
                    out4_: int
                    out4_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))
                    d_6_fromU_ = out4_
                    d_7_fromL_: int
                    out5_: int
                    out5_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from")))
                    d_7_fromL_ = out5_
                    d_8_fromSU_: int
                    out6_: int
                    out6_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " FROM")))
                    d_8_fromSU_ = out6_
                    d_9_fromSL_: int
                    out7_: int
                    out7_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " from")))
                    d_9_fromSL_ = out7_
                    d_10_hasFrom_: bool
                    d_10_hasFrom_ = ((((d_6_fromU_) + (d_7_fromL_)) + (d_8_fromSU_)) + (d_9_fromSL_)) > (0)
                    d_11_constrainedPrompt_: _dafny.Seq
                    d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and (d_10_hasFrom_):
                        d_12_cg_: _dafny.Seq
                        d_13_ci_: bool
                        d_14_cc_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_cg_ = out8_
                        d_13_ci_ = out9_
                        d_14_cc_ = out10_
                        generated = d_12_cg_
                        insideConstrainedOut = d_13_ci_
                        currentConstrainedOut = d_14_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("1")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_next_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " from"))]), _dafny.BigRational('5e0'), eosToken)
                        d_15_next_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            if (d_1_steps_) < (maxSteps):
                                d_16_cg_: _dafny.Seq
                                d_17_ci_: bool
                                d_18_cc_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_16_cg_ = out12_
                                d_17_ci_ = out13_
                                d_18_cc_ = out14_
                                generated = d_16_cg_
                                insideConstrainedOut = d_17_ci_
                                currentConstrainedOut = d_18_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("1")
                        elif True:
                            d_19_ag_: _dafny.Seq
                            d_20_ai_: bool
                            d_21_ac_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                            d_19_ag_ = out15_
                            d_20_ai_ = out16_
                            d_21_ac_ = out17_
                            generated = d_19_ag_
                            insideConstrainedOut = d_20_ai_
                            currentConstrainedOut = d_21_ac_
                    elif True:
                        d_22_next_: _dafny.Seq
                        out18_: _dafny.Seq
                        out18_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), eosToken)
                        d_22_next_ = out18_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_22_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
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
                            d_26_ag_: _dafny.Seq
                            d_27_ai_: bool
                            d_28_ac_: _dafny.Seq
                            out22_: _dafny.Seq
                            out23_: bool
                            out24_: _dafny.Seq
                            out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                            d_26_ag_ = out22_
                            d_27_ai_ = out23_
                            d_28_ac_ = out24_
                            generated = d_26_ag_
                            insideConstrainedOut = d_27_ai_
                            currentConstrainedOut = d_28_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_29_remaining_: int
            d_29_remaining_ = (maxSteps) - (d_1_steps_)
            d_30_closeBudget_: int = int(0)
            if (d_29_remaining_) <= (80):
                d_30_closeBudget_ = d_29_remaining_
            elif True:
                d_30_closeBudget_ = 80
            d_31_cg_: _dafny.Seq
            d_32_ci_: bool
            d_33_cc_: _dafny.Seq
            out25_: _dafny.Seq
            out26_: bool
            out27_: _dafny.Seq
            out25_, out26_, out27_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_30_closeBudget_)
            d_31_cg_ = out25_
            d_32_ci_ = out26_
            d_33_cc_ = out27_
            generated = d_31_cg_
            insideConstrainedOut = d_32_ci_
            currentConstrainedOut = d_33_cc_
            d_1_steps_ = (d_1_steps_) + (d_30_closeBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

