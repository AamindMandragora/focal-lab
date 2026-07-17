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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Think step by step. Write the final arithmetic expression using the problem's variable names (like n1, n2, p, frac, total) between << and >>. Write an expression with multiple operators, such as <<n1 * p1 + n2 * p2>> or <<total - int(total * frac)>> or <<count * (a + b + c)>>. Include all necessary arithmetic operations."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_opGroup_: _dafny.Seq
        d_3_opGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " +")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " -")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " *")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " /")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%"))])
        d_4_extendedGroups_: _dafny.Seq
        d_4_extendedGroups_ = (validTokenGroups) + (_dafny.SeqWithoutIsStrInference([d_3_opGroup_]))
        d_5_preambleLimit_: int
        if (maxSteps) >= (400):
            d_5_preambleLimit_ = 400
        elif True:
            d_5_preambleLimit_ = maxSteps
        with _dafny.label("0"):
            while ((d_2_steps_) < (d_5_preambleLimit_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_6_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_6_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_6_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                        if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_7_og_: _dafny.Seq
            d_8_oi_: bool
            d_9_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_7_og_ = out1_
            d_8_oi_ = out2_
            d_9_oc_ = out3_
            generated = d_7_og_
            insideConstrainedOut = d_8_oi_
            currentConstrainedOut = d_9_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        d_10_constrainedHardLimit_: int
        if (maxSteps) >= (100):
            d_10_constrainedHardLimit_ = (maxSteps) - (100)
        elif True:
            d_10_constrainedHardLimit_ = maxSteps
        d_11_spCount_: int
        d_11_spCount_ = 0
        with _dafny.label("1"):
            while ((d_2_steps_) < (d_10_constrainedHardLimit_)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    if (d_11_spCount_) >= (5):
                        d_12_cg_: _dafny.Seq
                        d_13_ci_: bool
                        d_14_cc_: _dafny.Seq
                        d_15_closed_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_12_cg_ = out4_
                        d_13_ci_ = out5_
                        d_14_cc_ = out6_
                        d_15_closed_ = out7_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if d_15_closed_:
                            generated = d_12_cg_
                            insideConstrainedOut = d_13_ci_
                            currentConstrainedOut = d_14_cc_
                        elif True:
                            if (insideConstrainedOut) and ((d_2_steps_) < (d_10_constrainedHardLimit_)):
                                d_16_constrainedPrompt_: _dafny.Seq
                                d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_17_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, d_4_extendedGroups_, _dafny.BigRational('4e0'), eosToken)
                                d_17_next_ = out8_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_17_next_) == (eosToken):
                                    raise _dafny.Break("1")
                                elif True:
                                    d_18_ag_: _dafny.Seq
                                    d_19_ai_: bool
                                    d_20_ac_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                    d_18_ag_ = out9_
                                    d_19_ai_ = out10_
                                    d_20_ac_ = out11_
                                    generated = d_18_ag_
                                    insideConstrainedOut = d_19_ai_
                                    currentConstrainedOut = d_20_ac_
                                    d_11_spCount_ = (d_11_spCount_) + (1)
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_next_: _dafny.Seq
                        out12_: _dafny.Seq
                        out12_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, d_4_extendedGroups_, _dafny.BigRational('4e0'), eosToken)
                        d_22_next_ = out12_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_22_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_23_ag_: _dafny.Seq
                            d_24_ai_: bool
                            d_25_ac_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                            d_23_ag_ = out13_
                            d_24_ai_ = out14_
                            d_25_ac_ = out15_
                            generated = d_23_ag_
                            insideConstrainedOut = d_24_ai_
                            currentConstrainedOut = d_25_ac_
                            d_11_spCount_ = (d_11_spCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_26_closeBudget_: int
            d_26_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_27_cg_: _dafny.Seq
            d_28_ci_: bool
            d_29_cc_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_closeBudget_)
            d_27_cg_ = out16_
            d_28_ci_ = out17_
            d_29_cc_ = out18_
            generated = d_27_cg_
            insideConstrainedOut = d_28_ci_
            currentConstrainedOut = d_29_cc_
            d_2_steps_ = (d_2_steps_) + (d_26_closeBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

