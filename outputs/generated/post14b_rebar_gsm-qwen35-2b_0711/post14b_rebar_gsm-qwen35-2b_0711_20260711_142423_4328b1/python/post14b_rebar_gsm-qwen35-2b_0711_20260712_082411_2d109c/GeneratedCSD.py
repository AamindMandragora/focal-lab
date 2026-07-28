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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Reason through the problem carefully. At the very end, write ONLY your final arithmetic formula as a Python expression between << and >>. The expression must exactly match the formula you derived in your reasoning above. Use the variable names from the problem (like n1, n2, p, frac, total, etc.) and Python operators +, -, *, /, //, int(). Example: if you derived 'total = n1 * p1 + n2 * p2', write <<n1 * p1 + n2 * p2>>. Do NOT write a single variable name alone."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_preambleLimit_: int
        if (maxSteps) >= (400):
            d_3_preambleLimit_ = 400
        elif True:
            d_3_preambleLimit_ = maxSteps
        with _dafny.label("0"):
            while ((d_2_steps_) < (d_3_preambleLimit_)) and (not(insideConstrainedOut)):
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
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_5_og_: _dafny.Seq
            d_6_oi_: bool
            d_7_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_5_og_ = out1_
            d_6_oi_ = out2_
            d_7_oc_ = out3_
            generated = d_5_og_
            insideConstrainedOut = d_6_oi_
            currentConstrainedOut = d_7_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        d_8_constrainedHardLimit_: int
        if ((d_2_steps_) + (100)) <= (((maxSteps) - (200) if (maxSteps) >= (200) else 0)):
            d_8_constrainedHardLimit_ = (d_2_steps_) + (100)
        elif True:
            if (maxSteps) >= (200):
                d_8_constrainedHardLimit_ = (maxSteps) - (200)
            elif True:
                d_8_constrainedHardLimit_ = maxSteps
        d_9_spCount_: int
        d_9_spCount_ = 0
        with _dafny.label("1"):
            while ((d_2_steps_) < (d_8_constrainedHardLimit_)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    if (d_9_spCount_) >= (3):
                        d_10_cg_: _dafny.Seq
                        d_11_ci_: bool
                        d_12_cc_: _dafny.Seq
                        d_13_closed_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_10_cg_ = out4_
                        d_11_ci_ = out5_
                        d_12_cc_ = out6_
                        d_13_closed_ = out7_
                        d_2_steps_ = (d_2_steps_) + (1)
                        generated = d_10_cg_
                        insideConstrainedOut = d_11_ci_
                        currentConstrainedOut = d_12_cc_
                        if d_13_closed_:
                            raise _dafny.Break("1")
                        elif True:
                            if (insideConstrainedOut) and ((d_2_steps_) < (d_8_constrainedHardLimit_)):
                                d_14_constrainedPrompt_: _dafny.Seq
                                d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_15_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_15_next_ = out8_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_15_next_) == (eosToken):
                                    raise _dafny.Break("1")
                                elif True:
                                    d_16_ag_: _dafny.Seq
                                    d_17_ai_: bool
                                    d_18_ac_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                    d_16_ag_ = out9_
                                    d_17_ai_ = out10_
                                    d_18_ac_ = out11_
                                    generated = d_16_ag_
                                    insideConstrainedOut = d_17_ai_
                                    currentConstrainedOut = d_18_ac_
                                    d_9_spCount_ = (d_9_spCount_) + (1)
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_next_: _dafny.Seq
                        out12_: _dafny.Seq
                        out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_20_next_ = out12_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_21_ag_: _dafny.Seq
                            d_22_ai_: bool
                            d_23_ac_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_21_ag_ = out13_
                            d_22_ai_ = out14_
                            d_23_ac_ = out15_
                            generated = d_21_ag_
                            insideConstrainedOut = d_22_ai_
                            currentConstrainedOut = d_23_ac_
                            d_9_spCount_ = (d_9_spCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_24_rg_: _dafny.Seq
            d_25_rc_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: _dafny.Seq
            out16_, out17_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_24_rg_ = out16_
            d_25_rc_ = out17_
            generated = d_24_rg_
            currentConstrainedOut = d_25_rc_
            if (d_2_steps_) < (maxSteps):
                d_26_cg_: _dafny.Seq
                d_27_ci_: bool
                d_28_cc_: _dafny.Seq
                d_29_closed_: bool
                out18_: _dafny.Seq
                out19_: bool
                out20_: _dafny.Seq
                out21_: bool
                out18_, out19_, out20_, out21_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                d_26_cg_ = out18_
                d_27_ci_ = out19_
                d_28_cc_ = out20_
                d_29_closed_ = out21_
                d_2_steps_ = (d_2_steps_) + (1)
                generated = d_26_cg_
                insideConstrainedOut = d_27_ci_
                currentConstrainedOut = d_28_cc_
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_30_closeBudget_: int
            d_30_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_31_cg_: _dafny.Seq
            d_32_ci_: bool
            d_33_cc_: _dafny.Seq
            out22_: _dafny.Seq
            out23_: bool
            out24_: _dafny.Seq
            out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_30_closeBudget_)
            d_31_cg_ = out22_
            d_32_ci_ = out23_
            d_33_cc_ = out24_
            generated = d_31_cg_
            insideConstrainedOut = d_32_ci_
            currentConstrainedOut = d_33_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

