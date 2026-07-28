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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step in plain text. Then write ONLY the final arithmetic expression inside << >> delimiters. Use template variable names (like n, n1, frac, price, etc.). RULES: (1) Write only ONE << >> span containing the final answer. (2) Use int() to floor fractional results: <<int(n * frac)>>, <<n - int(n * frac)>>. (3) Use // for integer floor division: <<(a - b) // c>>. (4) Do NOT use {{ }} around variables. (5) Do NOT use ** for powers. (6) Use only: +, -, *, /, //, (, ), int(), and variable names. After writing your << >> answer, stop immediately."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeTarget_: int
        d_4_freeTarget_ = _dafny.euclidian_division((maxSteps) * (7), 10)
        d_5_minInterceptStep_: int
        d_5_minInterceptStep_ = 20
        d_6_enteredSpan_: bool
        d_6_enteredSpan_ = insideConstrained
        d_7_pendingOpenAngle_: bool
        d_7_pendingOpenAngle_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((not(d_6_enteredSpan_)) and ((d_2_steps_) >= (d_4_freeTarget_))) and (((maxSteps) - (d_2_steps_)) >= (3)):
                            d_8_og_: _dafny.Seq
                            d_9_oi_: bool
                            d_10_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_og_ = out0_
                            d_9_oi_ = out1_
                            d_10_oc_ = out2_
                            generated = d_8_og_
                            insideConstrainedOut = d_9_oi_
                            currentConstrainedOut = d_10_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_6_enteredSpan_ = True
                            d_7_pendingOpenAngle_ = False
                        elif True:
                            d_11_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and ((d_2_steps_) > (d_5_minInterceptStep_)):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                d_12_og_: _dafny.Seq
                                d_13_oi_: bool
                                d_14_oc_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_12_og_ = out4_
                                d_13_oi_ = out5_
                                d_14_oc_ = out6_
                                generated = d_12_og_
                                insideConstrainedOut = d_13_oi_
                                currentConstrainedOut = d_14_oc_
                                d_6_enteredSpan_ = True
                                d_7_pendingOpenAngle_ = False
                            elif (((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")))) and (d_7_pendingOpenAngle_)) and ((d_2_steps_) > (d_5_minInterceptStep_)):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                d_15_og_: _dafny.Seq
                                d_16_oi_: bool
                                d_17_oc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_15_og_ = out7_
                                d_16_oi_ = out8_
                                d_17_oc_ = out9_
                                generated = d_15_og_
                                insideConstrainedOut = d_16_oi_
                                currentConstrainedOut = d_17_oc_
                                d_6_enteredSpan_ = True
                                d_7_pendingOpenAngle_ = False
                            elif (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                d_7_pendingOpenAngle_ = True
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                d_7_pendingOpenAngle_ = False
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_18_cg_: _dafny.Seq
                            d_19_ci_: bool
                            d_20_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_18_cg_ = out10_
                            d_19_ci_ = out11_
                            d_20_cc_ = out12_
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("0")
                        elif ((maxSteps) - (d_2_steps_)) <= (5):
                            d_21_closeBudget_: int
                            d_21_closeBudget_ = (maxSteps) - (d_2_steps_)
                            d_22_cg_: _dafny.Seq
                            d_23_ci_: bool
                            d_24_cc_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_closeBudget_)
                            d_22_cg_ = out13_
                            d_23_ci_ = out14_
                            d_24_cc_ = out15_
                            generated = d_22_cg_
                            insideConstrainedOut = d_23_ci_
                            currentConstrainedOut = d_24_cc_
                            d_2_steps_ = maxSteps
                            raise _dafny.Break("0")
                        elif True:
                            d_25_constrainedPrompt_: _dafny.Seq
                            d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_26_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                            d_26_next_ = out16_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_26_next_) == (eosToken):
                                d_27_closeBudget_: int
                                d_27_closeBudget_ = (maxSteps) - (d_2_steps_)
                                if (d_27_closeBudget_) > (15):
                                    d_27_closeBudget_ = 15
                                if (d_27_closeBudget_) >= (1):
                                    d_28_cg_: _dafny.Seq
                                    d_29_ci_: bool
                                    d_30_cc_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudget_)
                                    d_28_cg_ = out17_
                                    d_29_ci_ = out18_
                                    d_30_cc_ = out19_
                                    generated = d_28_cg_
                                    insideConstrainedOut = d_29_ci_
                                    currentConstrainedOut = d_30_cc_
                                    d_2_steps_ = (d_2_steps_) + (d_27_closeBudget_)
                                raise _dafny.Break("0")
                            elif True:
                                d_31_ag_: _dafny.Seq
                                d_32_ai_: bool
                                d_33_ac_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                d_31_ag_ = out20_
                                d_32_ai_ = out21_
                                d_33_ac_ = out22_
                                generated = d_31_ag_
                                insideConstrainedOut = d_32_ai_
                                currentConstrainedOut = d_33_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

