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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write variable names without curly braces (write total not {total}). At the very end write the answer as <<expression>> using variable names, numbers, and operators +, -, *, /, //, %, int(). Example: <<n1 + n2>>, <<int(a * b / c)>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_remainingBudget_: int
                        d_2_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_2_remainingBudget_) <= (3):
                            raise _dafny.Break("0")
                        elif (d_2_remainingBudget_) <= (80):
                            d_3_genStr_: _dafny.Seq
                            d_3_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
                            d_4_openCount_: int
                            d_4_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_3_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                            d_5_closeCount_: int
                            d_5_closeCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_3_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                            if ((d_4_openCount_) > (0)) and ((d_5_closeCount_) >= (d_4_openCount_)):
                                raise _dafny.Break("0")
                            elif True:
                                d_6_og_: _dafny.Seq
                                d_7_oi_: bool
                                d_8_oc_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_6_og_ = out0_
                                d_7_oi_ = out1_
                                d_8_oc_ = out2_
                                generated = d_6_og_
                                insideConstrainedOut = d_7_oi_
                                currentConstrainedOut = d_8_oc_
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_10_og2_: _dafny.Seq
                                    d_11_oi2_: bool
                                    d_12_oc2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_10_og2_ = out4_
                                    d_11_oi2_ = out5_
                                    d_12_oc2_ = out6_
                                    generated = d_10_og2_
                                    insideConstrainedOut = d_11_oi2_
                                    currentConstrainedOut = d_12_oc2_
                    elif True:
                        d_13_remaining_: int
                        d_13_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_13_remaining_) <= (1):
                            d_14_cg0_: _dafny.Seq
                            d_15_ci0_: bool
                            d_16_cc0_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_13_remaining_)
                            d_14_cg0_ = out7_
                            d_15_ci0_ = out8_
                            d_16_cc0_ = out9_
                            generated = d_14_cg0_
                            insideConstrainedOut = d_15_ci0_
                            currentConstrainedOut = d_16_cc0_
                            d_1_steps_ = maxSteps
                            raise _dafny.Break("0")
                        elif True:
                            d_17_cg1_: _dafny.Seq
                            d_18_ci1_: bool
                            d_19_cc1_: _dafny.Seq
                            d_20_closed1_: bool
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out13_: bool
                            out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_17_cg1_ = out10_
                            d_18_ci1_ = out11_
                            d_19_cc1_ = out12_
                            d_20_closed1_ = out13_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_20_closed1_:
                                generated = d_17_cg1_
                                insideConstrainedOut = d_18_ci1_
                                currentConstrainedOut = d_19_cc1_
                            elif True:
                                d_21_constrainedPrompt_: _dafny.Seq
                                d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_22_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_22_next_ = out14_
                                if (d_22_next_) == (eosToken):
                                    d_23_remaining2_: int
                                    d_23_remaining2_ = (maxSteps) - (d_1_steps_)
                                    d_24_cg2_: _dafny.Seq
                                    d_25_ci2_: bool
                                    d_26_cc2_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_remaining2_)
                                    d_24_cg2_ = out15_
                                    d_25_ci2_ = out16_
                                    d_26_cc2_ = out17_
                                    generated = d_24_cg2_
                                    insideConstrainedOut = d_25_ci2_
                                    currentConstrainedOut = d_26_cc2_
                                    d_1_steps_ = maxSteps
                                    raise _dafny.Break("0")
                                elif True:
                                    d_27_ag_: _dafny.Seq
                                    d_28_ai_: bool
                                    d_29_ac_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                    d_27_ag_ = out18_
                                    d_28_ai_ = out19_
                                    d_29_ac_ = out20_
                                    generated = d_27_ag_
                                    insideConstrainedOut = d_28_ai_
                                    currentConstrainedOut = d_29_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_30_remainingA_: int
            d_30_remainingA_ = (maxSteps) - (d_1_steps_)
            d_31_cgA_: _dafny.Seq
            d_32_ciA_: bool
            d_33_ccA_: _dafny.Seq
            out21_: _dafny.Seq
            out22_: bool
            out23_: _dafny.Seq
            out21_, out22_, out23_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_30_remainingA_)
            d_31_cgA_ = out21_
            d_32_ciA_ = out22_
            d_33_ccA_ = out23_
            generated = d_31_cgA_
            insideConstrainedOut = d_32_ciA_
            currentConstrainedOut = d_33_ccA_
            d_1_steps_ = maxSteps
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_34_genStr2_: _dafny.Seq
            d_34_genStr2_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
            d_35_openCount2_: int
            d_35_openCount2_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_34_genStr2_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            if (d_35_openCount2_) == (0):
                d_36_remainingB_: int
                d_36_remainingB_ = (maxSteps) - (d_1_steps_)
                if (d_36_remainingB_) >= (5):
                    d_37_ogB_: _dafny.Seq
                    d_38_oiB_: bool
                    d_39_ocB_: _dafny.Seq
                    out24_: _dafny.Seq
                    out25_: bool
                    out26_: _dafny.Seq
                    out24_, out25_, out26_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_37_ogB_ = out24_
                    d_38_oiB_ = out25_
                    d_39_ocB_ = out26_
                    generated = d_37_ogB_
                    insideConstrainedOut = d_38_oiB_
                    currentConstrainedOut = d_39_ocB_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_1_steps_) < (maxSteps):
                        d_40_remainingB2_: int
                        d_40_remainingB2_ = (maxSteps) - (d_1_steps_)
                        d_41_cgB_: _dafny.Seq
                        d_42_ciB_: bool
                        d_43_ccB_: _dafny.Seq
                        out27_: _dafny.Seq
                        out28_: bool
                        out29_: _dafny.Seq
                        out27_, out28_, out29_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_40_remainingB2_)
                        d_41_cgB_ = out27_
                        d_42_ciB_ = out28_
                        d_43_ccB_ = out29_
                        generated = d_41_cgB_
                        insideConstrainedOut = d_42_ciB_
                        currentConstrainedOut = d_43_ccB_
                        d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

