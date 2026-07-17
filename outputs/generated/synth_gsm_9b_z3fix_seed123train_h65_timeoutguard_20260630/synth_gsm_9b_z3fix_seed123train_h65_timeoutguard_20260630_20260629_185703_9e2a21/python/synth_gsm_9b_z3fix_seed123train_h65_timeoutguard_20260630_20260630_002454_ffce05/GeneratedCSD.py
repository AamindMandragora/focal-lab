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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. Use plain variable names (no curly braces like {x}, just write x). Use // for integer division. The FINAL answer must be written as <<expression>> at the end. Example formats: <<n1 + n2>>, <<n * (r + 1)>>, <<(a * b) // c>>, <<int(n * frac)>>. Write exactly one final answer expression inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_spanBudget_: int
        d_3_spanBudget_ = 100
        d_4_nearBudgetThreshold_: int
        d_4_nearBudgetThreshold_ = 200
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remainingBudget_: int
                        d_5_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_5_remainingBudget_) <= (2):
                            raise _dafny.Break("0")
                        d_6_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_6_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                        if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_7_remAfter_: int
                            d_7_remAfter_ = (maxSteps) - (d_1_steps_)
                            if (d_7_remAfter_) <= (d_4_nearBudgetThreshold_):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanSteps_ = 0
                    elif (d_2_spanSteps_) >= (d_3_spanBudget_):
                        d_8_remainingSteps_: int
                        d_8_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_8_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_9_closeBudget2_: int
                        if (d_8_remainingSteps_) < (40):
                            d_9_closeBudget2_ = d_8_remainingSteps_
                        elif True:
                            d_9_closeBudget2_ = 40
                        d_10_cg2_: _dafny.Seq
                        d_11_ci2_: bool
                        d_12_cc2_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_closeBudget2_)
                        d_10_cg2_ = out1_
                        d_11_ci2_ = out2_
                        d_12_cc2_ = out3_
                        generated = d_10_cg2_
                        insideConstrainedOut = d_11_ci2_
                        currentConstrainedOut = d_12_cc2_
                        d_1_steps_ = (d_1_steps_) + (d_9_closeBudget2_)
                        d_2_spanSteps_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_remainingSteps_: int
                        d_13_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_13_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_14_closeBudget_: int
                        if (d_13_remainingSteps_) < (20):
                            d_14_closeBudget_ = d_13_remainingSteps_
                        elif True:
                            d_14_closeBudget_ = 20
                        d_15_cg_: _dafny.Seq
                        d_16_ci_: bool
                        d_17_cc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
                        d_15_cg_ = out4_
                        d_16_ci_ = out5_
                        d_17_cc_ = out6_
                        generated = d_15_cg_
                        insideConstrainedOut = d_16_ci_
                        currentConstrainedOut = d_17_cc_
                        d_1_steps_ = (d_1_steps_) + (d_14_closeBudget_)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                        d_19_next_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                        if (d_19_next_) == (eosToken):
                            d_20_remainingSteps_: int
                            d_20_remainingSteps_ = (maxSteps) - (d_1_steps_)
                            if (d_20_remainingSteps_) == (0):
                                raise _dafny.Break("0")
                            d_21_closeBudget3_: int
                            if (d_20_remainingSteps_) < (30):
                                d_21_closeBudget3_ = d_20_remainingSteps_
                            elif True:
                                d_21_closeBudget3_ = 30
                            d_22_cg3_: _dafny.Seq
                            d_23_ci3_: bool
                            d_24_cc3_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_closeBudget3_)
                            d_22_cg3_ = out8_
                            d_23_ci3_ = out9_
                            d_24_cc3_ = out10_
                            generated = d_22_cg3_
                            insideConstrainedOut = d_23_ci3_
                            currentConstrainedOut = d_24_cc3_
                            d_1_steps_ = (d_1_steps_) + (d_21_closeBudget3_)
                            raise _dafny.Break("0")
                        elif True:
                            d_25_ag_: _dafny.Seq
                            d_26_ai_: bool
                            d_27_ac_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_25_ag_ = out11_
                            d_26_ai_ = out12_
                            d_27_ac_ = out13_
                            generated = d_25_ag_
                            insideConstrainedOut = d_26_ai_
                            currentConstrainedOut = d_27_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_28_remainingA_: int
            d_28_remainingA_ = (maxSteps) - (d_1_steps_)
            d_29_closeBudgetA_: int
            if (d_28_remainingA_) < (50):
                d_29_closeBudgetA_ = d_28_remainingA_
            elif True:
                d_29_closeBudgetA_ = 50
            d_30_cgA_: _dafny.Seq
            d_31_ciA_: bool
            d_32_ccA_: _dafny.Seq
            out14_: _dafny.Seq
            out15_: bool
            out16_: _dafny.Seq
            out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudgetA_)
            d_30_cgA_ = out14_
            d_31_ciA_ = out15_
            d_32_ccA_ = out16_
            generated = d_30_cgA_
            insideConstrainedOut = d_31_ciA_
            currentConstrainedOut = d_32_ccA_
            d_1_steps_ = (d_1_steps_) + (d_29_closeBudgetA_)
        d_33_openCount_: int
        out17_: int
        out17_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_33_openCount_ = out17_
        if (((d_33_openCount_) == (0)) and ((d_1_steps_) < (maxSteps))) and (not(insideConstrainedOut)):
            d_34_remainingB_: int
            d_34_remainingB_ = (maxSteps) - (d_1_steps_)
            if (d_34_remainingB_) >= (5):
                d_35_ogB_: _dafny.Seq
                d_36_oiB_: bool
                d_37_ocB_: _dafny.Seq
                out18_: _dafny.Seq
                out19_: bool
                out20_: _dafny.Seq
                out18_, out19_, out20_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_35_ogB_ = out18_
                d_36_oiB_ = out19_
                d_37_ocB_ = out20_
                generated = d_35_ogB_
                insideConstrainedOut = d_36_oiB_
                currentConstrainedOut = d_37_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                d_38_remainingB2_: int
                d_38_remainingB2_ = (maxSteps) - (d_1_steps_)
                d_39_closeBudgetB_: int
                if (d_38_remainingB2_) < (120):
                    d_39_closeBudgetB_ = d_38_remainingB2_
                elif True:
                    d_39_closeBudgetB_ = 120
                if (d_39_closeBudgetB_) > (0):
                    d_40_cgB_: _dafny.Seq
                    d_41_ciB_: bool
                    d_42_ccB_: _dafny.Seq
                    out21_: _dafny.Seq
                    out22_: bool
                    out23_: _dafny.Seq
                    out21_, out22_, out23_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_39_closeBudgetB_)
                    d_40_cgB_ = out21_
                    d_41_ciB_ = out22_
                    d_42_ccB_ = out23_
                    generated = d_40_cgB_
                    insideConstrainedOut = d_41_ciB_
                    currentConstrainedOut = d_42_ccB_
                    d_1_steps_ = (d_1_steps_) + (d_39_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

