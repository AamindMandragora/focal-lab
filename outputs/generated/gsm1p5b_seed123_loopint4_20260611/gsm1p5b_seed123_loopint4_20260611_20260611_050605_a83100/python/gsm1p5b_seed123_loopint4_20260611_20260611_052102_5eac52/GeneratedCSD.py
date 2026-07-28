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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. At the very end, wrap your final numeric answer as a simple arithmetic expression using ONLY digits and operators (+,-,*,/,(,)) inside << >>. Example: <<(3+4)*2>>. No variable names inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeGenBudget_: int
        if (maxSteps) > (25):
            d_2_freeGenBudget_ = (maxSteps) - (20)
        elif True:
            if (maxSteps) > (5):
                d_2_freeGenBudget_ = (maxSteps) - (5)
            elif True:
                d_2_freeGenBudget_ = maxSteps
        d_3_constrainedAnswerOpened_: bool
        d_3_constrainedAnswerOpened_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_1_steps_) >= (d_2_freeGenBudget_)) and (not(d_3_constrainedAnswerOpened_)):
                            d_4_og_: _dafny.Seq
                            d_5_oi_: bool
                            d_6_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_og_ = out0_
                            d_5_oi_ = out1_
                            d_6_oc_ = out2_
                            generated = d_4_og_
                            insideConstrainedOut = d_5_oi_
                            currentConstrainedOut = d_6_oc_
                            d_3_constrainedAnswerOpened_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                if (not(d_3_constrainedAnswerOpened_)) and ((d_1_steps_) < (maxSteps)):
                                    d_8_og_: _dafny.Seq
                                    d_9_oi_: bool
                                    d_10_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_8_og_ = out4_
                                    d_9_oi_ = out5_
                                    d_10_oc_ = out6_
                                    generated = d_8_og_
                                    insideConstrainedOut = d_9_oi_
                                    currentConstrainedOut = d_10_oc_
                                    d_3_constrainedAnswerOpened_ = True
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_11_og_: _dafny.Seq
                                    d_12_oi_: bool
                                    d_13_oc_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_11_og_ = out7_
                                    d_12_oi_ = out8_
                                    d_13_oc_ = out9_
                                    generated = d_11_og_
                                    insideConstrainedOut = d_12_oi_
                                    currentConstrainedOut = d_13_oc_
                                    d_3_constrainedAnswerOpened_ = True
                    elif True:
                        d_14_isComplete_: bool
                        d_14_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_14_isComplete_:
                            d_15_cg_: _dafny.Seq
                            d_16_ci_: bool
                            d_17_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_15_cg_ = out10_
                            d_16_ci_ = out11_
                            d_17_cc_ = out12_
                            generated = d_15_cg_
                            insideConstrainedOut = d_16_ci_
                            currentConstrainedOut = d_17_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_18_validCount_: int
                            out13_: int
                            out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_18_validCount_ = out13_
                            if (d_18_validCount_) == (0):
                                d_19_rg_: _dafny.Seq
                                d_20_rc_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: _dafny.Seq
                                out14_, out15_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_19_rg_ = out14_
                                d_20_rc_ = out15_
                                generated = d_19_rg_
                                currentConstrainedOut = d_20_rc_
                                d_21_isNowComplete_: bool
                                d_21_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_21_isNowComplete_:
                                    d_22_cg_: _dafny.Seq
                                    d_23_ci_: bool
                                    d_24_cc_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_22_cg_ = out16_
                                    d_23_ci_ = out17_
                                    d_24_cc_ = out18_
                                    generated = d_22_cg_
                                    insideConstrainedOut = d_23_ci_
                                    currentConstrainedOut = d_24_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                            elif True:
                                d_25_constrainedPrompt_: _dafny.Seq
                                d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_26_next_: _dafny.Seq
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_26_next_ = out19_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_26_next_) == (eosToken):
                                    d_27_isNowComplete_: bool
                                    d_27_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if (d_27_isNowComplete_) and ((d_1_steps_) < (maxSteps)):
                                        d_28_cg_: _dafny.Seq
                                        d_29_ci_: bool
                                        d_30_cc_: _dafny.Seq
                                        out20_: _dafny.Seq
                                        out21_: bool
                                        out22_: _dafny.Seq
                                        out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_28_cg_ = out20_
                                        d_29_ci_ = out21_
                                        d_30_cc_ = out22_
                                        generated = d_28_cg_
                                        insideConstrainedOut = d_29_ci_
                                        currentConstrainedOut = d_30_cc_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    raise _dafny.Break("0")
                                elif True:
                                    d_31_ag_: _dafny.Seq
                                    d_32_ai_: bool
                                    d_33_ac_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                    d_31_ag_ = out23_
                                    d_32_ai_ = out24_
                                    d_33_ac_ = out25_
                                    generated = d_31_ag_
                                    insideConstrainedOut = d_32_ai_
                                    currentConstrainedOut = d_33_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

