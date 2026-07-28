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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. At the very end, output the answer as a single arithmetic expression using ONLY digits and operators (+, -, *, /, (, )) with no spaces, variable names, or text. Wrap it in << >>. Example: <<(12+3)*4>>. The final << >> must contain only digits and arithmetic operators.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_3_og_: _dafny.Seq
                                d_4_oi_: bool
                                d_5_oc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_3_og_ = out1_
                                d_4_oi_ = out2_
                                d_5_oc_ = out3_
                                generated = d_3_og_
                                insideConstrainedOut = d_4_oi_
                                currentConstrainedOut = d_5_oc_
                    elif True:
                        d_6_isComplete_: bool
                        d_6_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_isComplete_:
                            d_7_cg_: _dafny.Seq
                            d_8_ci_: bool
                            d_9_cc_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_cg_ = out4_
                            d_8_ci_ = out5_
                            d_9_cc_ = out6_
                            generated = d_7_cg_
                            insideConstrainedOut = d_8_ci_
                            currentConstrainedOut = d_9_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_10_validCount_ = out7_
                            if (d_10_validCount_) == (0):
                                d_11_rg_: _dafny.Seq
                                d_12_rc_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: _dafny.Seq
                                out8_, out9_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_11_rg_ = out8_
                                d_12_rc_ = out9_
                                generated = d_11_rg_
                                currentConstrainedOut = d_12_rc_
                                d_13_isNowComplete_: bool
                                d_13_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_13_isNowComplete_:
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
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_17_constrainedPrompt_: _dafny.Seq
                                d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_18_next_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                                d_18_next_ = out13_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_18_next_) == (eosToken):
                                    d_19_isNowComplete_: bool
                                    d_19_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if d_19_isNowComplete_:
                                        d_20_cg_: _dafny.Seq
                                        d_21_ci_: bool
                                        d_22_cc_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_20_cg_ = out14_
                                        d_21_ci_ = out15_
                                        d_22_cc_ = out16_
                                        generated = d_20_cg_
                                        insideConstrainedOut = d_21_ci_
                                        currentConstrainedOut = d_22_cc_
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                elif True:
                                    d_23_ag_: _dafny.Seq
                                    d_24_ai_: bool
                                    d_25_ac_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                    d_23_ag_ = out17_
                                    d_24_ai_ = out18_
                                    d_25_ac_ = out19_
                                    generated = d_23_ag_
                                    insideConstrainedOut = d_24_ai_
                                    currentConstrainedOut = d_25_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

