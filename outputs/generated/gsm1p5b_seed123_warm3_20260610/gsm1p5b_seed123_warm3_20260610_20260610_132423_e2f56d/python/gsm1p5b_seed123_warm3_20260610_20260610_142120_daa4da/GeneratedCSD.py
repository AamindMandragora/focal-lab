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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. At the very end, write ONLY the final arithmetic expression inside << >> delimiters. Use only numbers, +, -, *, /, (, ) inside the delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freePhaseLimit_: int
        if (maxSteps) >= (100):
            d_2_freePhaseLimit_ = 25
        elif (maxSteps) >= (50):
            d_2_freePhaseLimit_ = 15
        elif (maxSteps) >= (10):
            d_2_freePhaseLimit_ = 5
        elif True:
            d_2_freePhaseLimit_ = 1
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_shouldForce_: bool
                        d_3_shouldForce_ = (d_1_steps_) >= (d_2_freePhaseLimit_)
                        d_4_hasEnoughBudget_: bool
                        d_4_hasEnoughBudget_ = ((maxSteps) - (d_1_steps_)) >= (2)
                        if (d_3_shouldForce_) and (d_4_hasEnoughBudget_):
                            d_5_og_: _dafny.Seq
                            d_6_oi_: bool
                            d_7_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_og_ = out0_
                            d_6_oi_ = out1_
                            d_7_oc_ = out2_
                            generated = d_5_og_
                            insideConstrainedOut = d_6_oi_
                            currentConstrainedOut = d_7_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (not(d_3_shouldForce_)) or (not(d_4_hasEnoughBudget_)):
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                if ((maxSteps) - (d_1_steps_)) >= (2):
                                    d_9_og_: _dafny.Seq
                                    d_10_oi_: bool
                                    d_11_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_9_og_ = out4_
                                    d_10_oi_ = out5_
                                    d_11_oc_ = out6_
                                    generated = d_9_og_
                                    insideConstrainedOut = d_10_oi_
                                    currentConstrainedOut = d_11_oc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                        elif True:
                            d_12_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_cg_: _dafny.Seq
                        d_14_ci_: bool
                        d_15_cc_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_cg_ = out8_
                        d_14_ci_ = out9_
                        d_15_cc_ = out10_
                        generated = d_13_cg_
                        insideConstrainedOut = d_14_ci_
                        currentConstrainedOut = d_15_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_17_next_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_17_next_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_17_next_) == (eosToken):
                            d_18_rg_: _dafny.Seq
                            d_19_rc_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: _dafny.Seq
                            out12_, out13_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_18_rg_ = out12_
                            d_19_rc_ = out13_
                            generated = d_18_rg_
                            currentConstrainedOut = d_19_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_20_cg2_: _dafny.Seq
                                d_21_ci2_: bool
                                d_22_cc2_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_20_cg2_ = out14_
                                d_21_ci2_ = out15_
                                d_22_cc2_ = out16_
                                generated = d_20_cg2_
                                insideConstrainedOut = d_21_ci2_
                                currentConstrainedOut = d_22_cc2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_23_ag_: _dafny.Seq
                            d_24_ai_: bool
                            d_25_ac_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                            d_23_ag_ = out17_
                            d_24_ai_ = out18_
                            d_25_ac_ = out19_
                            generated = d_23_ag_
                            insideConstrainedOut = d_24_ai_
                            currentConstrainedOut = d_25_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_26_cg_: _dafny.Seq
                d_27_ci_: bool
                d_28_cc_: _dafny.Seq
                out20_: _dafny.Seq
                out21_: bool
                out22_: _dafny.Seq
                out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_26_cg_ = out20_
                d_27_ci_ = out21_
                d_28_cc_ = out22_
                generated = d_26_cg_
                insideConstrainedOut = d_27_ci_
                currentConstrainedOut = d_28_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_29_rg_: _dafny.Seq
                d_30_rc_: _dafny.Seq
                out23_: _dafny.Seq
                out24_: _dafny.Seq
                out23_, out24_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_29_rg_ = out23_
                d_30_rc_ = out24_
                generated = d_29_rg_
                currentConstrainedOut = d_30_rc_
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                    d_31_cg2_: _dafny.Seq
                    d_32_ci2_: bool
                    d_33_cc2_: _dafny.Seq
                    out25_: _dafny.Seq
                    out26_: bool
                    out27_: _dafny.Seq
                    out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_31_cg2_ = out25_
                    d_32_ci2_ = out26_
                    d_33_cc2_ = out27_
                    generated = d_31_cg2_
                    insideConstrainedOut = d_32_ci2_
                    currentConstrainedOut = d_33_cc2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

