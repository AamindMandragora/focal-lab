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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. At the very end, output the final numeric answer as a single arithmetic expression using ONLY numbers and operators (+, -, *, /, (, )). Enclose this FINAL expression between << and >>. Example: <<(5 * 3 + 2) / 4>>. No variable names, no words inside << >>. Only one final << >> at the very end.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_reserveForAnswer_: int
        d_2_reserveForAnswer_ = 30
        d_3_forcedFinalSpan_: bool
        d_3_forcedFinalSpan_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        if ((d_4_remaining_) <= (d_2_reserveForAnswer_)) and (not(d_3_forcedFinalSpan_)):
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
                            d_3_forcedFinalSpan_ = True
                        elif True:
                            d_8_chunkMax_: int
                            if (d_4_remaining_) > (d_2_reserveForAnswer_):
                                d_8_chunkMax_ = (d_4_remaining_) - (d_2_reserveForAnswer_)
                            elif True:
                                d_8_chunkMax_ = 1
                            if (d_8_chunkMax_) == (0):
                                d_8_chunkMax_ = 1
                            if (d_8_chunkMax_) > (20):
                                d_8_chunkMax_ = 20
                            d_9_generatedOut_: _dafny.Seq
                            d_10_stoppedOnOpenSpan_: bool
                            d_11_stoppedOnEos_: bool
                            d_12_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkMax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_generatedOut_ = out3_
                            d_10_stoppedOnOpenSpan_ = out4_
                            d_11_stoppedOnEos_ = out5_
                            d_12_stepsUsed_ = out6_
                            d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                            generated = d_9_generatedOut_
                            if d_11_stoppedOnEos_:
                                if (not(d_3_forcedFinalSpan_)) and ((d_1_steps_) < (maxSteps)):
                                    d_13_og_: _dafny.Seq
                                    d_14_oi_: bool
                                    d_15_oc_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_13_og_ = out7_
                                    d_14_oi_ = out8_
                                    d_15_oc_ = out9_
                                    generated = d_13_og_
                                    insideConstrainedOut = d_14_oi_
                                    currentConstrainedOut = d_15_oc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_forcedFinalSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif d_10_stoppedOnOpenSpan_:
                                d_16_og_: _dafny.Seq
                                d_17_oi_: bool
                                d_18_oc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_16_og_ = out10_
                                d_17_oi_ = out11_
                                d_18_oc_ = out12_
                                generated = d_16_og_
                                insideConstrainedOut = d_17_oi_
                                currentConstrainedOut = d_18_oc_
                    elif True:
                        d_19_cg_: _dafny.Seq
                        d_20_ci_: bool
                        d_21_cc_: _dafny.Seq
                        d_22_closed_: bool
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out16_: bool
                        out13_, out14_, out15_, out16_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_19_cg_ = out13_
                        d_20_ci_ = out14_
                        d_21_cc_ = out15_
                        d_22_closed_ = out16_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_22_closed_:
                            generated = d_19_cg_
                            insideConstrainedOut = d_20_ci_
                            currentConstrainedOut = d_21_cc_
                        elif True:
                            if (d_1_steps_) >= (maxSteps):
                                d_23_rg_: _dafny.Seq
                                d_24_rc_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: _dafny.Seq
                                out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_23_rg_ = out17_
                                d_24_rc_ = out18_
                                generated = d_23_rg_
                                currentConstrainedOut = d_24_rc_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            d_25_constrainedPrompt_: _dafny.Seq
                            d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_26_next_: _dafny.Seq
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_26_next_ = out19_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_26_next_) == (eosToken):
                                d_27_rg_: _dafny.Seq
                                d_28_rc_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: _dafny.Seq
                                out20_, out21_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_27_rg_ = out20_
                                d_28_rc_ = out21_
                                generated = d_27_rg_
                                currentConstrainedOut = d_28_rc_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_29_ag_: _dafny.Seq
                                d_30_ai_: bool
                                d_31_ac_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                d_29_ag_ = out22_
                                d_30_ai_ = out23_
                                d_31_ac_ = out24_
                                generated = d_29_ag_
                                insideConstrainedOut = d_30_ai_
                                currentConstrainedOut = d_31_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

