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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. At the very end, write the final numeric answer as a single arithmetic expression using ONLY actual numbers and operators (+, -, *, /, (, )). Wrap it between << and >>. Example: <<5*3+2>>. Numbers only inside << >>, no variable names.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_reserveForForcedSpan_: int
        d_2_reserveForForcedSpan_ = 30
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remainingSteps_: int
                        d_3_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_3_remainingSteps_) <= (d_2_reserveForForcedSpan_):
                            if (d_1_steps_) < (maxSteps):
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
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_7_chunkSize_: int
                            d_7_chunkSize_ = (d_3_remainingSteps_) - (d_2_reserveForForcedSpan_)
                            if (d_7_chunkSize_) > (60):
                                d_7_chunkSize_ = 60
                            if (d_7_chunkSize_) == (0):
                                d_8_og_: _dafny.Seq
                                d_9_oi_: bool
                                d_10_oc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_8_og_ = out3_
                                d_9_oi_ = out4_
                                d_10_oc_ = out5_
                                generated = d_8_og_
                                insideConstrainedOut = d_9_oi_
                                currentConstrainedOut = d_10_oc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_11_genOut_: _dafny.Seq
                                d_12_stoppedOnOpenSpan_: bool
                                d_13_stoppedOnEos_: bool
                                d_14_stepsUsed_: int
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: bool
                                out9_: int
                                out6_, out7_, out8_, out9_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_11_genOut_ = out6_
                                d_12_stoppedOnOpenSpan_ = out7_
                                d_13_stoppedOnEos_ = out8_
                                d_14_stepsUsed_ = out9_
                                d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                                generated = d_11_genOut_
                                if d_13_stoppedOnEos_:
                                    if ((maxSteps) - (d_1_steps_)) >= (3):
                                        d_15_og_: _dafny.Seq
                                        d_16_oi_: bool
                                        d_17_oc_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                        d_15_og_ = out10_
                                        d_16_oi_ = out11_
                                        d_17_oc_ = out12_
                                        generated = d_15_og_
                                        insideConstrainedOut = d_16_oi_
                                        currentConstrainedOut = d_17_oc_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        raise _dafny.Break("0")
                                elif d_12_stoppedOnOpenSpan_:
                                    d_18_og_: _dafny.Seq
                                    d_19_oi_: bool
                                    d_20_oc_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_18_og_ = out13_
                                    d_19_oi_ = out14_
                                    d_20_oc_ = out15_
                                    generated = d_18_og_
                                    insideConstrainedOut = d_19_oi_
                                    currentConstrainedOut = d_20_oc_
                                elif True:
                                    pass
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_21_cg_: _dafny.Seq
                            d_22_ci_: bool
                            d_23_cc_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_21_cg_ = out16_
                            d_22_ci_ = out17_
                            d_23_cc_ = out18_
                            generated = d_21_cg_
                            insideConstrainedOut = d_22_ci_
                            currentConstrainedOut = d_23_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_24_constrainedPrompt_: _dafny.Seq
                            d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_25_next_: _dafny.Seq
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_25_next_ = out19_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_25_next_) == (eosToken):
                                d_26_rg_: _dafny.Seq
                                d_27_rc_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: _dafny.Seq
                                out20_, out21_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_26_rg_ = out20_
                                d_27_rc_ = out21_
                                generated = d_26_rg_
                                currentConstrainedOut = d_27_rc_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_28_ag_: _dafny.Seq
                                d_29_ai_: bool
                                d_30_ac_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                d_28_ag_ = out22_
                                d_29_ai_ = out23_
                                d_30_ac_ = out24_
                                generated = d_28_ag_
                                insideConstrainedOut = d_29_ai_
                                currentConstrainedOut = d_30_ac_
                    pass
            pass
        if insideConstrainedOut:
            d_31_rg_: _dafny.Seq
            d_32_rc_: _dafny.Seq
            out25_: _dafny.Seq
            out26_: _dafny.Seq
            out25_, out26_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_31_rg_ = out25_
            d_32_rc_ = out26_
            generated = d_31_rg_
            currentConstrainedOut = d_32_rc_
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

