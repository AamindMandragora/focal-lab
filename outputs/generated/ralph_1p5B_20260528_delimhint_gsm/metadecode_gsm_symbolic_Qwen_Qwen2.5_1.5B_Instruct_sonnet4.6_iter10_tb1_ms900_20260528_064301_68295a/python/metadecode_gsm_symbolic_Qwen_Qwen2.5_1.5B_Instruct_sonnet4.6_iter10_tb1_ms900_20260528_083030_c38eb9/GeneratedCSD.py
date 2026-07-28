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
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_remaining_: int
                        d_2_remaining_ = (maxSteps) - (d_1_steps_)
                        d_3_chunkBudget_: int
                        if (d_2_remaining_) > (80):
                            d_3_chunkBudget_ = 80
                        elif True:
                            d_3_chunkBudget_ = d_2_remaining_
                        d_4_chunkGenerated_: _dafny.Seq
                        d_5_stoppedOnOpen_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkGenerated_ = out0_
                        d_5_stoppedOnOpen_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        generated = d_4_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOnOpen_:
                            d_8_eg_: _dafny.Seq
                            d_9_ei_: bool
                            d_10_ec_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_8_eg_ = out4_
                            d_9_ei_ = out5_
                            d_10_ec_ = out6_
                            generated = d_8_eg_
                            insideConstrainedOut = d_9_ei_
                            currentConstrainedOut = d_10_ec_
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_11_og_: _dafny.Seq
                                d_12_oi_: bool
                                d_13_oc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_11_og_ = out7_
                                d_12_oi_ = out8_
                                d_13_oc_ = out9_
                                generated = d_11_og_
                                insideConstrainedOut = d_12_oi_
                                currentConstrainedOut = d_13_oc_
                                d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
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
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_17_isDead_: bool
                        out13_: bool
                        out13_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_17_isDead_ = out13_
                        if d_17_isDead_:
                            if (len(currentConstrainedOut)) == (0):
                                raise _dafny.Break("0")
                            d_18_rg_: _dafny.Seq
                            d_19_rc_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out14_, out15_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_18_rg_ = out14_
                            d_19_rc_ = out15_
                            generated = d_18_rg_
                            currentConstrainedOut = d_19_rc_
                        elif True:
                            d_20_constrainedPrompt_: _dafny.Seq
                            d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_21_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_21_next_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_21_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_22_ag_: _dafny.Seq
                                d_23_ai_: bool
                                d_24_ac_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                d_22_ag_ = out17_
                                d_23_ai_ = out18_
                                d_24_ac_ = out19_
                                generated = d_22_ag_
                                insideConstrainedOut = d_23_ai_
                                currentConstrainedOut = d_24_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

