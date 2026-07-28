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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For each calculation and the final answer, wrap the expression in << >> delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_remaining_: int
                        d_2_remaining_ = (maxSteps) - (d_1_steps_)
                        d_3_budget_: int
                        if (d_2_remaining_) < (10):
                            d_3_budget_ = d_2_remaining_
                        elif True:
                            d_3_budget_ = 10
                        d_4_chunkGenerated_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_budget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkGenerated_ = out0_
                        d_5_stoppedOnOpenSpan_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        generated = d_4_chunkGenerated_
                        if d_5_stoppedOnOpenSpan_:
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
                        elif d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                    elif True:
                        d_11_cg_: _dafny.Seq
                        d_12_ci_: bool
                        d_13_cc_: _dafny.Seq
                        d_14_closed_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_11_cg_ = out7_
                        d_12_ci_ = out8_
                        d_13_cc_ = out9_
                        d_14_closed_ = out10_
                        if d_14_closed_:
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = d_11_cg_
                            insideConstrainedOut = d_12_ci_
                            currentConstrainedOut = d_13_cc_
                        elif True:
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_16_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 8, eosToken)
                            d_16_next_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                d_17_rg_: _dafny.Seq
                                d_18_rc_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: _dafny.Seq
                                out12_, out13_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_17_rg_ = out12_
                                d_18_rc_ = out13_
                                generated = d_17_rg_
                                currentConstrainedOut = d_18_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_19_closedG_: _dafny.Seq
                                    d_20_closedI_: bool
                                    d_21_closedC_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_19_closedG_ = out14_
                                    d_20_closedI_ = out15_
                                    d_21_closedC_ = out16_
                                    generated = d_19_closedG_
                                    insideConstrainedOut = d_20_closedI_
                                    currentConstrainedOut = d_21_closedC_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_22_ag_: _dafny.Seq
                                d_23_ai_: bool
                                d_24_ac_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                d_22_ag_ = out17_
                                d_23_ai_ = out18_
                                d_24_ac_ = out19_
                                generated = d_22_ag_
                                insideConstrainedOut = d_23_ai_
                                currentConstrainedOut = d_24_ac_
                    pass
            pass
        if insideConstrainedOut:
            d_25_rg_: _dafny.Seq
            d_26_rc_: _dafny.Seq
            out20_: _dafny.Seq
            out21_: _dafny.Seq
            out20_, out21_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_25_rg_ = out20_
            d_26_rc_ = out21_
            generated = d_25_rg_
            currentConstrainedOut = d_26_rc_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_27_closedG_: _dafny.Seq
                d_28_closedI_: bool
                d_29_closedC_: _dafny.Seq
                out22_: _dafny.Seq
                out23_: bool
                out24_: _dafny.Seq
                out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_27_closedG_ = out22_
                d_28_closedI_ = out23_
                d_29_closedC_ = out24_
                generated = d_27_closedG_
                insideConstrainedOut = d_28_closedI_
                currentConstrainedOut = d_29_closedC_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

