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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap each intermediate calculation and the final answer in << >> delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_maxSpanSteps_: int
        d_3_maxSpanSteps_ = 15
        d_4_chunkSize_: int
        d_4_chunkSize_ = 20
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remaining_: int
                        d_5_remaining_ = (maxSteps) - (d_1_steps_)
                        d_6_budget_: int
                        if (d_5_remaining_) < (d_4_chunkSize_):
                            d_6_budget_ = d_5_remaining_
                        elif True:
                            d_6_budget_ = d_4_chunkSize_
                        if (d_6_budget_) == (0):
                            raise _dafny.Break("0")
                        d_7_chunkGenerated_: _dafny.Seq
                        d_8_stoppedOnOpenSpan_: bool
                        d_9_stoppedOnEos_: bool
                        d_10_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_budget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_7_chunkGenerated_ = out0_
                        d_8_stoppedOnOpenSpan_ = out1_
                        d_9_stoppedOnEos_ = out2_
                        d_10_stepsUsed_ = out3_
                        d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                        generated = d_7_chunkGenerated_
                        if d_8_stoppedOnOpenSpan_:
                            if ((d_1_steps_) + (2)) <= (maxSteps):
                                d_11_eg_: _dafny.Seq
                                d_12_ei_: bool
                                d_13_ec_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_11_eg_ = out4_
                                d_12_ei_ = out5_
                                d_13_ec_ = out6_
                                generated = d_11_eg_
                                insideConstrainedOut = d_12_ei_
                                currentConstrainedOut = d_13_ec_
                                d_2_spanSteps_ = 0
                        elif d_9_stoppedOnEos_:
                            raise _dafny.Break("0")
                    elif True:
                        if ((d_2_spanSteps_) >= (d_3_maxSpanSteps_)) or (((d_1_steps_) + (1)) >= (maxSteps)):
                            d_14_rg_: _dafny.Seq
                            d_15_rc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: _dafny.Seq
                            out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_14_rg_ = out7_
                            d_15_rc_ = out8_
                            generated = d_14_rg_
                            currentConstrainedOut = d_15_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_16_closedG_: _dafny.Seq
                                d_17_closedI_: bool
                                d_18_closedC_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_16_closedG_ = out9_
                                d_17_closedI_ = out10_
                                d_18_closedC_ = out11_
                                generated = d_16_closedG_
                                insideConstrainedOut = d_17_closedI_
                                currentConstrainedOut = d_18_closedC_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = 0
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanSteps_ = 0
                            raise _dafny.Break("0")
                        elif True:
                            d_19_cg_: _dafny.Seq
                            d_20_ci_: bool
                            d_21_cc_: _dafny.Seq
                            d_22_closed_: bool
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out15_: bool
                            out12_, out13_, out14_, out15_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_19_cg_ = out12_
                            d_20_ci_ = out13_
                            d_21_cc_ = out14_
                            d_22_closed_ = out15_
                            if d_22_closed_:
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_19_cg_
                                insideConstrainedOut = d_20_ci_
                                currentConstrainedOut = d_21_cc_
                                d_2_spanSteps_ = 0
                            elif True:
                                d_23_constrainedPrompt_: _dafny.Seq
                                d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_24_next_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_24_next_ = out16_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                                if (d_24_next_) == (eosToken):
                                    d_25_rg2_: _dafny.Seq
                                    d_26_rc2_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_25_rg2_ = out17_
                                    d_26_rc2_ = out18_
                                    generated = d_25_rg2_
                                    currentConstrainedOut = d_26_rc2_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_27_closedG2_: _dafny.Seq
                                        d_28_closedI2_: bool
                                        d_29_closedC2_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_27_closedG2_ = out19_
                                        d_28_closedI2_ = out20_
                                        d_29_closedC2_ = out21_
                                        generated = d_27_closedG2_
                                        insideConstrainedOut = d_28_closedI2_
                                        currentConstrainedOut = d_29_closedC2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    raise _dafny.Break("0")
                                elif True:
                                    d_30_ag_: _dafny.Seq
                                    d_31_ai_: bool
                                    d_32_ac_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out23_: bool
                                    out24_: _dafny.Seq
                                    out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                    d_30_ag_ = out22_
                                    d_31_ai_ = out23_
                                    d_32_ac_ = out24_
                                    generated = d_30_ag_
                                    insideConstrainedOut = d_31_ai_
                                    currentConstrainedOut = d_32_ac_
                    pass
            pass
        if insideConstrainedOut:
            d_33_rg_: _dafny.Seq
            d_34_rc_: _dafny.Seq
            out25_: _dafny.Seq
            out26_: _dafny.Seq
            out25_, out26_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_33_rg_ = out25_
            d_34_rc_ = out26_
            generated = d_33_rg_
            currentConstrainedOut = d_34_rc_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_35_closedG_: _dafny.Seq
                d_36_closedI_: bool
                d_37_closedC_: _dafny.Seq
                out27_: _dafny.Seq
                out28_: bool
                out29_: _dafny.Seq
                out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_35_closedG_ = out27_
                d_36_closedI_ = out28_
                d_37_closedC_ = out29_
                generated = d_35_closedG_
                insideConstrainedOut = d_36_closedI_
                currentConstrainedOut = d_37_closedC_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

