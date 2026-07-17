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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For each calculation and the final answer, wrap the expression in << >> delimiters. Keep each span to one short arithmetic expression.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_maxSpanSteps_: int
        d_3_maxSpanSteps_ = 20
        d_4_chunkSize_: int
        d_4_chunkSize_ = 25
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
                            if ((d_1_steps_) + (2)) < (maxSteps):
                                d_14_og_: _dafny.Seq
                                d_15_oi_: bool
                                d_16_oc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_14_og_ = out7_
                                d_15_oi_ = out8_
                                d_16_oc_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_14_og_
                                insideConstrainedOut = d_15_oi_
                                currentConstrainedOut = d_16_oc_
                                d_2_spanSteps_ = 0
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        if ((d_2_spanSteps_) >= (d_3_maxSpanSteps_)) or (((d_1_steps_) + (1)) >= (maxSteps)):
                            d_17_rg_: _dafny.Seq
                            d_18_rc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: _dafny.Seq
                            out10_, out11_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_17_rg_ = out10_
                            d_18_rc_ = out11_
                            generated = d_17_rg_
                            currentConstrainedOut = d_18_rc_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_19_closedG_: _dafny.Seq
                                d_20_closedI_: bool
                                d_21_closedC_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_19_closedG_ = out12_
                                d_20_closedI_ = out13_
                                d_21_closedC_ = out14_
                                generated = d_19_closedG_
                                insideConstrainedOut = d_20_closedI_
                                currentConstrainedOut = d_21_closedC_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = 0
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanSteps_ = 0
                                if (d_1_steps_) < (maxSteps):
                                    d_22_next2_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out15_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                    d_22_next2_ = out15_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_22_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_22_next2_]))
                                        if (d_22_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                            d_23_eg2_: _dafny.Seq
                                            d_24_ei2_: bool
                                            d_25_ec2_: _dafny.Seq
                                            out16_: _dafny.Seq
                                            out17_: bool
                                            out18_: _dafny.Seq
                                            out16_, out17_, out18_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                            d_23_eg2_ = out16_
                                            d_24_ei2_ = out17_
                                            d_25_ec2_ = out18_
                                            generated = d_23_eg2_
                                            insideConstrainedOut = d_24_ei2_
                                            currentConstrainedOut = d_25_ec2_
                                            d_2_spanSteps_ = 0
                        elif True:
                            d_26_cg_: _dafny.Seq
                            d_27_ci_: bool
                            d_28_cc_: _dafny.Seq
                            d_29_closed_: bool
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out22_: bool
                            out19_, out20_, out21_, out22_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_26_cg_ = out19_
                            d_27_ci_ = out20_
                            d_28_cc_ = out21_
                            d_29_closed_ = out22_
                            if d_29_closed_:
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_26_cg_
                                insideConstrainedOut = d_27_ci_
                                currentConstrainedOut = d_28_cc_
                                d_2_spanSteps_ = 0
                            elif True:
                                d_30_constrainedPrompt_: _dafny.Seq
                                d_30_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_31_next_: _dafny.Seq
                                out23_: _dafny.Seq
                                out23_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_30_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_31_next_ = out23_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                                if (d_31_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_32_ag_: _dafny.Seq
                                    d_33_ai_: bool
                                    d_34_ac_: _dafny.Seq
                                    out24_: _dafny.Seq
                                    out25_: bool
                                    out26_: _dafny.Seq
                                    out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_next_)
                                    d_32_ag_ = out24_
                                    d_33_ai_ = out25_
                                    d_34_ac_ = out26_
                                    generated = d_32_ag_
                                    insideConstrainedOut = d_33_ai_
                                    currentConstrainedOut = d_34_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

