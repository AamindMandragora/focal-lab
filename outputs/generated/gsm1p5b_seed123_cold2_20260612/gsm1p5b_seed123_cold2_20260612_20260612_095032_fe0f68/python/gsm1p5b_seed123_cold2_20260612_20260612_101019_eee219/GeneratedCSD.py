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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. Show your work with intermediate calculations. Wrap each calculation and the final answer inside << >> delimiters. Use only arithmetic expressions like <<3*4+2=14>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_chunkBudget_: int
        d_3_chunkBudget_ = 80
        d_4_spanBudget_: int
        d_4_spanBudget_ = 30
        d_5_trailBudget_: int
        d_5_trailBudget_ = 20
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remaining_: int
                        d_6_remaining_ = (maxSteps) - (d_2_steps_)
                        d_7_budget_: int
                        if (d_6_remaining_) < (d_3_chunkBudget_):
                            d_7_budget_ = d_6_remaining_
                        elif True:
                            d_7_budget_ = d_3_chunkBudget_
                        if (d_7_budget_) == (0):
                            raise _dafny.Break("0")
                        d_8_genOut_: _dafny.Seq
                        d_9_stoppedOnOpen_: bool
                        d_10_stoppedOnEos_: bool
                        d_11_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_budget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_8_genOut_ = out0_
                        d_9_stoppedOnOpen_ = out1_
                        d_10_stoppedOnEos_ = out2_
                        d_11_stepsUsed_ = out3_
                        d_2_steps_ = (d_2_steps_) + (d_11_stepsUsed_)
                        generated = d_8_genOut_
                        if d_10_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_9_stoppedOnOpen_:
                            d_12_eg_: _dafny.Seq
                            d_13_ei_: bool
                            d_14_ec_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_12_eg_ = out4_
                            d_13_ei_ = out5_
                            d_14_ec_ = out6_
                            generated = d_12_eg_
                            insideConstrainedOut = d_13_ei_
                            currentConstrainedOut = d_14_ec_
                        elif True:
                            if (d_2_steps_) < (maxSteps):
                                d_15_og_: _dafny.Seq
                                d_16_oi_: bool
                                d_17_oc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_15_og_ = out7_
                                d_16_oi_ = out8_
                                d_17_oc_ = out9_
                                d_2_steps_ = (d_2_steps_) + (1)
                                generated = d_15_og_
                                insideConstrainedOut = d_16_oi_
                                currentConstrainedOut = d_17_oc_
                            elif True:
                                raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        if (d_2_steps_) < (maxSteps):
                            d_18_cg_: _dafny.Seq
                            d_19_ci_: bool
                            d_20_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_18_cg_ = out10_
                            d_19_ci_ = out11_
                            d_20_cc_ = out12_
                            d_2_steps_ = (d_2_steps_) + (1)
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                            d_3_chunkBudget_ = d_5_trailBudget_
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        if (d_2_steps_) < (maxSteps):
                            if (len(currentConstrainedOut)) >= (d_4_spanBudget_):
                                d_21_rg_: _dafny.Seq
                                d_22_rc_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: _dafny.Seq
                                out13_, out14_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_21_rg_ = out13_
                                d_22_rc_ = out14_
                                generated = d_21_rg_
                                currentConstrainedOut = d_22_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                    d_23_cg_: _dafny.Seq
                                    d_24_ci_: bool
                                    d_25_cc_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_23_cg_ = out15_
                                    d_24_ci_ = out16_
                                    d_25_cc_ = out17_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    generated = d_23_cg_
                                    insideConstrainedOut = d_24_ci_
                                    currentConstrainedOut = d_25_cc_
                                elif True:
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    raise _dafny.Break("0")
                            elif True:
                                d_26_constrainedPrompt_: _dafny.Seq
                                d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_27_next_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_27_next_ = out18_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_27_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_28_ag_: _dafny.Seq
                                    d_29_ai_: bool
                                    d_30_ac_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                    d_28_ag_ = out19_
                                    d_29_ai_ = out20_
                                    d_30_ac_ = out21_
                                    generated = d_28_ag_
                                    insideConstrainedOut = d_29_ai_
                                    currentConstrainedOut = d_30_ac_
                        elif True:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

