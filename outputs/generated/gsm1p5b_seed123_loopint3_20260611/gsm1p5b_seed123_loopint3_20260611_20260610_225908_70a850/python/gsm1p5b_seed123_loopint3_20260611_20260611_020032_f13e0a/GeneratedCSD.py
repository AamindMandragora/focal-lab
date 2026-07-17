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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. Show all reasoning in plain text. At the very end, output the final numeric answer as a single arithmetic expression with ONLY real numbers and operators (+, -, *, /). Do NOT use variable names, letters, or placeholders. Wrap ONLY the final numeric result between << and >>. Example: <<(30*7+15)/2>>. Do not use << >> anywhere else in your response.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_forceThreshold_: int
        d_2_forceThreshold_ = 80
        d_3_maxConstrainedSteps_: int
        d_3_maxConstrainedSteps_ = 75
        d_4_constrainedPhase_: bool
        d_4_constrainedPhase_ = insideConstrained
        d_5_constrainedStepCount_: int
        d_5_constrainedStepCount_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_6_remainingSteps_: int
                    d_6_remainingSteps_ = (maxSteps) - (d_1_steps_)
                    if not(insideConstrainedOut):
                        if (d_6_remainingSteps_) <= (d_2_forceThreshold_):
                            d_7_og_: _dafny.Seq
                            d_8_oi_: bool
                            d_9_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_og_ = out0_
                            d_8_oi_ = out1_
                            d_9_oc_ = out2_
                            generated = d_7_og_
                            insideConstrainedOut = d_8_oi_
                            currentConstrainedOut = d_9_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_4_constrainedPhase_ = True
                            d_5_constrainedStepCount_ = 0
                        elif True:
                            d_10_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                if (not(d_4_constrainedPhase_)) and ((d_1_steps_) < (maxSteps)):
                                    d_11_og_: _dafny.Seq
                                    d_12_oi_: bool
                                    d_13_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_11_og_ = out4_
                                    d_12_oi_ = out5_
                                    d_13_oc_ = out6_
                                    generated = d_11_og_
                                    insideConstrainedOut = d_12_oi_
                                    currentConstrainedOut = d_13_oc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_4_constrainedPhase_ = True
                                    d_5_constrainedStepCount_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                    elif True:
                        if ((d_5_constrainedStepCount_) >= (d_3_maxConstrainedSteps_)) or ((d_6_remainingSteps_) <= (2)):
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
                                d_16_cg_: _dafny.Seq
                                d_17_ci_: bool
                                d_18_cc_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_16_cg_ = out9_
                                d_17_ci_ = out10_
                                d_18_cc_ = out11_
                                generated = d_16_cg_
                                insideConstrainedOut = d_17_ci_
                                currentConstrainedOut = d_18_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_5_constrainedStepCount_ = 0
                            raise _dafny.Break("0")
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_19_cg_: _dafny.Seq
                            d_20_ci_: bool
                            d_21_cc_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_cg_ = out12_
                            d_20_ci_ = out13_
                            d_21_cc_ = out14_
                            generated = d_19_cg_
                            insideConstrainedOut = d_20_ci_
                            currentConstrainedOut = d_21_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_5_constrainedStepCount_ = 0
                        elif True:
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_23_next_: _dafny.Seq
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_23_next_ = out15_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_5_constrainedStepCount_ = (d_5_constrainedStepCount_) + (1)
                            if (d_23_next_) == (eosToken):
                                d_24_rg_: _dafny.Seq
                                d_25_rc_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: _dafny.Seq
                                out16_, out17_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_24_rg_ = out16_
                                d_25_rc_ = out17_
                                generated = d_24_rg_
                                currentConstrainedOut = d_25_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_26_cg_: _dafny.Seq
                                    d_27_ci_: bool
                                    d_28_cc_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_26_cg_ = out18_
                                    d_27_ci_ = out19_
                                    d_28_cc_ = out20_
                                    generated = d_26_cg_
                                    insideConstrainedOut = d_27_ci_
                                    currentConstrainedOut = d_28_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_5_constrainedStepCount_ = 0
                                raise _dafny.Break("0")
                            elif True:
                                d_29_ag_: _dafny.Seq
                                d_30_ai_: bool
                                d_31_ac_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_29_ag_ = out21_
                                d_30_ai_ = out22_
                                d_31_ac_ = out23_
                                generated = d_29_ag_
                                insideConstrainedOut = d_30_ai_
                                currentConstrainedOut = d_31_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

