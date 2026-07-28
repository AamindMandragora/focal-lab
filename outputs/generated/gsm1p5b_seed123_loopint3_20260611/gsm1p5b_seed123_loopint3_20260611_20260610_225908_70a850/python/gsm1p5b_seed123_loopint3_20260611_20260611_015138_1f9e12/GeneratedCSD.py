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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. Show your reasoning. At the very end, write ONLY the final numeric answer as a single arithmetic expression using ONLY numbers and operators (+, -, *, /, (, )). Wrap it between << and >>. Example: <<5*3+2>>. CRITICAL: No variable names, no letters, only numbers and arithmetic operators inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanOpened_: bool
        d_2_spanOpened_ = insideConstrained
        d_3_constrainedReserve_: int
        d_3_constrainedReserve_ = 65
        d_4_maxConstrainedSteps_: int
        d_4_maxConstrainedSteps_ = 60
        d_5_constrainedStepsUsed_: int
        d_5_constrainedStepsUsed_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_6_remainingSteps_: int
                    d_6_remainingSteps_ = (maxSteps) - (d_1_steps_)
                    if not(insideConstrainedOut):
                        if (not(d_2_spanOpened_)) and ((d_6_remainingSteps_) <= (d_3_constrainedReserve_)):
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
                            d_2_spanOpened_ = True
                            d_5_constrainedStepsUsed_ = 0
                        elif (d_2_spanOpened_) and ((d_6_remainingSteps_) <= (d_3_constrainedReserve_)):
                            d_10_og_: _dafny.Seq
                            d_11_oi_: bool
                            d_12_oc_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_og_ = out3_
                            d_11_oi_ = out4_
                            d_12_oc_ = out5_
                            generated = d_10_og_
                            insideConstrainedOut = d_11_oi_
                            currentConstrainedOut = d_12_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_5_constrainedStepsUsed_ = 0
                        elif True:
                            d_13_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                                if (d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_14_og_: _dafny.Seq
                                    d_15_oi_: bool
                                    d_16_oc_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_14_og_ = out7_
                                    d_15_oi_ = out8_
                                    d_16_oc_ = out9_
                                    generated = d_14_og_
                                    insideConstrainedOut = d_15_oi_
                                    currentConstrainedOut = d_16_oc_
                                    d_2_spanOpened_ = True
                                    d_5_constrainedStepsUsed_ = 0
                    elif True:
                        if ((d_5_constrainedStepsUsed_) >= (d_4_maxConstrainedSteps_)) or ((d_6_remainingSteps_) <= (2)):
                            d_17_rg_: _dafny.Seq
                            d_18_rc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: _dafny.Seq
                            out10_, out11_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_17_rg_ = out10_
                            d_18_rc_ = out11_
                            generated = d_17_rg_
                            currentConstrainedOut = d_18_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
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
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_5_constrainedStepsUsed_ = 0
                            raise _dafny.Break("0")
                        d_22_isComplete_: bool
                        d_22_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_22_isComplete_:
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
                            generated = d_23_cg_
                            insideConstrainedOut = d_24_ci_
                            currentConstrainedOut = d_25_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_5_constrainedStepsUsed_ = 0
                        elif True:
                            d_26_constrainedPrompt_: _dafny.Seq
                            d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_27_next_: _dafny.Seq
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_27_next_ = out18_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_5_constrainedStepsUsed_ = (d_5_constrainedStepsUsed_) + (1)
                            if (d_27_next_) == (eosToken):
                                d_28_rg_: _dafny.Seq
                                d_29_rc_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: _dafny.Seq
                                out19_, out20_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_28_rg_ = out19_
                                d_29_rc_ = out20_
                                generated = d_28_rg_
                                currentConstrainedOut = d_29_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_30_cg_: _dafny.Seq
                                    d_31_ci_: bool
                                    d_32_cc_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_30_cg_ = out21_
                                    d_31_ci_ = out22_
                                    d_32_cc_ = out23_
                                    generated = d_30_cg_
                                    insideConstrainedOut = d_31_ci_
                                    currentConstrainedOut = d_32_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_5_constrainedStepsUsed_ = 0
                                raise _dafny.Break("0")
                            elif True:
                                d_33_ag_: _dafny.Seq
                                d_34_ai_: bool
                                d_35_ac_: _dafny.Seq
                                out24_: _dafny.Seq
                                out25_: bool
                                out26_: _dafny.Seq
                                out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                d_33_ag_ = out24_
                                d_34_ai_ = out25_
                                d_35_ac_ = out26_
                                generated = d_33_ag_
                                insideConstrainedOut = d_34_ai_
                                currentConstrainedOut = d_35_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

