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
        d_2_constrainedSteps_: int
        d_2_constrainedSteps_ = 0
        d_3_maxConstrainedSteps_: int
        d_3_maxConstrainedSteps_ = 45
        d_4_spanOpened_: bool
        d_4_spanOpened_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remainingSteps_: int
                        d_5_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (not(d_4_spanOpened_)) and ((d_5_remainingSteps_) <= ((d_3_maxConstrainedSteps_) + (2))):
                            d_6_og_: _dafny.Seq
                            d_7_oi_: bool
                            d_8_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_og_ = out0_
                            d_7_oi_ = out1_
                            d_8_oc_ = out2_
                            generated = d_6_og_
                            insideConstrainedOut = d_7_oi_
                            currentConstrainedOut = d_8_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_4_spanOpened_ = True
                            d_2_constrainedSteps_ = 0
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_10_og_: _dafny.Seq
                                    d_11_oi_: bool
                                    d_12_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_10_og_ = out4_
                                    d_11_oi_ = out5_
                                    d_12_oc_ = out6_
                                    generated = d_10_og_
                                    insideConstrainedOut = d_11_oi_
                                    currentConstrainedOut = d_12_oc_
                                    d_4_spanOpened_ = True
                                    d_2_constrainedSteps_ = 0
                    elif True:
                        d_13_remainingSteps_: int
                        d_13_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if ((d_2_constrainedSteps_) >= (d_3_maxConstrainedSteps_)) or ((d_13_remainingSteps_) <= (2)):
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
                            d_2_constrainedSteps_ = 0
                            raise _dafny.Break("0")
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
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_22_closed_:
                            generated = d_19_cg_
                            insideConstrainedOut = d_20_ci_
                            currentConstrainedOut = d_21_cc_
                            d_2_constrainedSteps_ = 0
                        elif True:
                            d_23_constrainedPrompt_: _dafny.Seq
                            d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_24_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_24_next_ = out16_
                            d_2_constrainedSteps_ = (d_2_constrainedSteps_) + (1)
                            if (d_24_next_) == (eosToken):
                                d_25_rg_: _dafny.Seq
                                d_26_rc_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: _dafny.Seq
                                out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_25_rg_ = out17_
                                d_26_rc_ = out18_
                                generated = d_25_rg_
                                currentConstrainedOut = d_26_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_27_closedG_: _dafny.Seq
                                    d_28_closedI_: bool
                                    d_29_closedC_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_27_closedG_ = out19_
                                    d_28_closedI_ = out20_
                                    d_29_closedC_ = out21_
                                    generated = d_27_closedG_
                                    insideConstrainedOut = d_28_closedI_
                                    currentConstrainedOut = d_29_closedC_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_constrainedSteps_ = 0
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
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

