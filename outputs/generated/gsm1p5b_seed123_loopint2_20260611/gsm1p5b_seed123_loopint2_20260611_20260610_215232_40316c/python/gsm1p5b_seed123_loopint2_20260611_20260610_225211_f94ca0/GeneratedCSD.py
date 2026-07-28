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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. Show your work. At the very end, write only the final numeric answer as an arithmetic expression between << and >>. Use only actual numbers and +, -, *, /, (, ) inside << >>. Example: <<42>> or <<5 * 3 + 2>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_forceThreshold_: int
        d_2_forceThreshold_ = 60
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (((maxSteps) - (d_1_steps_)) <= (d_2_forceThreshold_)) and ((d_2_forceThreshold_) <= (((maxSteps) - (d_1_steps_)) + (d_2_forceThreshold_))):
                            if (d_1_steps_) < (maxSteps):
                                d_3_og_: _dafny.Seq
                                d_4_oi_: bool
                                d_5_oc_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_3_og_ = out0_
                                d_4_oi_ = out1_
                                d_5_oc_ = out2_
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_3_og_
                                insideConstrainedOut = d_4_oi_
                                currentConstrainedOut = d_5_oc_
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_6_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next_) == (eosToken):
                                if ((d_1_steps_) + (5)) <= (maxSteps):
                                    d_7_og_: _dafny.Seq
                                    d_8_oi_: bool
                                    d_9_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_7_og_ = out4_
                                    d_8_oi_ = out5_
                                    d_9_oc_ = out6_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    generated = d_7_og_
                                    insideConstrainedOut = d_8_oi_
                                    currentConstrainedOut = d_9_oc_
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                    elif True:
                        d_10_cg_: _dafny.Seq
                        d_11_ci_: bool
                        d_12_cc_: _dafny.Seq
                        d_13_closed_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_10_cg_ = out7_
                        d_11_ci_ = out8_
                        d_12_cc_ = out9_
                        d_13_closed_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_13_closed_:
                            generated = d_10_cg_
                            insideConstrainedOut = d_11_ci_
                            currentConstrainedOut = d_12_cc_
                            if (d_1_steps_) < (maxSteps):
                                d_14_next2_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_14_next2_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next2_]))
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            if (d_1_steps_) >= (maxSteps):
                                d_15_rg_: _dafny.Seq
                                d_16_rc_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: _dafny.Seq
                                out12_, out13_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_15_rg_ = out12_
                                d_16_rc_ = out13_
                                generated = d_15_rg_
                                currentConstrainedOut = d_16_rc_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            d_17_constrainedPrompt_: _dafny.Seq
                            d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_18_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_18_next_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_next_) == (eosToken):
                                d_19_rg_: _dafny.Seq
                                d_20_rc_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: _dafny.Seq
                                out15_, out16_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_19_rg_ = out15_
                                d_20_rc_ = out16_
                                generated = d_19_rg_
                                currentConstrainedOut = d_20_rc_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_21_ag_: _dafny.Seq
                                d_22_ai_: bool
                                d_23_ac_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                d_21_ag_ = out17_
                                d_22_ai_ = out18_
                                d_23_ac_ = out19_
                                generated = d_21_ag_
                                insideConstrainedOut = d_22_ai_
                                currentConstrainedOut = d_23_ac_
                    pass
            pass
        if insideConstrainedOut:
            d_24_rg_: _dafny.Seq
            d_25_rc_: _dafny.Seq
            out20_: _dafny.Seq
            out21_: _dafny.Seq
            out20_, out21_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_24_rg_ = out20_
            d_25_rc_ = out21_
            generated = d_24_rg_
            currentConstrainedOut = d_25_rc_
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

