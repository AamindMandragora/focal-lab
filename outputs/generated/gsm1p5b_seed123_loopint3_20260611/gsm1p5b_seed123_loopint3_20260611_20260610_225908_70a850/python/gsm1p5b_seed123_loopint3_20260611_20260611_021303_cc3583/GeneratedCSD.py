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
        d_2_closeReserve_: int
        d_2_closeReserve_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_3_remainingSteps_: int
                    d_3_remainingSteps_ = (maxSteps) - (d_1_steps_)
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_5_og_: _dafny.Seq
                                d_6_oi_: bool
                                d_7_oc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_5_og_ = out1_
                                d_6_oi_ = out2_
                                d_7_oc_ = out3_
                                generated = d_5_og_
                                insideConstrainedOut = d_6_oi_
                                currentConstrainedOut = d_7_oc_
                    elif True:
                        if (d_3_remainingSteps_) <= (d_2_closeReserve_):
                            d_8_rg_: _dafny.Seq
                            d_9_rc_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: _dafny.Seq
                            out4_, out5_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_8_rg_ = out4_
                            d_9_rc_ = out5_
                            generated = d_8_rg_
                            currentConstrainedOut = d_9_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_10_cg_: _dafny.Seq
                                d_11_ci_: bool
                                d_12_cc_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_10_cg_ = out6_
                                d_11_ci_ = out7_
                                d_12_cc_ = out8_
                                generated = d_10_cg_
                                insideConstrainedOut = d_11_ci_
                                currentConstrainedOut = d_12_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_13_cg_: _dafny.Seq
                            d_14_ci_: bool
                            d_15_cc_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_cg_ = out9_
                            d_14_ci_ = out10_
                            d_15_cc_ = out11_
                            generated = d_13_cg_
                            insideConstrainedOut = d_14_ci_
                            currentConstrainedOut = d_15_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_16_constrainedPrompt_: _dafny.Seq
                            d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_17_next_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_17_next_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_next_) == (eosToken):
                                d_18_rg_: _dafny.Seq
                                d_19_rc_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: _dafny.Seq
                                out13_, out14_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_18_rg_ = out13_
                                d_19_rc_ = out14_
                                generated = d_18_rg_
                                currentConstrainedOut = d_19_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_20_cg_: _dafny.Seq
                                    d_21_ci_: bool
                                    d_22_cc_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_20_cg_ = out15_
                                    d_21_ci_ = out16_
                                    d_22_cc_ = out17_
                                    generated = d_20_cg_
                                    insideConstrainedOut = d_21_ci_
                                    currentConstrainedOut = d_22_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_23_ag_: _dafny.Seq
                                d_24_ai_: bool
                                d_25_ac_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                d_23_ag_ = out18_
                                d_24_ai_ = out19_
                                d_25_ac_ = out20_
                                generated = d_23_ag_
                                insideConstrainedOut = d_24_ai_
                                currentConstrainedOut = d_25_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_26_rg_: _dafny.Seq
            d_27_rc_: _dafny.Seq
            out21_: _dafny.Seq
            out22_: _dafny.Seq
            out21_, out22_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_26_rg_ = out21_
            d_27_rc_ = out22_
            generated = d_26_rg_
            currentConstrainedOut = d_27_rc_
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_28_cg_: _dafny.Seq
                d_29_ci_: bool
                d_30_cc_: _dafny.Seq
                out23_: _dafny.Seq
                out24_: bool
                out25_: _dafny.Seq
                out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_28_cg_ = out23_
                d_29_ci_ = out24_
                d_30_cc_ = out25_
                generated = d_28_cg_
                insideConstrainedOut = d_29_ci_
                currentConstrainedOut = d_30_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

