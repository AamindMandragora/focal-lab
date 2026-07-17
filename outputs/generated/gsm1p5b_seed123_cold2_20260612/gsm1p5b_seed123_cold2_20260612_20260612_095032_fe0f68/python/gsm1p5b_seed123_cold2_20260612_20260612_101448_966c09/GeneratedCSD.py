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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For each calculation and the final answer, write it inside << >> like <<3*4=12>>. Keep expressions simple and complete."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_closeReserve_: int
        d_3_closeReserve_ = 15
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_4_remaining_: int
                    d_4_remaining_ = (maxSteps) - (d_2_steps_)
                    if not(insideConstrainedOut):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (d_4_remaining_) <= (d_3_closeReserve_):
                            d_6_rg_: _dafny.Seq
                            d_7_rc_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: _dafny.Seq
                            out1_, out2_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_6_rg_ = out1_
                            d_7_rc_ = out2_
                            generated = d_6_rg_
                            currentConstrainedOut = d_7_rc_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_8_cg_: _dafny.Seq
                                d_9_ci_: bool
                                d_10_cc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_8_cg_ = out3_
                                d_9_ci_ = out4_
                                d_10_cc_ = out5_
                                d_2_steps_ = (d_2_steps_) + (1)
                                generated = d_8_cg_
                                insideConstrainedOut = d_9_ci_
                                currentConstrainedOut = d_10_cc_
                            elif True:
                                d_2_steps_ = (d_2_steps_) + (1)
                                raise _dafny.Break("0")
                        elif (parser).IsCompletePrefix(currentConstrainedOut):
                            d_11_cg_: _dafny.Seq
                            d_12_ci_: bool
                            d_13_cc_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_cg_ = out6_
                            d_12_ci_ = out7_
                            d_13_cc_ = out8_
                            d_2_steps_ = (d_2_steps_) + (1)
                            generated = d_11_cg_
                            insideConstrainedOut = d_12_ci_
                            currentConstrainedOut = d_13_cc_
                        elif True:
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_15_next_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                            d_15_next_ = out9_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_15_next_) == (eosToken):
                                d_16_rg_: _dafny.Seq
                                d_17_rc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: _dafny.Seq
                                out10_, out11_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_16_rg_ = out10_
                                d_17_rc_ = out11_
                                generated = d_16_rg_
                                currentConstrainedOut = d_17_rc_
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    if (d_2_steps_) < (maxSteps):
                                        d_18_cg_: _dafny.Seq
                                        d_19_ci_: bool
                                        d_20_cc_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: _dafny.Seq
                                        out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_18_cg_ = out12_
                                        d_19_ci_ = out13_
                                        d_20_cc_ = out14_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                        generated = d_18_cg_
                                        insideConstrainedOut = d_19_ci_
                                        currentConstrainedOut = d_20_cc_
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_21_ag_: _dafny.Seq
                                d_22_ai_: bool
                                d_23_ac_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                d_21_ag_ = out15_
                                d_22_ai_ = out16_
                                d_23_ac_ = out17_
                                generated = d_21_ag_
                                insideConstrainedOut = d_22_ai_
                                currentConstrainedOut = d_23_ac_
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    if (d_2_steps_) < (maxSteps):
                                        d_24_cg_: _dafny.Seq
                                        d_25_ci_: bool
                                        d_26_cc_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_24_cg_ = out18_
                                        d_25_ci_ = out19_
                                        d_26_cc_ = out20_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                        generated = d_24_cg_
                                        insideConstrainedOut = d_25_ci_
                                        currentConstrainedOut = d_26_cc_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

