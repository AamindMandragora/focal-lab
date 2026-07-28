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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write each arithmetic step and the final answer inside << >> delimiters like <<3*4=12>>. Keep each expression simple: one operation equals one number. Always close << with >>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_closeReserve_: int
        d_3_closeReserve_ = 12
        d_4_needsForceClose_: bool
        d_4_needsForceClose_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_5_remaining_: int
                    d_5_remaining_ = (maxSteps) - (d_2_steps_)
                    if not(insideConstrainedOut):
                        d_6_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_6_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_4_needsForceClose_ = False
                    elif True:
                        if (d_5_remaining_) <= (d_3_closeReserve_):
                            d_7_rg_: _dafny.Seq
                            d_8_rc_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: _dafny.Seq
                            out1_, out2_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_7_rg_ = out1_
                            d_8_rc_ = out2_
                            generated = d_7_rg_
                            currentConstrainedOut = d_8_rc_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_9_cg_: _dafny.Seq
                                d_10_ci_: bool
                                d_11_cc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_9_cg_ = out3_
                                d_10_ci_ = out4_
                                d_11_cc_ = out5_
                                d_2_steps_ = (d_2_steps_) + (1)
                                generated = d_9_cg_
                                insideConstrainedOut = d_10_ci_
                                currentConstrainedOut = d_11_cc_
                            elif True:
                                d_2_steps_ = (d_2_steps_) + (1)
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                        elif (parser).IsCompletePrefix(currentConstrainedOut):
                            d_12_cg_: _dafny.Seq
                            d_13_ci_: bool
                            d_14_cc_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_12_cg_ = out6_
                            d_13_ci_ = out7_
                            d_14_cc_ = out8_
                            d_2_steps_ = (d_2_steps_) + (1)
                            generated = d_12_cg_
                            insideConstrainedOut = d_13_ci_
                            currentConstrainedOut = d_14_cc_
                        elif True:
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_16_next_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                            d_16_next_ = out9_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                d_17_rg_: _dafny.Seq
                                d_18_rc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: _dafny.Seq
                                out10_, out11_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_17_rg_ = out10_
                                d_18_rc_ = out11_
                                generated = d_17_rg_
                                currentConstrainedOut = d_18_rc_
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    raise _dafny.Break("0")
                            elif True:
                                d_19_ag_: _dafny.Seq
                                d_20_ai_: bool
                                d_21_ac_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                d_19_ag_ = out12_
                                d_20_ai_ = out13_
                                d_21_ac_ = out14_
                                generated = d_19_ag_
                                insideConstrainedOut = d_20_ai_
                                currentConstrainedOut = d_21_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

