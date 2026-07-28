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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. After your reasoning, write ONLY the final arithmetic expression inside << >> delimiters. The expression must use only numbers and operators +,-,*,/,(,). Do not write anything inside << >> except the arithmetic expression.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_constrainedReserve_: int
        d_2_constrainedReserve_ = 62
        d_3_reasoningBudget_: int
        if (maxSteps) > ((d_2_constrainedReserve_) + (5)):
            d_3_reasoningBudget_ = (maxSteps) - (d_2_constrainedReserve_)
        elif True:
            if (maxSteps) > (3):
                d_3_reasoningBudget_ = (maxSteps) - (3)
            elif True:
                d_3_reasoningBudget_ = 0
        d_4_forcedOpen_: bool
        d_4_forcedOpen_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_1_steps_) >= (d_3_reasoningBudget_):
                            if ((d_1_steps_) + (1)) < (maxSteps):
                                d_5_og_: _dafny.Seq
                                d_6_oi_: bool
                                d_7_oc_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_5_og_ = out0_
                                d_6_oi_ = out1_
                                d_7_oc_ = out2_
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_5_og_
                                insideConstrainedOut = d_6_oi_
                                currentConstrainedOut = d_7_oc_
                                d_4_forcedOpen_ = True
                            elif True:
                                d_8_next_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_8_next_ = out3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_8_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                    raise _dafny.Break("0")
                        elif True:
                            d_9_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                d_3_reasoningBudget_ = d_1_steps_
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                    elif True:
                        d_10_cg_: _dafny.Seq
                        d_11_ci_: bool
                        d_12_cc_: _dafny.Seq
                        d_13_closed_: bool
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out5_, out6_, out7_, out8_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_10_cg_ = out5_
                        d_11_ci_ = out6_
                        d_12_cc_ = out7_
                        d_13_closed_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_13_closed_:
                            generated = d_10_cg_
                            insideConstrainedOut = d_11_ci_
                            currentConstrainedOut = d_12_cc_
                            raise _dafny.Break("0")
                        elif True:
                            if ((d_1_steps_) + (2)) >= (maxSteps):
                                d_14_rg_: _dafny.Seq
                                d_15_rc_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: _dafny.Seq
                                out9_, out10_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_14_rg_ = out9_
                                d_15_rc_ = out10_
                                generated = d_14_rg_
                                currentConstrainedOut = d_15_rc_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            d_16_constrainedPrompt_: _dafny.Seq
                            d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_17_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_17_next_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_next_) == (eosToken):
                                d_18_rg_: _dafny.Seq
                                d_19_rc_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: _dafny.Seq
                                out12_, out13_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_18_rg_ = out12_
                                d_19_rc_ = out13_
                                generated = d_18_rg_
                                currentConstrainedOut = d_19_rc_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_20_ag_: _dafny.Seq
                                d_21_ai_: bool
                                d_22_ac_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                d_20_ag_ = out14_
                                d_21_ai_ = out15_
                                d_22_ac_ = out16_
                                generated = d_20_ag_
                                insideConstrainedOut = d_21_ai_
                                currentConstrainedOut = d_22_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

