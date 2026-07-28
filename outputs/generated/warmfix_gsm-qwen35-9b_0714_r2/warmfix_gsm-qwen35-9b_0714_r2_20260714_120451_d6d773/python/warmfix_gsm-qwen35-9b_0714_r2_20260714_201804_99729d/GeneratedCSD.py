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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write your final answer as <<int(expression)>> using plain variable names (no curly braces). Example: <<int(n1 * c1 + n2 * c2)>>. Do NOT use ** or nested int() calls. Write exactly one <<int(...)>> at the very end."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_prefixBudget_: int
        if (maxSteps) > (150):
            d_3_prefixBudget_ = (maxSteps) - (120)
        elif True:
            if (maxSteps) > (30):
                d_3_prefixBudget_ = (maxSteps) - (20)
            elif True:
                d_3_prefixBudget_ = _dafny.euclidian_division(maxSteps, 2)
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_steps_) >= (d_3_prefixBudget_)) and (((maxSteps) - (d_2_steps_)) > (2)):
                            d_4_og_: _dafny.Seq
                            d_5_oi_: bool
                            d_6_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_og_ = out0_
                            d_5_oi_ = out1_
                            d_6_oc_ = out2_
                            generated = d_4_og_
                            insideConstrainedOut = d_5_oi_
                            currentConstrainedOut = d_6_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_7_closeBudget_: int
                            d_7_closeBudget_ = (maxSteps) - (d_2_steps_)
                            if (d_7_closeBudget_) > (0):
                                d_8_sg_: _dafny.Seq
                                d_9_si_: bool
                                d_10_sc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_7_closeBudget_)
                                d_8_sg_ = out3_
                                d_9_si_ = out4_
                                d_10_sc_ = out5_
                                generated = d_8_sg_
                                insideConstrainedOut = d_9_si_
                                currentConstrainedOut = d_10_sc_
                                d_2_steps_ = (d_2_steps_) + (d_7_closeBudget_)
                            raise _dafny.Break("0")
                        elif True:
                            d_11_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out6_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                d_12_eg_: _dafny.Seq
                                d_13_ei_: bool
                                d_14_ec_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_12_eg_ = out7_
                                d_13_ei_ = out8_
                                d_14_ec_ = out9_
                                generated = d_12_eg_
                                insideConstrainedOut = d_13_ei_
                                currentConstrainedOut = d_14_ec_
                                d_15_remainingBudget_: int
                                d_15_remainingBudget_ = (maxSteps) - (d_2_steps_)
                                if (d_15_remainingBudget_) > (0):
                                    d_16_sg_: _dafny.Seq
                                    d_17_si_: bool
                                    d_18_sc_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_remainingBudget_)
                                    d_16_sg_ = out10_
                                    d_17_si_ = out11_
                                    d_18_sc_ = out12_
                                    generated = d_16_sg_
                                    insideConstrainedOut = d_17_si_
                                    currentConstrainedOut = d_18_sc_
                                    d_2_steps_ = (d_2_steps_) + (d_15_remainingBudget_)
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                    elif True:
                        d_19_remainingBudget_: int
                        d_19_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        if (d_19_remainingBudget_) > (0):
                            d_20_sg_: _dafny.Seq
                            d_21_si_: bool
                            d_22_sc_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_remainingBudget_)
                            d_20_sg_ = out13_
                            d_21_si_ = out14_
                            d_22_sc_ = out15_
                            generated = d_20_sg_
                            insideConstrainedOut = d_21_si_
                            currentConstrainedOut = d_22_sc_
                            d_2_steps_ = (d_2_steps_) + (d_19_remainingBudget_)
                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

