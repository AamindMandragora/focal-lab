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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the variable names from the problem (without curly braces). At the end, write the final symbolic answer inside << >> using only letters, numbers, operators (+,-,*,/,//,%,**), and parentheses. Example: <<n*k*(12//n)>> or <<price*(1+percent/100)*7*usage+extra_price>>. Do NOT write template placeholders like {n} inside << >>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_freeStepsTarget_: int
        d_3_freeStepsTarget_ = 60
        if (maxSteps) < (120):
            d_3_freeStepsTarget_ = _dafny.euclidian_division(maxSteps, 2)
        d_4_forcedSpan_: bool
        d_4_forcedSpan_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_5_closeBudget_: int
                        d_5_closeBudget_ = (maxSteps) - (d_2_steps_)
                        d_6_cg_: _dafny.Seq
                        d_7_ci_: bool
                        d_8_cc_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_5_closeBudget_)
                        d_6_cg_ = out0_
                        d_7_ci_ = out1_
                        d_8_cc_ = out2_
                        generated = d_6_cg_
                        insideConstrainedOut = d_7_ci_
                        currentConstrainedOut = d_8_cc_
                        d_2_steps_ = maxSteps
                        raise _dafny.Break("0")
                    elif (not(d_4_forcedSpan_)) and (((d_2_steps_) >= (d_3_freeStepsTarget_)) or (((maxSteps) - (d_2_steps_)) <= (5))):
                        if ((maxSteps) - (d_2_steps_)) >= (2):
                            d_9_og_: _dafny.Seq
                            d_10_oi_: bool
                            d_11_oc_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_9_og_ = out3_
                            d_10_oi_ = out4_
                            d_11_oc_ = out5_
                            generated = d_9_og_
                            insideConstrainedOut = d_10_oi_
                            currentConstrainedOut = d_11_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_4_forcedSpan_ = True
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_12_next_: _dafny.Seq
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_12_next_ = out6_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            if (not(d_4_forcedSpan_)) and (((maxSteps) - (d_2_steps_)) >= (2)):
                                d_13_og_: _dafny.Seq
                                d_14_oi_: bool
                                d_15_oc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_13_og_ = out7_
                                d_14_oi_ = out8_
                                d_15_oc_ = out9_
                                generated = d_13_og_
                                insideConstrainedOut = d_14_oi_
                                currentConstrainedOut = d_15_oc_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_4_forcedSpan_ = True
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

