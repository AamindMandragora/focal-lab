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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap each intermediate symbolic expression and the final numeric answer inside << >> delimiters. For example: The total is <<3 + 4 = 7>> apples. Final answer: <<7>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                generated = out1_
                                insideConstrainedOut = out2_
                                currentConstrainedOut = out3_
                    elif True:
                        d_3_cg_: _dafny.Seq
                        d_4_ci_: bool
                        d_5_cc_: _dafny.Seq
                        d_6_closed_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_3_cg_ = out4_
                        d_4_ci_ = out5_
                        d_5_cc_ = out6_
                        d_6_closed_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_6_closed_:
                            generated = d_3_cg_
                            insideConstrainedOut = d_4_ci_
                            currentConstrainedOut = d_5_cc_
                        elif True:
                            d_7_constrainedPrompt_: _dafny.Seq
                            d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_8_next_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_8_next_ = out8_
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_9_ag_: _dafny.Seq
                                d_10_ai_: bool
                                d_11_ac_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                                d_9_ag_ = out9_
                                d_10_ai_ = out10_
                                d_11_ac_ = out11_
                                generated = d_9_ag_
                                insideConstrainedOut = d_10_ai_
                                currentConstrainedOut = d_11_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_12_closeBudget_: int
            d_12_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_13_cg_: _dafny.Seq
            d_14_ci_: bool
            d_15_cc_: _dafny.Seq
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_closeBudget_)
            d_13_cg_ = out12_
            d_14_ci_ = out13_
            d_15_cc_ = out14_
            generated = d_13_cg_
            insideConstrainedOut = d_14_ci_
            currentConstrainedOut = d_15_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

