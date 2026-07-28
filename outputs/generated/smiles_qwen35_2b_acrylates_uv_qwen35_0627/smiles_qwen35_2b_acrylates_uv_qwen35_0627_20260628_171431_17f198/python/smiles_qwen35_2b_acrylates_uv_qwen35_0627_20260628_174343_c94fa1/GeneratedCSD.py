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
        d_1_steps_: int
        d_1_steps_ = 0
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        d_2_genBudget_: int
        d_2_genBudget_ = _dafny.euclidian_division(maxSteps, 2)
        with _dafny.label("0"):
            while ((d_1_steps_) < (d_2_genBudget_)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_3_constrainedPrompt_: _dafny.Seq
                    d_3_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_4_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_3_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_4_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_4_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        if not((parser).IsCompletePrefix(currentConstrainedOut)):
                            d_5_valid_: bool
                            out4_: bool
                            out4_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_4_next_)
                            d_5_valid_ = out4_
                            if d_5_valid_:
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_4_next_)
                                generated = out5_
                                insideConstrainedOut = out6_
                                currentConstrainedOut = out7_
                        if (insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                            if (d_1_steps_) < (maxSteps):
                                d_6_cg_: _dafny.Seq
                                d_7_ci_: bool
                                d_8_cc_: _dafny.Seq
                                d_9_closed_: bool
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out11_: bool
                                out8_, out9_, out10_, out11_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_6_cg_ = out8_
                                d_7_ci_ = out9_
                                d_8_cc_ = out10_
                                d_9_closed_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_6_cg_
                                insideConstrainedOut = d_7_ci_
                                currentConstrainedOut = d_8_cc_
                                if d_9_closed_:
                                    raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_10_closeBudget_: int
            d_10_closeBudget_ = (maxSteps) - (d_1_steps_)
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_closeBudget_)
            generated = out12_
            insideConstrainedOut = out13_
            currentConstrainedOut = out14_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

