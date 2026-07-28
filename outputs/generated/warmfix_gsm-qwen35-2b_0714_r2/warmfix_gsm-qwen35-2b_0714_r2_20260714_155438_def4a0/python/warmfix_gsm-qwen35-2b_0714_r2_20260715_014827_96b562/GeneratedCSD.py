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
        if (maxSteps) == (0):
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. At the end, write the complete arithmetic formula as: The answer is <<EXPR>> where EXPR uses numbers and operators +, -, *, / with parentheses. Write the full formula like (n1 * price + n2 * price2), not just a single variable."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_stepsAvail_: int
        d_2_stepsAvail_ = maxSteps
        if (d_2_stepsAvail_) > (0):
            d_3_mg_: _dafny.Seq
            d_4_mi_: bool
            d_5_mc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).GenerateWithManagedSpan(lm, parser, prompt, generated, insideConstrainedOut, currentConstrainedOut, d_2_stepsAvail_, validTokenGroups, _dafny.BigRational('1e0'), 3, eosToken)
            d_3_mg_ = out0_
            d_4_mi_ = out1_
            d_5_mc_ = out2_
            generated = d_3_mg_
            insideConstrainedOut = d_4_mi_
            currentConstrainedOut = d_5_mc_
            cost = d_2_stepsAvail_
        if (insideConstrainedOut) and ((cost) < (maxSteps)):
            d_6_finalBudget_: int
            d_6_finalBudget_ = (maxSteps) - (cost)
            d_7_wg_: _dafny.Seq
            d_8_wi_: bool
            d_9_wc_: _dafny.Seq
            out3_: _dafny.Seq
            out4_: bool
            out5_: _dafny.Seq
            out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_6_finalBudget_)
            d_7_wg_ = out3_
            d_8_wi_ = out4_
            d_9_wc_ = out5_
            generated = d_7_wg_
            insideConstrainedOut = d_8_wi_
            currentConstrainedOut = d_9_wc_
            cost = (cost) + (d_6_finalBudget_)
        return generated, insideConstrainedOut, currentConstrainedOut, cost

