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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query enclosed in << and >> and no explanation.")))
        if (maxSteps) == (0):
            cost = 0
        elif not(insideConstrainedOut):
            d_1_openedGenerated_: _dafny.Seq
            d_2_openedInside_: bool
            d_3_openedCurrent_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_1_openedGenerated_ = out0_
            d_2_openedInside_ = out1_
            d_3_openedCurrent_ = out2_
            generated = d_1_openedGenerated_
            insideConstrainedOut = d_2_openedInside_
            currentConstrainedOut = d_3_openedCurrent_
            cost = 1
        elif (parser).IsCompletePrefix(currentConstrainedOut):
            d_4_closedGenerated_: _dafny.Seq
            d_5_closedInside_: bool
            d_6_closedCurrent_: _dafny.Seq
            out3_: _dafny.Seq
            out4_: bool
            out5_: _dafny.Seq
            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_4_closedGenerated_ = out3_
            d_5_closedInside_ = out4_
            d_6_closedCurrent_ = out5_
            generated = d_4_closedGenerated_
            insideConstrainedOut = d_5_closedInside_
            currentConstrainedOut = d_6_closedCurrent_
            cost = 1
        elif True:
            d_7_stablePrefix_: _dafny.Seq
            d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
            d_8_constrainedPrompt_: _dafny.Seq
            d_8_constrainedPrompt_ = (prompt) + (d_7_stablePrefix_)
            d_9_next_: _dafny.Seq
            out6_: _dafny.Seq
            out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
            d_9_next_ = out6_
            cost = 1
            if (d_9_next_) != (eosToken):
                d_10_appendedGenerated_: _dafny.Seq
                d_11_appendedInside_: bool
                d_12_appendedCurrent_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                d_10_appendedGenerated_ = out7_
                d_11_appendedInside_ = out8_
                d_12_appendedCurrent_ = out9_
                generated = d_10_appendedGenerated_
                insideConstrainedOut = d_11_appendedInside_
                currentConstrainedOut = d_12_appendedCurrent_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

