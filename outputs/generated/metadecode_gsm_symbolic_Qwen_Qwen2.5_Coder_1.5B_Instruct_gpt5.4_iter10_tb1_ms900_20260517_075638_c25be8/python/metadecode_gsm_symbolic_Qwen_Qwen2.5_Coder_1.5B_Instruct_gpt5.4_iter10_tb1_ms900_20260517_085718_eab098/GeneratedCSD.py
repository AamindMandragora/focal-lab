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
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
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
                d_9_nextIn_: _dafny.Seq
                d_10_wasConstrained_: bool
                out6_: _dafny.Seq
                out7_: bool
                out6_, out7_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_9_nextIn_ = out6_
                d_10_wasConstrained_ = out7_
                if (d_9_nextIn_) == (eosToken):
                    cost = 1
                elif True:
                    d_11_appendedGenerated_: _dafny.Seq
                    d_12_appendedInside_: bool
                    d_13_appendedCurrent_: _dafny.Seq
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_nextIn_)
                    d_11_appendedGenerated_ = out8_
                    d_12_appendedInside_ = out9_
                    d_13_appendedCurrent_ = out10_
                    generated = d_11_appendedGenerated_
                    insideConstrainedOut = d_12_appendedInside_
                    currentConstrainedOut = d_13_appendedCurrent_
                    cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

