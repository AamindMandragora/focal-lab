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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_done_: bool
        d_2_done_ = False
        while ((d_1_steps_) < (maxSteps)) and (not(d_2_done_)):
            d_3_oldSteps_: int
            d_3_oldSteps_ = d_1_steps_
            d_4_before_: _dafny.Seq
            d_4_before_ = generated
            if not(insideConstrainedOut):
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                generated = out0_
                insideConstrainedOut = out1_
                currentConstrainedOut = out2_
                d_1_steps_ = (d_3_oldSteps_) + (1)
                cost = d_1_steps_
            elif (parser).IsCompletePrefix(currentConstrainedOut):
                out3_: _dafny.Seq
                out4_: bool
                out5_: _dafny.Seq
                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                generated = out3_
                insideConstrainedOut = out4_
                currentConstrainedOut = out5_
                d_1_steps_ = (d_3_oldSteps_) + (1)
                cost = d_1_steps_
                d_2_done_ = True
            elif True:
                d_5_next_: _dafny.Seq
                out6_: _dafny.Seq
                out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, prompt, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                d_5_next_ = out6_
                d_1_steps_ = (d_3_oldSteps_) + (1)
                cost = d_1_steps_
                if (d_5_next_) == (eosToken):
                    d_2_done_ = True
                elif True:
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_5_next_)
                    generated = out7_
                    insideConstrainedOut = out8_
                    currentConstrainedOut = out9_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

