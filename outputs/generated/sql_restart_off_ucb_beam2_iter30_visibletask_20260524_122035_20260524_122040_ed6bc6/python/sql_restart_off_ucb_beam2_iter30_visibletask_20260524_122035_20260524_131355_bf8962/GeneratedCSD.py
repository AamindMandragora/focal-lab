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
        if (maxSteps) == (0):
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output only the final SQL query, then stop.")))
        d_1_steps_: int
        d_1_steps_ = 0
        while (d_1_steps_) < (maxSteps):
            if insideConstrainedOut:
                if (parser).IsCompletePrefix(currentConstrainedOut):
                    d_2_closedG_: _dafny.Seq
                    d_3_closedI_: bool
                    d_4_closedC_: _dafny.Seq
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_2_closedG_ = out0_
                    d_3_closedI_ = out1_
                    d_4_closedC_ = out2_
                    generated = d_2_closedG_
                    insideConstrainedOut = d_3_closedI_
                    currentConstrainedOut = d_4_closedC_
                    d_1_steps_ = (d_1_steps_) + (1)
                    cost = (cost) + (1)
                elif True:
                    d_5_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                    d_5_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    cost = (cost) + (1)
                    if (d_5_next_) == (eosToken):
                        return generated, insideConstrainedOut, currentConstrainedOut, cost
                    d_6_stillIncomplete_: bool
                    d_6_stillIncomplete_ = not((parser).IsCompletePrefix(currentConstrainedOut))
                    d_7_valid_: bool
                    out4_: bool
                    out4_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_5_next_)
                    d_7_valid_ = out4_
                    if (d_6_stillIncomplete_) and (d_7_valid_):
                        d_8_appG_: _dafny.Seq
                        d_9_appI_: bool
                        d_10_appC_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_5_next_)
                        d_8_appG_ = out5_
                        d_9_appI_ = out6_
                        d_10_appC_ = out7_
                        generated = d_8_appG_
                        insideConstrainedOut = d_9_appI_
                        currentConstrainedOut = d_10_appC_
            elif True:
                d_11_next_: _dafny.Seq
                out8_: _dafny.Seq
                out8_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_11_next_ = out8_
                d_1_steps_ = (d_1_steps_) + (1)
                cost = (cost) + (1)
                if (d_11_next_) == (eosToken):
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
        return generated, insideConstrainedOut, currentConstrainedOut, cost

