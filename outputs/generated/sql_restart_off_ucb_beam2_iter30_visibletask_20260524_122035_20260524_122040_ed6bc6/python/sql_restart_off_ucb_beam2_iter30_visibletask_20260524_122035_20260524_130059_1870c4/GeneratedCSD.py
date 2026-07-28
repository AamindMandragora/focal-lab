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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SQL query that answers the question. Output only the SQL.")))
        d_1_steps_: int
        d_1_steps_ = 0
        while (d_1_steps_) < (maxSteps):
            if insideConstrainedOut:
                if (parser).IsCompletePrefix(currentConstrainedOut):
                    d_2_g2_: _dafny.Seq
                    d_3_ic2_: bool
                    d_4_cc2_: _dafny.Seq
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_2_g2_ = out0_
                    d_3_ic2_ = out1_
                    d_4_cc2_ = out2_
                    generated = d_2_g2_
                    insideConstrainedOut = d_3_ic2_
                    currentConstrainedOut = d_4_cc2_
                    d_1_steps_ = (d_1_steps_) + (1)
                    cost = d_1_steps_
                elif True:
                    d_5_nxt_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                    d_5_nxt_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    cost = d_1_steps_
                    if (d_5_nxt_) == (eosToken):
                        return generated, insideConstrainedOut, currentConstrainedOut, cost
                    d_6_ok_: bool
                    out4_: bool
                    out4_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_5_nxt_)
                    d_6_ok_ = out4_
                    if d_6_ok_:
                        d_7_g3_: _dafny.Seq
                        d_8_ic3_: bool
                        d_9_cc3_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_5_nxt_)
                        d_7_g3_ = out5_
                        d_8_ic3_ = out6_
                        d_9_cc3_ = out7_
                        generated = d_7_g3_
                        insideConstrainedOut = d_8_ic3_
                        currentConstrainedOut = d_9_cc3_
            elif True:
                d_10_nxt_: _dafny.Seq
                out8_: _dafny.Seq
                out8_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_10_nxt_ = out8_
                d_1_steps_ = (d_1_steps_) + (1)
                cost = d_1_steps_
                if (d_10_nxt_) == (eosToken):
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_nxt_]))
        return generated, insideConstrainedOut, currentConstrainedOut, cost

