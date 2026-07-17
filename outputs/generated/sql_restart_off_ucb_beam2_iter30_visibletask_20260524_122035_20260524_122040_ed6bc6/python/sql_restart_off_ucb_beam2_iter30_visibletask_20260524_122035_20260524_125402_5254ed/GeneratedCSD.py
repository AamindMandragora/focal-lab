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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one line: SQL: <<query>> and nothing else. Use lowercase SQL keywords. Use fully-qualified table.column names without aliases. Use only schema tables and columns. End immediately after >>. No semicolons, no code fences, no explanations.")))
        d_1_steps_: int
        d_1_steps_ = 0
        while (d_1_steps_) < (maxSteps):
            if insideConstrainedOut:
                if (parser).IsCompletePrefix(currentConstrainedOut):
                    d_2_cg_: _dafny.Seq
                    d_3_ci_: bool
                    d_4_cc_: _dafny.Seq
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_2_cg_ = out0_
                    d_3_ci_ = out1_
                    d_4_cc_ = out2_
                    generated = d_2_cg_
                    insideConstrainedOut = d_3_ci_
                    currentConstrainedOut = d_4_cc_
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
                    d_6_g_: _dafny.Seq
                    d_7_ic_: bool
                    d_8_cc_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_5_next_)
                    d_6_g_ = out4_
                    d_7_ic_ = out5_
                    d_8_cc_ = out6_
                    generated = d_6_g_
                    insideConstrainedOut = d_7_ic_
                    currentConstrainedOut = d_8_cc_
            elif True:
                d_9_next_: _dafny.Seq
                out7_: _dafny.Seq
                out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_9_next_ = out7_
                d_1_steps_ = (d_1_steps_) + (1)
                cost = (cost) + (1)
                if (d_9_next_) == (eosToken):
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
        return generated, insideConstrainedOut, currentConstrainedOut, cost

