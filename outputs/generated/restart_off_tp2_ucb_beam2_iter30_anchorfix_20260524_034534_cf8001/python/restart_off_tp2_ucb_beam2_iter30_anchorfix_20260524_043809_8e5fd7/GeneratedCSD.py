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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this gsm_symbolic word problem. CRITICAL RULES: (1) Use bare variable names verbatim (no braces) inside expressions, e.g. write n1 not {n1}. (2) Re-read the problem carefully and list every named quantity before computing. Identify exactly what the question asks for. (3) Each intermediate arithmetic step goes inside << >>. Use // for integer division when counting discrete items. (4) The final answer MUST be the LAST << >> in your response, and it MUST be wrapped in int(...) like <<int(n1*c1 + n2*c2 + c3)>> because gsm_symbolic answers are integers. (5) Double-check: does your expression use the right variable in each spot? (e.g. if the problem says 'k_3 times', the term must use k_3 not k_2). Be concise but careful. End immediately after the final <<int(...)>> expression.")))
        while (cost) < (maxSteps):
            d_1_remaining_: int
            d_1_remaining_ = (maxSteps) - (cost)
            d_2_newGenerated_: _dafny.Seq
            d_3_stoppedOnOpenSpan_: bool
            d_4_stoppedOnEos_: bool
            d_5_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_1_remaining_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_2_newGenerated_ = out0_
            d_3_stoppedOnOpenSpan_ = out1_
            d_4_stoppedOnEos_ = out2_
            d_5_stepsUsed_ = out3_
            generated = d_2_newGenerated_
            cost = (cost) + (d_5_stepsUsed_)
            if d_4_stoppedOnEos_:
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            if (d_5_stepsUsed_) == (0):
                return generated, insideConstrainedOut, currentConstrainedOut, cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost

