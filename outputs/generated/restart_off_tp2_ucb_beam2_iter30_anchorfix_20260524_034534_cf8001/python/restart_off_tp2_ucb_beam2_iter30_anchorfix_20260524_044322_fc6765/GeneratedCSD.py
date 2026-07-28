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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You solve symbolic math word problems. Output rules: (R1) Use ONLY the variable names that appear in the problem text (e.g. n1, w1, frac2, total, sides). Do NOT invent variables (no c1 if only w1 is given). Write them WITHOUT braces inside expressions. (R2) Wrap each arithmetic step in << ... >>. The LAST <<...>> in your response is the final answer. (R3) For a count of discrete items use Python integer division //. For percentages, multiply by 100. Use int(...) ONLY when the problem says 'how many' and you actually had to truncate (typically with //, in which case int() is usually unnecessary). Prefer bare expressions over int() wrappers. (R4) Before computing, briefly restate each named quantity in plain text so you do not confuse them in multi-stage problems. (R5) Be concise; end immediately after the final <<...>>. Examples:\n- 'A truck carries {total} lbs. Items weigh {n1} bags of {w1} lbs and {n2} sacks of {w2} lbs. Trips? Total weight is <<n1*w1 + n2*w2>>. Trips = <<(n1*w1 + n2*w2)//total>>.'\n- 'They have {t} min to walk; {t1} min to corner, {t2} min more. Remaining = <<t - t1 - t2>>.'\n- 'Plants per ledge {r}, ledges {w}, give away {n} each, plus {x} new. Remaining = <<w*r + x - w*n>>.'\nNow solve the problem:")))
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

