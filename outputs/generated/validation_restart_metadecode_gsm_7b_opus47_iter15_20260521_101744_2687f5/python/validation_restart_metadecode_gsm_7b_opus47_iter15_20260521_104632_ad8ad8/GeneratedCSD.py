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
        insideConstrainedOut = False
        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. For every arithmetic computation, immediately wrap it in << and >> delimiters in the canonical GSM form, for example <<3+4=7>>. Each << must be closed by a matching >> on the same line, and each span must contain exactly one short arithmetic equation of the form a<op>b=c. Do not leave any << unclosed. After all reasoning steps, end your answer with a final line of the form: #### <integer>. Keep the response concise.")))
        if (maxSteps) > (0):
            d_1_budget_: int
            d_1_budget_ = maxSteps
            d_2_minReasoning_: int
            if (maxSteps) >= (8):
                d_2_minReasoning_ = 4
            elif True:
                d_2_minReasoning_ = 1
            d_3_craneGenerated_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).CraneGeneration(lm, parser, (prompt) + (generated), d_1_budget_, d_2_minReasoning_, eosToken)
            d_3_craneGenerated_ = out0_
            if (len(d_3_craneGenerated_)) <= (d_1_budget_):
                generated = (generated) + (d_3_craneGenerated_)
                cost = len(d_3_craneGenerated_)
                if (cost) == (0):
                    d_4_next_: _dafny.Seq
                    out1_: _dafny.Seq
                    out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_4_next_ = out1_
                    cost = 1
                    if (d_4_next_) != (eosToken):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
            elif True:
                cost = 0
        return generated, insideConstrainedOut, currentConstrainedOut, cost

