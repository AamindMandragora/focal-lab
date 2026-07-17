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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap every intermediate symbolic expression and the final answer inside visible << >> delimiters.")))
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_runBudget_: int
            d_1_runBudget_ = maxSteps
            d_2_minReasoningSteps_: int
            if (stepTokenBudget) == (0):
                d_2_minReasoningSteps_ = 1
            elif True:
                d_2_minReasoningSteps_ = stepTokenBudget
            d_3_baselineGenerated_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).CraneGeneration(lm, parser, (prompt) + (generatedPrefix), d_1_runBudget_, d_2_minReasoningSteps_, eosToken)
            d_3_baselineGenerated_ = out0_
            generated = d_3_baselineGenerated_
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            if ((len(generated)) > (0)) and (((generated)[(len(generated)) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                d_4_enteredGenerated_: _dafny.Seq
                d_5_enteredInside_: bool
                d_6_enteredCurrent_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_4_enteredGenerated_ = out1_
                d_5_enteredInside_ = out2_
                d_6_enteredCurrent_ = out3_
                generated = d_4_enteredGenerated_
                insideConstrainedOut = d_5_enteredInside_
                currentConstrainedOut = d_6_enteredCurrent_
            cost = d_1_runBudget_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

