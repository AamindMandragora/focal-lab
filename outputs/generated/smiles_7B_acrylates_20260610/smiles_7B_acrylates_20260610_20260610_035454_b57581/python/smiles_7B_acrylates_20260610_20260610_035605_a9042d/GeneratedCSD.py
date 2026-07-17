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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SMILES string for a novel acrylate-class molecule. The acrylate motif is C=CC(=O)O or a derivative. Output only the SMILES string. Do not copy any example from the prompt. No explanation, no Markdown, no delimiters.")))
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_constrainedGenerated_: _dafny.Seq
            d_2_terminatedByEos_: bool
            out0_: _dafny.Seq
            out1_: bool
            out0_, out1_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, maxSteps, eosToken)
            d_1_constrainedGenerated_ = out0_
            d_2_terminatedByEos_ = out1_
            generated = (generatedPrefix) + (d_1_constrainedGenerated_)
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

