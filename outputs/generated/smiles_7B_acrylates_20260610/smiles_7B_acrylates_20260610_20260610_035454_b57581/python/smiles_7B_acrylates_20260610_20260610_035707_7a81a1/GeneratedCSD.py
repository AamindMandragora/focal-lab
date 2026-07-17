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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output only a single valid SMILES string for a novel acrylate-class molecule. Acrylates contain the C=CC(=O)O motif or ester derivatives. Example structures: CC(=O)OCC=C, CCOC(=O)C=C, C=CC(=O)OCCCC. Do not copy prompt examples. Output the SMILES string only, no explanation, no extra text.")))
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
            cost = 1
            if ((maxSteps) > (0)) and ((cost) < (maxSteps)):
                cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

