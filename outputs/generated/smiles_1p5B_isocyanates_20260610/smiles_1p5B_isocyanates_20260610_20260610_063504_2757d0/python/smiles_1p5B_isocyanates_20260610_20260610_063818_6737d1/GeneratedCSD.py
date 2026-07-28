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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid novel SMILES string for the isocyanates class. Isocyanates contain the N=C=O functional group. Example pattern: R-N=C=O where R is any organic group. Output only the SMILES string, nothing else."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_constrainedOut_: _dafny.Seq
        d_3_terminatedByEos_: bool
        out0_: _dafny.Seq
        out1_: bool
        out0_, out1_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, (maxSteps) - (1), eosToken)
        d_2_constrainedOut_ = out0_
        d_3_terminatedByEos_ = out1_
        generated = (generatedPrefix) + (d_2_constrainedOut_)
        cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

