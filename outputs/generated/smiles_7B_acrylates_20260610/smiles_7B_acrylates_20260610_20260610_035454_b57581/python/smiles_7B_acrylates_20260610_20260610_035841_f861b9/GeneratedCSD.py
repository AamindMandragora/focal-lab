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
        if (maxSteps) == (0):
            cost = 0
        elif True:
            (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single novel SMILES string for an acrylate ester. Acrylates contain the vinyl ester group: C=CC(=O)O-R where R is an organic group. Examples of acrylate SMILES: C=CC(=O)OCC, C=CC(=O)OCCO, C=CC(=O)OC(C)C. Generate a NEW acrylate not in the examples. Output only the SMILES string with no explanation.")))
            d_1_constrainedGenerated_: _dafny.Seq
            d_2_terminatedByEos_: bool
            out0_: _dafny.Seq
            out1_: bool
            out0_, out1_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, (maxSteps) - (1), eosToken)
            d_1_constrainedGenerated_ = out0_
            d_2_terminatedByEos_ = out1_
            generated = (generatedPrefix) + (d_1_constrainedGenerated_)
            d_3_tokensUsed_: int
            d_3_tokensUsed_ = len(d_1_constrainedGenerated_)
            if d_2_terminatedByEos_:
                d_3_tokensUsed_ = (d_3_tokensUsed_) + (1)
            if (d_3_tokensUsed_) == (0):
                cost = 1
            elif (d_3_tokensUsed_) > (maxSteps):
                cost = maxSteps
            elif True:
                cost = d_3_tokensUsed_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

