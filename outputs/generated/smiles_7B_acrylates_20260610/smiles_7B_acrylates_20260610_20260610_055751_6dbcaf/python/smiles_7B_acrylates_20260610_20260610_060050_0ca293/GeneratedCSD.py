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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SMILES string for a novel acrylate molecule. Acrylates contain the acryloyl group C=CC(=O)O or C=CC(=O)N. Examples of acrylate SMILES: CCOC(=O)C=C (ethyl acrylate), COC(=O)C=C (methyl acrylate), OC(=O)C=C (acrylic acid), C=CC(=O)OCC (ethyl acrylate), C=CC(=O)OCCO (2-hydroxyethyl acrylate). Output only the SMILES string, nothing else."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_budget_: int
        if (maxSteps) > (2):
            d_2_budget_ = (maxSteps) - (1)
        elif True:
            d_2_budget_ = maxSteps
        d_3_constrainedGenerated_: _dafny.Seq
        d_4_terminatedByEos_: bool
        out0_: _dafny.Seq
        out1_: bool
        out0_, out1_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, d_2_budget_, eosToken)
        d_3_constrainedGenerated_ = out0_
        d_4_terminatedByEos_ = out1_
        generated = (generatedPrefix) + (d_3_constrainedGenerated_)
        insideConstrainedOut = False
        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        d_5_genLen_: int
        d_5_genLen_ = len(d_3_constrainedGenerated_)
        if (d_5_genLen_) < (d_2_budget_):
            cost = (d_5_genLen_) + (1)
        elif True:
            cost = d_2_budget_
        if (cost) > (maxSteps):
            cost = maxSteps
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

