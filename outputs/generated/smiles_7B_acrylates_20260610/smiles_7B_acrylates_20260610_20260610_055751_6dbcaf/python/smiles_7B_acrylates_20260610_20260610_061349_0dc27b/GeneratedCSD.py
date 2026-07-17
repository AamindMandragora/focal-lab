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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output a single valid SMILES string for a novel acrylate molecule. Acrylates contain the substructure C=CC(=O)O (ester) or C=CC(=O)N (amide). The SMILES must start with C=C or end with OC(=O)C=C pattern. Examples of valid acrylate SMILES: C=CC(=O)OCC, C=CC(=O)OCCO, C=CC(=O)OC(C)C, CCOC(=O)C=C, C=CC(=O)OCC(CC)CCCC. Output ONLY the SMILES string."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_2_constrainedGenerated_: _dafny.Seq
            d_3_terminatedByEos_: bool
            out0_: _dafny.Seq
            out1_: bool
            out0_, out1_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, maxSteps, eosToken)
            d_2_constrainedGenerated_ = out0_
            d_3_terminatedByEos_ = out1_
            generated = (generatedPrefix) + (d_2_constrainedGenerated_)
            if (len(d_2_constrainedGenerated_)) > (0):
                cost = len(d_2_constrainedGenerated_)
                if d_3_terminatedByEos_:
                    cost = (cost) + (1)
                if (cost) > (maxSteps):
                    cost = maxSteps
            elif True:
                cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

