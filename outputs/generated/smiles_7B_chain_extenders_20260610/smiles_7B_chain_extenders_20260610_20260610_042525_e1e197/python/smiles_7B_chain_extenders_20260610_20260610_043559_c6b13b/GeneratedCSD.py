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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES string for a chain_extender molecule. Chain extenders are small bifunctional molecules used in polyurethane synthesis: diols (two OH groups), diamines (two NH2 groups), or amino alcohols (one OH and one NH2). Examples include ethylene glycol (OCCO), 1,4-butanediol (OCCCCO), ethylenediamine (NCCN), 1,6-hexanediol (OCCCCCCO). Output ONLY the SMILES string for a novel chain extender not present in the prompt, with 2-8 carbons and exactly two reactive functional end groups. No explanation, no extra text."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if (maxSteps) > (0):
            d_2_constrainedGenerated_: _dafny.Seq
            d_3_terminatedByEos_: bool
            out0_: _dafny.Seq
            out1_: bool
            out0_, out1_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, maxSteps, eosToken)
            d_2_constrainedGenerated_ = out0_
            d_3_terminatedByEos_ = out1_
            generated = (generatedPrefix) + (d_2_constrainedGenerated_)
            cost = maxSteps
        elif True:
            cost = 0
        return generated, insideConstrainedOut, currentConstrainedOut, cost

