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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Task: generate exactly one valid SMILES string for a novel acrylate ester molecule. Acrylates contain the vinyl ester core group C=CC(=O)O attached to an alkyl group R. Simple examples: C=CC(=O)OCC (ethyl acrylate), C=CC(=O)OCCC (propyl acrylate), C=CC(=O)OC (methyl acrylate), C=CC(=O)OC(C)C (isopropyl acrylate), C=CC(=O)OCCCC (butyl acrylate). Generate only the SMILES string, nothing else. Prefer simple aliphatic R groups over aromatic or ring-containing ones."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_prefixBudget_: int
        d_2_prefixBudget_ = 0
        d_3_boostAmount_: _dafny.BigRational
        d_3_boostAmount_ = _dafny.BigRational('8e0')
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 5
        d_5_gOut_: _dafny.Seq
        d_6_iOut_: bool
        d_7_cOut_: _dafny.Seq
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out0_, out1_, out2_ = (d_0_helpers_).GenerateWithPrefixAndManagedSpan(lm, parser, prompt, generated, insideConstrainedOut, currentConstrainedOut, maxSteps, d_2_prefixBudget_, validTokenGroups, d_3_boostAmount_, d_4_narrowThreshold_, eosToken)
        d_5_gOut_ = out0_
        d_6_iOut_ = out1_
        d_7_cOut_ = out2_
        generated = d_5_gOut_
        insideConstrainedOut = d_6_iOut_
        currentConstrainedOut = d_7_cOut_
        cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

