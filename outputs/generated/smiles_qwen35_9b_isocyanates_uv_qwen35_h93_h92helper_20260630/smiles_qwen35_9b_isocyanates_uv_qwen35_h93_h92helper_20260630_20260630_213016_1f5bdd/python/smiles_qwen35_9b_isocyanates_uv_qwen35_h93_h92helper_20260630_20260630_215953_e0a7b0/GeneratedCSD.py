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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You must output a single valid SMILES string for a molecule in the ISOCYANATE class. Isocyanates contain the functional group -N=C=O (nitrogen double-bonded to carbon double-bonded to oxygen). The SMILES must contain the fragment N=C=O or O=C=N. Valid examples: O=C=NCC, O=C=NCCC, O=C=NC(C)C, O=C=Nc1ccccc1, CCN=C=O, CCCN=C=O, CC(C)N=C=O. You must NOT copy these examples. Start your SMILES with O=C=N and add a substituent R group to complete the isocyanate. Output the SMILES string immediately.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_prefixBudget_: int
        d_2_prefixBudget_ = 60
        if (d_2_prefixBudget_) > (maxSteps):
            d_2_prefixBudget_ = maxSteps
        d_3_resGenerated_: _dafny.Seq
        d_4_resInside_: bool
        d_5_resCurrent_: _dafny.Seq
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out0_, out1_, out2_ = (d_0_helpers_).GenerateWithPrefixAndManagedSpan(lm, parser, prompt, generated, insideConstrainedOut, currentConstrainedOut, maxSteps, d_2_prefixBudget_, validTokenGroups, _dafny.BigRational('4e0'), 8, eosToken)
        d_3_resGenerated_ = out0_
        d_4_resInside_ = out1_
        d_5_resCurrent_ = out2_
        generated = d_3_resGenerated_
        insideConstrainedOut = d_4_resInside_
        currentConstrainedOut = d_5_resCurrent_
        cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

