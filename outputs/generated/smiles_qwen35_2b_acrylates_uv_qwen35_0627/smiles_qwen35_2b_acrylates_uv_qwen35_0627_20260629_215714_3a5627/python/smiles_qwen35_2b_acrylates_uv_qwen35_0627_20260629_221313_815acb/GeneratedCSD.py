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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one valid SMILES for a novel acrylate ester molecule. The SMILES must include the vinyl acrylate core C=CC(=O)O or C=C(C)C(=O)O with an ester substituent of at least 2 carbons (e.g. ethyl, propyl, butyl, cyclohexyl, benzyl). Examples of valid acrylates: C=CC(=O)OCC, C=CC(=O)OCCC, C=CC(=O)OCC(C)C, C=C(C)C(=O)OCCCC, C=CC(=O)OCCCCC. Output ONLY the SMILES string, nothing else. Do not copy any example from the prompt context.")))
        if (insideConstrainedOut) and ((maxSteps) > (0)):
            d_1_closeBudget_: int
            d_1_closeBudget_ = _dafny.euclidian_division(maxSteps, 2)
            if (d_1_closeBudget_) < (1):
                d_1_closeBudget_ = 1
            d_2_cg_: _dafny.Seq
            d_3_ci_: bool
            d_4_cc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_1_closeBudget_)
            d_2_cg_ = out0_
            d_3_ci_ = out1_
            d_4_cc_ = out2_
            generated = d_2_cg_
            insideConstrainedOut = d_3_ci_
            currentConstrainedOut = d_4_cc_
            cost = d_1_closeBudget_
        if (not(insideConstrainedOut)) and ((cost) < (maxSteps)):
            d_5_remainingBudget_: int
            d_5_remainingBudget_ = (maxSteps) - (cost)
            d_6_constrainedGenerated_: _dafny.Seq
            d_7_terminatedByEos_: bool
            out3_: _dafny.Seq
            out4_: bool
            out3_, out4_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, d_5_remainingBudget_, eosToken)
            d_6_constrainedGenerated_ = out3_
            d_7_terminatedByEos_ = out4_
            generated = (generatedPrefix) + (d_6_constrainedGenerated_)
            cost = maxSteps
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

