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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are generating SMILES strings for acrylate molecules. An acrylate MUST contain the acryloyl ester group. Generate complete SMILES like: C=CC(=O)OCC, C=CC(=O)OCCC, C=CC(=O)OC, C=C(C)C(=O)OCC, C=CC(=O)OCCCC, C=CC(=O)OC(C)C. The SMILES must be chemically valid and contain the C=CC(=O)O substructure (vinyl acrylate group). Do not generate single atoms."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if insideConstrainedOut:
            d_2_rg_: _dafny.Seq
            d_3_rc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: _dafny.Seq
            out0_, out1_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_2_rg_ = out0_
            d_3_rc_ = out1_
            generated = d_2_rg_
            currentConstrainedOut = d_3_rc_
            d_4_isComplete_: bool
            d_4_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if (d_4_isComplete_) and ((cost) < (maxSteps)):
                d_5_cg_: _dafny.Seq
                d_6_ci_: bool
                d_7_cc_: _dafny.Seq
                out2_: _dafny.Seq
                out3_: bool
                out4_: _dafny.Seq
                out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_5_cg_ = out2_
                d_6_ci_ = out3_
                d_7_cc_ = out4_
                generated = d_5_cg_
                insideConstrainedOut = d_6_ci_
                currentConstrainedOut = d_7_cc_
                cost = (cost) + (1)
            elif not(d_4_isComplete_):
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        d_8_genBudget_: int
        if (maxSteps) > (cost):
            d_8_genBudget_ = (maxSteps) - (cost)
        elif True:
            d_8_genBudget_ = 0
        if (d_8_genBudget_) > (0):
            d_9_constrainedResult_: _dafny.Seq
            d_10_terminatedByEos_: bool
            out5_: _dafny.Seq
            out6_: bool
            out5_, out6_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, d_8_genBudget_, eosToken)
            d_9_constrainedResult_ = out5_
            d_10_terminatedByEos_ = out6_
            generated = (generatedPrefix) + (d_9_constrainedResult_)
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            cost = (cost) + (d_8_genBudget_)
        return generated, insideConstrainedOut, currentConstrainedOut, cost

