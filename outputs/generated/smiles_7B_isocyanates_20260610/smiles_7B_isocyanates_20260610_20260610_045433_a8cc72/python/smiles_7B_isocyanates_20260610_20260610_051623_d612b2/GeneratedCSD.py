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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES string for a novel isocyanate molecule. Isocyanates must contain the isocyanate group N=C=O (written as N=C=O or as part of O=C=N). Output only the SMILES string. Good examples: CCN=C=O, CCCN=C=O, O=C=NCC, ClCCN=C=O, BrCN=C=O, O=C=Nc1ccccc1. Generate a new isocyanate not in this list.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_2_cg_: _dafny.Seq
                d_3_ci_: bool
                d_4_cc_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_2_cg_ = out0_
                d_3_ci_ = out1_
                d_4_cc_ = out2_
                generated = d_2_cg_
                insideConstrainedOut = d_3_ci_
                currentConstrainedOut = d_4_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_5_rg_: _dafny.Seq
                d_6_rc_: _dafny.Seq
                out3_: _dafny.Seq
                out4_: _dafny.Seq
                out3_, out4_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_5_rg_ = out3_
                d_6_rc_ = out4_
                generated = d_5_rg_
                currentConstrainedOut = d_6_rc_
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                    d_7_cg2_: _dafny.Seq
                    d_8_ci2_: bool
                    d_9_cc2_: _dafny.Seq
                    out5_: _dafny.Seq
                    out6_: bool
                    out7_: _dafny.Seq
                    out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_7_cg2_ = out5_
                    d_8_ci2_ = out6_
                    d_9_cc2_ = out7_
                    generated = d_7_cg2_
                    insideConstrainedOut = d_8_ci2_
                    currentConstrainedOut = d_9_cc2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        if (d_1_steps_) < (maxSteps):
            d_10_remainingSteps_: int
            d_10_remainingSteps_ = (maxSteps) - (d_1_steps_)
            d_11_constrainedGenerated_: _dafny.Seq
            d_12_terminatedByEos_: bool
            out8_: _dafny.Seq
            out9_: bool
            out8_, out9_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, (prompt) + (generated), d_10_remainingSteps_, eosToken)
            d_11_constrainedGenerated_ = out8_
            d_12_terminatedByEos_ = out9_
            generated = (generated) + (d_11_constrainedGenerated_)
            d_1_steps_ = (d_1_steps_) + (d_10_remainingSteps_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

