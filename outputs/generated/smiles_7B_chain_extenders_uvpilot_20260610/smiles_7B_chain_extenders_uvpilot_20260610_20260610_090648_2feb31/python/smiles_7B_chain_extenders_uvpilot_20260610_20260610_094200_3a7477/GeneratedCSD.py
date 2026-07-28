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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are generating SMILES strings for chain_extenders molecules. Chain extenders are small bifunctional molecules used in polymer synthesis. They typically have two reactive groups like -OH (diol) or -NH2 (diamine) connected by a short carbon chain. Examples of classes: aliphatic diols (e.g., butanediol), aromatic diols, aliphatic diamines, amino alcohols. Generate a NOVEL molecule not seen before. The SMILES must be chemically valid. Output ONLY the SMILES string with no other text.")))
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_og_: _dafny.Seq
            d_3_oi_: bool
            d_4_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_2_og_ = out0_
            d_3_oi_ = out1_
            d_4_oc_ = out2_
            generated = d_2_og_
            insideConstrainedOut = d_3_oi_
            currentConstrainedOut = d_4_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_5_remainingSteps_: int
        d_5_remainingSteps_ = 0
        if (maxSteps) > ((d_1_steps_) + (2)):
            d_5_remainingSteps_ = ((maxSteps) - (d_1_steps_)) - (2)
        elif (maxSteps) > (d_1_steps_):
            d_5_remainingSteps_ = (maxSteps) - (d_1_steps_)
        if (insideConstrainedOut) and ((d_5_remainingSteps_) > (0)):
            d_6_constrainedGenerated_: _dafny.Seq
            d_7_terminatedByEos_: bool
            out3_: _dafny.Seq
            out4_: bool
            out3_, out4_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, d_5_remainingSteps_, eosToken)
            d_6_constrainedGenerated_ = out3_
            d_7_terminatedByEos_ = out4_
            d_8_cgLen_: int
            d_8_cgLen_ = len(d_6_constrainedGenerated_)
            d_1_steps_ = (d_1_steps_) + (d_5_remainingSteps_)
            generated = (generated) + (d_6_constrainedGenerated_)
            currentConstrainedOut = d_6_constrainedGenerated_
            d_9_rg_: _dafny.Seq
            d_10_rc_: _dafny.Seq
            out5_: _dafny.Seq
            out6_: _dafny.Seq
            out5_, out6_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_9_rg_ = out5_
            d_10_rc_ = out6_
            generated = d_9_rg_
            currentConstrainedOut = d_10_rc_
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_11_cg_: _dafny.Seq
                d_12_ci_: bool
                d_13_cc_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_11_cg_ = out7_
                d_12_ci_ = out8_
                d_13_cc_ = out9_
                generated = d_11_cg_
                insideConstrainedOut = d_12_ci_
                currentConstrainedOut = d_13_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

