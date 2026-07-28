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
        if (maxSteps) == (0):
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SMILES string for a novel acrylate molecule. Acrylates contain CH2=CH-C(=O)-O- (vinyl ester of acrylic acid). Output ONLY the SMILES string with no explanation, no punctuation, no markdown. Valid acrylate SMILES examples: C=CC(=O)OCC, C=CC(=O)OCCO, C=CC(=O)OC(C)C. Generate a new one not in examples."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_constrainedGenerated_: _dafny.Seq
        d_3_terminatedByEos_: bool
        out0_: _dafny.Seq
        out1_: bool
        out0_, out1_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, maxSteps, eosToken)
        d_2_constrainedGenerated_ = out0_
        d_3_terminatedByEos_ = out1_
        d_4_newTokens_: _dafny.Seq
        d_4_newTokens_ = d_2_constrainedGenerated_
        d_5_newLen_: int
        d_5_newLen_ = len(d_4_newTokens_)
        if (d_5_newLen_) > (maxSteps):
            d_5_newLen_ = maxSteps
        d_6_i_: int
        d_6_i_ = 0
        while ((d_6_i_) < (d_5_newLen_)) and ((cost) < (maxSteps)):
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([(d_4_newTokens_)[d_6_i_]]))
            cost = (cost) + (1)
            d_6_i_ = (d_6_i_) + (1)
        if ((cost) == (0)) and ((maxSteps) > (0)):
            d_7_next_: _dafny.Seq
            out2_: _dafny.Seq
            out2_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
            d_7_next_ = out2_
            cost = 1
            if (d_7_next_) != (eosToken):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
        return generated, insideConstrainedOut, currentConstrainedOut, cost

