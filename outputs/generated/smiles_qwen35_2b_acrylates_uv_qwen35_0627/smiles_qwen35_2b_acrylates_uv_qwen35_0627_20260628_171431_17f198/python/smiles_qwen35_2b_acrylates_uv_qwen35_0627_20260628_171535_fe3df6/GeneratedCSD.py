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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SMILES string for a novel acrylate ester. Output only the SMILES, nothing else. Example acrylate pattern: C=CC(=O)O followed by an ester group.")))
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        d_2_genBudget_: int
        if (maxSteps) >= (4):
            d_2_genBudget_ = (maxSteps) - (2)
        elif True:
            d_2_genBudget_ = maxSteps
        if ((d_2_genBudget_) >= (1)) and (insideConstrainedOut):
            d_3_constrainedGenerated_: _dafny.Seq
            d_4_terminatedByEos_: bool
            out3_: _dafny.Seq
            out4_: bool
            out3_, out4_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, d_2_genBudget_, eosToken)
            d_3_constrainedGenerated_ = out3_
            d_4_terminatedByEos_ = out4_
            currentConstrainedOut = d_3_constrainedGenerated_
            generated = (generated) + (d_3_constrainedGenerated_)
            d_1_steps_ = d_2_genBudget_
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_5_closeBudget_: int
            d_5_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_6_cg_: _dafny.Seq
            d_7_ci_: bool
            d_8_cc_: _dafny.Seq
            out5_: _dafny.Seq
            out6_: bool
            out7_: _dafny.Seq
            out5_, out6_, out7_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_5_closeBudget_)
            d_6_cg_ = out5_
            d_7_ci_ = out6_
            d_8_cc_ = out7_
            generated = d_6_cg_
            insideConstrainedOut = d_7_ci_
            currentConstrainedOut = d_8_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

