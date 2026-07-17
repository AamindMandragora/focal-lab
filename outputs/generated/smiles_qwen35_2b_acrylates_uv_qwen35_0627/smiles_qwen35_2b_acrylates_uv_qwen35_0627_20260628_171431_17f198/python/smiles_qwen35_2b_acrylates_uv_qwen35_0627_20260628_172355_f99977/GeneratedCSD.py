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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one valid SMILES string for a novel acrylate ester compound. An acrylate contains the CH2=CH-C(=O)-O- group. Output ONLY the SMILES, no other text. Example acrylates: C=CC(=O)OCC, C=CC(=O)OCCO, C=CC(=O)OC(C)C, C=CC(=O)OCCCO.")))
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        d_2_genBudget_: int
        if (maxSteps) > (10):
            d_2_genBudget_ = (maxSteps) - (5)
        elif True:
            d_2_genBudget_ = maxSteps
        if ((d_1_steps_) < (d_2_genBudget_)) and (insideConstrainedOut):
            d_3_stable_: _dafny.Seq
            d_3_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
            d_4_constrainedPrompt_: _dafny.Seq
            d_4_constrainedPrompt_ = (prompt) + (d_3_stable_)
            d_5_constrainedGenerated_: _dafny.Seq
            d_6_terminatedByEos_: bool
            out3_: _dafny.Seq
            out4_: bool
            out3_, out4_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, d_4_constrainedPrompt_, d_2_genBudget_, eosToken)
            d_5_constrainedGenerated_ = out3_
            d_6_terminatedByEos_ = out4_
            generated = (d_3_stable_) + (d_5_constrainedGenerated_)
            currentConstrainedOut = d_5_constrainedGenerated_
            d_1_steps_ = (d_1_steps_) + (d_2_genBudget_)
        if insideConstrainedOut:
            if (parser).IsCompletePrefix(currentConstrainedOut):
                if (d_1_steps_) < (maxSteps):
                    d_7_cg_: _dafny.Seq
                    d_8_ci_: bool
                    d_9_cc_: _dafny.Seq
                    out5_: _dafny.Seq
                    out6_: bool
                    out7_: _dafny.Seq
                    out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_7_cg_ = out5_
                    d_8_ci_ = out6_
                    d_9_cc_ = out7_
                    generated = d_7_cg_
                    insideConstrainedOut = d_8_ci_
                    currentConstrainedOut = d_9_cc_
                    d_1_steps_ = (d_1_steps_) + (1)
            elif (d_1_steps_) < (maxSteps):
                d_10_closeBudget_: int
                d_10_closeBudget_ = (maxSteps) - (d_1_steps_)
                d_11_cg_: _dafny.Seq
                d_12_ci_: bool
                d_13_cc_: _dafny.Seq
                out8_: _dafny.Seq
                out9_: bool
                out10_: _dafny.Seq
                out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_closeBudget_)
                d_11_cg_ = out8_
                d_12_ci_ = out9_
                d_13_cc_ = out10_
                generated = d_11_cg_
                insideConstrainedOut = d_12_ci_
                currentConstrainedOut = d_13_cc_
                d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

