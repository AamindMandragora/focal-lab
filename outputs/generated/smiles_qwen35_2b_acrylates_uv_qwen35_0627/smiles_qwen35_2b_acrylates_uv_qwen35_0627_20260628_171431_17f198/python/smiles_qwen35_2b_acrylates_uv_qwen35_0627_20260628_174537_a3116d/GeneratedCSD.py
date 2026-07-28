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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SMILES string for an acrylate ester molecule. Acrylates contain the CH2=CH-C(=O)-O-R substructure. Common examples: C=CC(=O)OCC, C=CC(=O)OCCO, C=CC(=O)OC. Output only the SMILES string.")))
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        d_2_closeReserve_: int
        d_2_closeReserve_ = 10
        if (d_2_closeReserve_) > (maxSteps):
            d_2_closeReserve_ = maxSteps
        with _dafny.label("0"):
            while (insideConstrainedOut) and (((d_1_steps_) + (d_2_closeReserve_)) < (maxSteps)):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        raise _dafny.Break("0")
                    d_3_constrainedPrompt_: _dafny.Seq
                    d_3_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_4_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_3_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                    d_4_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_4_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_4_next_)
                        generated = out4_
                        insideConstrainedOut = out5_
                        currentConstrainedOut = out6_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_5_closeBudget_: int
            d_5_closeBudget_ = (maxSteps) - (d_1_steps_)
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_5_closeBudget_)
            generated = out7_
            insideConstrainedOut = out8_
            currentConstrainedOut = out9_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

