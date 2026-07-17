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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SMILES string for a novel acrylate molecule not seen in the examples. Acrylates contain C=CC(=O)O or similar acrylate ester pattern. Output only the SMILES string.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        if ((len(currentConstrainedOut)) == (0)) and ((d_1_steps_) < (maxSteps)):
            d_2_genBudget_: int
            d_2_genBudget_ = (maxSteps) - (d_1_steps_)
            d_3_constrainedGenerated_: _dafny.Seq
            d_4_terminatedByEos_: bool
            out3_: _dafny.Seq
            out4_: bool
            out3_, out4_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, (prompt) + (generated), d_2_genBudget_, eosToken)
            d_3_constrainedGenerated_ = out3_
            d_4_terminatedByEos_ = out4_
            generated = (generated) + (d_3_constrainedGenerated_)
            currentConstrainedOut = d_3_constrainedGenerated_
            d_1_steps_ = (d_1_steps_) + (len(d_3_constrainedGenerated_))
            if (d_4_terminatedByEos_) and ((d_1_steps_) < (maxSteps)):
                d_1_steps_ = (d_1_steps_) + (1)
            if (d_1_steps_) > (maxSteps):
                d_1_steps_ = maxSteps
        elif True:
            with _dafny.label("2_0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("2_0"):
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            raise _dafny.Break("2_0")
                        d_5_constrainedPrompt_: _dafny.Seq
                        d_5_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_6_next_: _dafny.Seq
                        out5_: _dafny.Seq
                        out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_5_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_6_next_ = out5_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("2_0")
                        elif True:
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_6_next_)
                            generated = out6_
                            insideConstrainedOut = out7_
                            currentConstrainedOut = out8_
                        pass
                pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

