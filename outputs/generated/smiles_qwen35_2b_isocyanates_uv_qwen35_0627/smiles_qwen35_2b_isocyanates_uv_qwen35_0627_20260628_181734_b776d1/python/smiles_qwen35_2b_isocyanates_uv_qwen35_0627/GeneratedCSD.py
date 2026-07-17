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
        (d_0_helpers_).AppendTaskGuidance(lm, (((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate one valid SMILES string for a NEW isocyanate molecule (contains N=C=O or [N+]=[C-]=O). "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output only the SMILES string. Do not repeat any molecule from the examples. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Isocyanate examples include: phenyl isocyanate C6H5NCO, methyl isocyanate CH3NCO, ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "hexyl isocyanate CCCCCCN=C=O. Generate a structurally different one."))))
        if not(insideConstrainedOut):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                with _dafny.c_label("0"):
                    if ((d_1_steps_) + (2)) >= (maxSteps):
                        raise _dafny.Break("0")
                    d_2_stable_: _dafny.Seq
                    d_2_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_3_constrainedPrompt_: _dafny.Seq
                    d_3_constrainedPrompt_ = (prompt) + (d_2_stable_)
                    d_4_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    d_5_validCount_: int
                    out3_: int
                    out3_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                    d_5_validCount_ = out3_
                    if (d_5_validCount_) <= (8):
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_3_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), eosToken)
                        d_4_next_ = out4_
                    elif True:
                        out5_: _dafny.Seq
                        out5_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_3_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                        d_4_next_ = out5_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_4_next_) == (eosToken):
                        raise _dafny.Break("0")
                    out6_: _dafny.Seq
                    out7_: bool
                    out8_: _dafny.Seq
                    out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_4_next_)
                    generated = out6_
                    insideConstrainedOut = out7_
                    currentConstrainedOut = out8_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_6_closeBudget_: int
            d_6_closeBudget_ = (maxSteps) - (d_1_steps_)
            out9_: _dafny.Seq
            out10_: bool
            out11_: _dafny.Seq
            out9_, out10_, out11_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_6_closeBudget_)
            generated = out9_
            insideConstrainedOut = out10_
            currentConstrainedOut = out11_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

