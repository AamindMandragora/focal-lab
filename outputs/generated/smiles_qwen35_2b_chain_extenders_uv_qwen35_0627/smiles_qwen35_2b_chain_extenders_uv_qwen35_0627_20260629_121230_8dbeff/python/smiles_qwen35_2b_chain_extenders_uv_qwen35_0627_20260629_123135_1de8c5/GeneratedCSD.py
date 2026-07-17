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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES string for a novel chain_extender molecule. Chain extenders for polyurethane synthesis are bifunctional small molecules: diols (two OH groups), diamines (two NH2 groups), or amino alcohols (one NH2 and one OH). Examples of structural patterns: OCCO, OCCCCO, OCCCCCCCO, NCCN, NCCCN, NCCO, OCC(O)CO. Generate something structurally distinct from these - consider branched versions, longer chains (C5-C8), ether linkages (OCCO backbone), or cyclic variants. Output ONLY the SMILES with no text."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_localCurrent_: _dafny.Seq
        d_3_localCurrent_ = _dafny.SeqWithoutIsStrInference([])
        d_4_constrainedPrompt_: _dafny.Seq
        d_4_constrainedPrompt_ = prompt
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(d_3_localCurrent_):
                        raise _dafny.Break("0")
                    d_5_validCount_: int
                    out0_: int
                    out0_ = (d_0_helpers_).ValidTokenCount(parser, d_3_localCurrent_)
                    d_5_validCount_ = out0_
                    d_6_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    if (d_5_validCount_) <= (5):
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_4_constrainedPrompt_, d_3_localCurrent_, validTokenGroups, _dafny.BigRational('8e0'), eosToken)
                        d_6_next_ = out1_
                    elif (d_5_validCount_) <= (15):
                        out2_: _dafny.Seq
                        out2_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_4_constrainedPrompt_, d_3_localCurrent_, validTokenGroups, _dafny.BigRational('5e0'), 15, eosToken)
                        d_6_next_ = out2_
                    elif True:
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_4_constrainedPrompt_, d_3_localCurrent_, generated, _dafny.BigRational('25e-1'), eosToken)
                        d_6_next_ = out3_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_6_next_) == (eosToken):
                        raise _dafny.Break("0")
                    d_7_valid_: bool
                    out4_: bool
                    out4_ = (d_0_helpers_).IsTokenValidNext(parser, d_3_localCurrent_, d_6_next_)
                    d_7_valid_ = out4_
                    if d_7_valid_:
                        d_3_localCurrent_ = (d_3_localCurrent_) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

