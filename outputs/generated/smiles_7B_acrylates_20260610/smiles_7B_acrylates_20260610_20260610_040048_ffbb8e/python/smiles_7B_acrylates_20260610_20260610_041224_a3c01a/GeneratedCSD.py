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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SMILES string for a novel acrylate ester molecule. Acrylates contain the CH2=CH-C(=O)-O- group. Output ONLY the SMILES string with no explanation, no Markdown, no punctuation. Example acrylates: C=CC(=O)OCC, C=CC(=O)OCCCO, C=CC(=O)OC(C)C, C=CC(=O)OCCO, C=CC(=O)OCC(C)C, C=CC(=O)OCCCCC, C=CC(=O)OC1CCCCC1, C=CC(=O)OCC(O)CO. Generate a NEW acrylate not listed above."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_localConstrained_: _dafny.Seq
        d_2_localConstrained_ = _dafny.SeqWithoutIsStrInference([])
        d_3_steps_: int
        d_3_steps_ = 0
        with _dafny.label("0"):
            while (d_3_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_4_constrainedPrompt_: _dafny.Seq
                    d_4_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_5_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_4_constrainedPrompt_, d_2_localConstrained_, generated, _dafny.BigRational('2e0'), eosToken)
                    d_5_next_ = out0_
                    d_3_steps_ = (d_3_steps_) + (1)
                    if (d_5_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        d_2_localConstrained_ = (d_2_localConstrained_) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        if (parser).IsCompletePrefix(d_2_localConstrained_):
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

