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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output a single valid SMILES string for a novel acrylate ester. Acrylates have the CH2=CH-C(=O)-O- core (acryloyl group). Example structures: C=CC(=O)OCC, C=CC(=O)OCCCO, C=CC(=O)OC(C)C, C=CC(=O)OCCO, C=CC(=O)OCC(C)C. Generate a new one not in the examples. Output the SMILES only."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if (maxSteps) == (0):
            pass
        elif True:
            d_2_localConstrained_: _dafny.Seq
            d_2_localConstrained_ = _dafny.SeqWithoutIsStrInference([])
            d_3_steps_: int
            d_3_steps_ = 0
            with _dafny.label("1_0"):
                while (d_3_steps_) < (maxSteps):
                    with _dafny.c_label("1_0"):
                        if ((parser).IsCompletePrefix(d_2_localConstrained_)) and ((len(d_2_localConstrained_)) > (0)):
                            raise _dafny.Break("1_0")
                        d_4_constrainedPrompt_: _dafny.Seq
                        d_4_constrainedPrompt_ = (prompt) + (generatedPrefix)
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_4_constrainedPrompt_, d_2_localConstrained_, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                        d_5_next_ = out0_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            d_2_localConstrained_ = (d_2_localConstrained_) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        pass
                pass
            cost = d_3_steps_
            if ((cost) == (0)) and ((maxSteps) > (0)):
                cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

