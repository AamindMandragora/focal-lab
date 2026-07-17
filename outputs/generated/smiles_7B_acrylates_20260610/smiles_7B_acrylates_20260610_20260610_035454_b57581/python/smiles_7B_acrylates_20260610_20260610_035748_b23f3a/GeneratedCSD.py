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
        insideConstrainedOut = False
        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output only a single valid SMILES string for a novel acrylate ester molecule. Acrylates contain the acryloyl group C=CC(=O)O. Produce a complete, syntactically valid SMILES string for a new acrylate not in the prompt examples. Output the SMILES only.")))
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_localCurrent_: _dafny.Seq
            d_2_localCurrent_ = _dafny.SeqWithoutIsStrInference([])
            d_3_done_: bool
            d_3_done_ = False
            while ((d_1_steps_) < (maxSteps)) and (not(d_3_done_)):
                d_4_constrainedPrompt_: _dafny.Seq
                d_4_constrainedPrompt_ = prompt
                d_5_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_4_constrainedPrompt_, d_2_localCurrent_, eosToken)
                d_5_next_ = out0_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_5_next_) == (eosToken):
                    d_3_done_ = True
                elif True:
                    d_6_valid_: bool
                    out1_: bool
                    out1_ = (d_0_helpers_).IsTokenValidNext(parser, d_2_localCurrent_, d_5_next_)
                    d_6_valid_ = out1_
                    if d_6_valid_:
                        d_2_localCurrent_ = (d_2_localCurrent_) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        if (parser).IsCompletePrefix(d_2_localCurrent_):
                            d_3_done_ = True
                    elif True:
                        d_3_done_ = True
            if (d_1_steps_) == (0):
                cost = 1
            elif True:
                cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

