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
        (d_0_helpers_).AppendTaskGuidance(lm, ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL SELECT statement in the format: SQL: <<SELECT ...>>. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use only table and column names from the provided schema. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do not add any explanation, markdown, or multiple queries."))))
        d_1_steps_: int
        d_1_steps_ = 0
        if (d_1_steps_) < (maxSteps):
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL"))]))
            d_1_steps_ = (d_1_steps_) + (1)
        if (d_1_steps_) < (maxSteps):
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":"))]))
            d_1_steps_ = (d_1_steps_) + (1)
        if (d_1_steps_) < (maxSteps):
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
            d_1_steps_ = (d_1_steps_) + (1)
        if (d_1_steps_) < (maxSteps):
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_2_chunkBudget_: int
                    if ((maxSteps) - (d_1_steps_)) > (1):
                        d_2_chunkBudget_ = ((maxSteps) - (d_1_steps_)) - (1)
                    elif True:
                        d_2_chunkBudget_ = 0
                    if (d_2_chunkBudget_) == (0):
                        raise _dafny.Break("0")
                    d_3_genOut_: _dafny.Seq
                    d_4_stoppedOnOpen_: bool
                    d_5_stoppedOnEos_: bool
                    d_6_stepsUsed_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_3_genOut_ = out0_
                    d_4_stoppedOnOpen_ = out1_
                    d_5_stoppedOnEos_ = out2_
                    d_6_stepsUsed_ = out3_
                    generated = d_3_genOut_
                    d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                    raise _dafny.Break("0")
                    pass
            pass
        if (d_1_steps_) < (maxSteps):
            if ((len(generated)) == (0)) or (((generated)[(len(generated)) - (1)]) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

