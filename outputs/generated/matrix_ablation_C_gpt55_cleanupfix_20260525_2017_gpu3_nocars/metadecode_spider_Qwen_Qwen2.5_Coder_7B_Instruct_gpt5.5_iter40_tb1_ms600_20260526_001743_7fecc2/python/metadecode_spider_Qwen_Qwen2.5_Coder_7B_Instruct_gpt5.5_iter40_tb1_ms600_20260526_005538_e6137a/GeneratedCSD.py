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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Return exactly one answer in the form SQL: <<query>> and nothing else. The query must be a single valid SQLite query for the given schema and question. Do not explain. Do not repeat SQL: or << inside the query. Prefer semantically direct Spider-style SQL using the exact table and column names from the schema.")))
        if (maxSteps) == (0):
            cost = 0
        elif (maxSteps) == (1):
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:"))]))
            cost = 1
        elif (maxSteps) == (2):
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
            cost = 2
        elif True:
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
            d_1_steps_: int
            d_1_steps_ = 3
            d_2_seenClose_: bool
            d_2_seenClose_ = False
            while (((d_1_steps_) + (1)) < (maxSteps)) and (not(d_2_seenClose_)):
                d_3_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_3_next_ = out0_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_3_next_) == (eosToken):
                    d_2_seenClose_ = True
                elif ((d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")))) or ((d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                    pass
                elif (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    d_2_seenClose_ = True
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
            if (not(d_2_seenClose_)) and ((d_1_steps_) < (maxSteps)):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                d_1_steps_ = (d_1_steps_) + (1)
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

