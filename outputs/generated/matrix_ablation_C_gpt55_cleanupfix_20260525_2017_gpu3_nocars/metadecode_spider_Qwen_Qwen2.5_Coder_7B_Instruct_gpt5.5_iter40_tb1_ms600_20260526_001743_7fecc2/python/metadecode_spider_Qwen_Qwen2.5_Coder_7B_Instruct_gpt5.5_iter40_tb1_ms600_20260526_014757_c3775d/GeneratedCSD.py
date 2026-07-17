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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Return exactly one answer and nothing else. The answer format is SQL: <<query>>. Inside the delimiters, write the single best executable SQLite query for the schema and question. Reason internally from the schema before emitting tokens. Prefer Spider-canonical semantics: use exact table and column names, include necessary joins, avoid unnecessary aliases, use EXCEPT for 'not used/not in' set differences when appropriate, use INTERSECT for 'both/all of' requirements when appropriate, and do not add explanations or a trailing semicolon.")))
        if (maxSteps) == (0):
            cost = 0
        elif True:
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_sawClose_: bool
            d_2_sawClose_ = False
            if (d_1_steps_) < (maxSteps):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:"))]))
                d_1_steps_ = (d_1_steps_) + (1)
            if (d_1_steps_) < (maxSteps):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
                d_1_steps_ = (d_1_steps_) + (1)
            if (d_1_steps_) < (maxSteps):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                d_1_steps_ = (d_1_steps_) + (1)
            with _dafny.label("1_0"):
                while ((d_1_steps_) < (maxSteps)) and (not(d_2_sawClose_)):
                    with _dafny.c_label("1_0"):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("1_0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                d_2_sawClose_ = True
                        pass
                pass
            if (not(d_2_sawClose_)) and ((d_1_steps_) < (maxSteps)):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                d_1_steps_ = (d_1_steps_) + (1)
                d_2_sawClose_ = True
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

