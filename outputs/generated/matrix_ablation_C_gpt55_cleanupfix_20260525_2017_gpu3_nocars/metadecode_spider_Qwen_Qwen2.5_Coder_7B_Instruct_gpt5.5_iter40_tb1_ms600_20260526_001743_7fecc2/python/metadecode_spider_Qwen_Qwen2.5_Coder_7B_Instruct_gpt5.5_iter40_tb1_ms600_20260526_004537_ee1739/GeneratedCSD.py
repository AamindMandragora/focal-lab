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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Return exactly one answer in the form SQL: <<query>> and nothing else. Generate one valid SQLite query for the given schema and question. Use only provided schema table and column names, prefer explicit joins, and choose Spider-style set operations, grouping, and filters when required.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (maxSteps) == (0):
            pass
        elif (maxSteps) == (1):
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:"))]))
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            d_1_steps_ = 1
        elif (maxSteps) == (2):
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            d_1_steps_ = 2
        elif True:
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            d_2_openedGenerated_: _dafny.Seq
            d_3_openedInside_: bool
            d_4_openedCurrent_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_2_openedGenerated_ = out0_
            d_3_openedInside_ = out1_
            d_4_openedCurrent_ = out2_
            generated = d_2_openedGenerated_
            insideConstrainedOut = d_3_openedInside_
            currentConstrainedOut = d_4_openedCurrent_
            d_1_steps_ = 3
            with _dafny.label("1_1_1_0"):
                while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                    with _dafny.c_label("1_1_1_0"):
                        d_5_stablePrefix_: _dafny.Seq
                        d_5_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_6_constrainedPrompt_: _dafny.Seq
                        d_6_constrainedPrompt_ = (prompt) + (d_5_stablePrefix_)
                        d_7_remaining_: int
                        d_7_remaining_ = (maxSteps) - (d_1_steps_)
                        d_8_symbolGenerated_: _dafny.Seq
                        d_9_symbolOut_: _dafny.Seq
                        d_10_hitEos_: bool
                        d_11_stepsUsed_: int
                        out3_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: int
                        out3_, out4_, out5_, out6_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_6_constrainedPrompt_, generated, currentConstrainedOut, d_7_remaining_, eosToken)
                        d_8_symbolGenerated_ = out3_
                        d_9_symbolOut_ = out4_
                        d_10_hitEos_ = out5_
                        d_11_stepsUsed_ = out6_
                        generated = d_8_symbolGenerated_
                        currentConstrainedOut = d_9_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                        if d_10_hitEos_:
                            raise _dafny.Break("1_1_1_0")
                        pass
                pass
            if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                d_12_closedGenerated_: _dafny.Seq
                d_13_closedInside_: bool
                d_14_closedCurrent_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_12_closedGenerated_ = out7_
                d_13_closedInside_ = out8_
                d_14_closedCurrent_ = out9_
                generated = d_12_closedGenerated_
                insideConstrainedOut = d_13_closedInside_
                currentConstrainedOut = d_14_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

