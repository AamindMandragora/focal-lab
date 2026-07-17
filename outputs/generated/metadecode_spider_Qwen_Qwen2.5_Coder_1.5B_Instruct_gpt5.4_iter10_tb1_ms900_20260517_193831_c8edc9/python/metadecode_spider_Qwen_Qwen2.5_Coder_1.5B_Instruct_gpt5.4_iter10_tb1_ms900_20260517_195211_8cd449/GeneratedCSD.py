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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Return exactly one SQL query as the entire answer. Put the full query inside one visible << >> span and output nothing else.")))
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_steps_: int
            d_1_steps_ = 0
            if not(insideConstrainedOut):
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
                d_1_steps_ = (d_1_steps_) + (1)
            if (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                d_5_stablePrefix_: _dafny.Seq
                d_5_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_6_constrainedPrompt_: _dafny.Seq
                d_6_constrainedPrompt_ = (prompt) + (d_5_stablePrefix_)
                d_7_remaining_: int
                d_7_remaining_ = (maxSteps) - (d_1_steps_)
                d_8_symbolBudget_: int
                if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_7_remaining_)):
                    d_8_symbolBudget_ = d_7_remaining_
                elif True:
                    d_8_symbolBudget_ = stepTokenBudget
                if (d_8_symbolBudget_) > (0):
                    d_9_symbolGenerated_: _dafny.Seq
                    d_10_symbolCurrent_: _dafny.Seq
                    d_11_hitEos_: bool
                    d_12_stepsUsed_: int
                    out3_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: int
                    out3_, out4_, out5_, out6_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_6_constrainedPrompt_, generated, currentConstrainedOut, d_8_symbolBudget_, eosToken)
                    d_9_symbolGenerated_ = out3_
                    d_10_symbolCurrent_ = out4_
                    d_11_hitEos_ = out5_
                    d_12_stepsUsed_ = out6_
                    generated = d_9_symbolGenerated_
                    insideConstrainedOut = True
                    currentConstrainedOut = d_10_symbolCurrent_
                    d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
            if (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                d_13_closedGenerated_: _dafny.Seq
                d_14_closedInside_: bool
                d_15_closedCurrent_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_13_closedGenerated_ = out7_
                d_14_closedInside_ = out8_
                d_15_closedCurrent_ = out9_
                generated = d_13_closedGenerated_
                insideConstrainedOut = d_14_closedInside_
                currentConstrainedOut = d_15_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

