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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query. Put the entire final query inside one visible << >> span and do not add any explanation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedHere_: bool
        d_2_openedHere_ = False
        d_3_closedHere_: bool
        d_3_closedHere_ = False
        if ((maxSteps) > (0)) and (not(insideConstrainedOut)):
            d_4_openedGenerated_: _dafny.Seq
            d_5_openedInside_: bool
            d_6_openedCurrent_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_4_openedGenerated_ = out0_
            d_5_openedInside_ = out1_
            d_6_openedCurrent_ = out2_
            generated = d_4_openedGenerated_
            insideConstrainedOut = d_5_openedInside_
            currentConstrainedOut = d_6_openedCurrent_
            d_1_steps_ = (d_1_steps_) + (1)
            d_2_openedHere_ = True
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                with _dafny.c_label("0"):
                    d_7_stablePrefix_: _dafny.Seq
                    d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_8_constrainedPrompt_: _dafny.Seq
                    d_8_constrainedPrompt_ = (prompt) + (d_7_stablePrefix_)
                    d_9_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                    d_9_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_9_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        d_10_appendedGenerated_: _dafny.Seq
                        d_11_appendedInside_: bool
                        d_12_appendedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                        d_10_appendedGenerated_ = out4_
                        d_11_appendedInside_ = out5_
                        d_12_appendedCurrent_ = out6_
                        generated = d_10_appendedGenerated_
                        insideConstrainedOut = d_11_appendedInside_
                        currentConstrainedOut = d_12_appendedCurrent_
                    pass
            pass
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
            d_3_closedHere_ = True
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

