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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
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
                    elif True:
                        d_5_complete_: bool
                        d_5_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_complete_:
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out3_
                            d_7_closedInside_ = out4_
                            d_8_closedCurrent_ = out5_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_remaining_: int
                            d_9_remaining_ = (maxSteps) - (d_1_steps_)
                            if (d_9_remaining_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_10_stablePrefix_: _dafny.Seq
                                d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_11_constrainedPrompt_: _dafny.Seq
                                d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                                d_12_symbolCurrent_: _dafny.Seq
                                d_13_hitEos_: bool
                                d_14_stepsUsed_: int
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: int
                                out6_, out7_, out8_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, d_9_remaining_, eosToken)
                                d_12_symbolCurrent_ = out6_
                                d_13_hitEos_ = out7_
                                d_14_stepsUsed_ = out8_
                                currentConstrainedOut = d_12_symbolCurrent_
                                generated = (d_10_stablePrefix_) + (currentConstrainedOut)
                                d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                                if d_13_hitEos_:
                                    raise _dafny.Break("0")
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

