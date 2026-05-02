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
                    if insideConstrainedOut:
                        d_2_complete_: bool
                        d_2_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_2_complete_:
                            d_3_closedGenerated_: _dafny.Seq
                            d_4_closedInside_: bool
                            d_5_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_3_closedGenerated_ = out0_
                            d_4_closedInside_ = out1_
                            d_5_closedCurrent_ = out2_
                            generated = d_3_closedGenerated_
                            insideConstrainedOut = d_4_closedInside_
                            currentConstrainedOut = d_5_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_remaining_: int
                            d_6_remaining_ = (maxSteps) - (d_1_steps_)
                            d_7_symbolBudget_: int
                            d_7_symbolBudget_ = d_6_remaining_
                            if (stepTokenBudget) < (d_7_symbolBudget_):
                                d_7_symbolBudget_ = stepTokenBudget
                            if (d_7_symbolBudget_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_8_stablePrefix_: _dafny.Seq
                                d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_9_constrainedPrompt_: _dafny.Seq
                                d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                                d_10_symbolOut_: _dafny.Seq
                                d_11_hitEos_: bool
                                d_12_stepsUsed_: int
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: int
                                out3_, out4_, out5_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, d_7_symbolBudget_, eosToken)
                                d_10_symbolOut_ = out3_
                                d_11_hitEos_ = out4_
                                d_12_stepsUsed_ = out5_
                                generated = (d_8_stablePrefix_) + (d_10_symbolOut_)
                                currentConstrainedOut = d_10_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                                if d_11_hitEos_:
                                    raise _dafny.Break("0")
                    elif True:
                        d_13_openedGenerated_: _dafny.Seq
                        d_14_openedInside_: bool
                        d_15_openedCurrent_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_13_openedGenerated_ = out6_
                        d_14_openedInside_ = out7_
                        d_15_openedCurrent_ = out8_
                        generated = d_13_openedGenerated_
                        insideConstrainedOut = d_14_openedInside_
                        currentConstrainedOut = d_15_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

