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
        d_2_openedOnce_: bool
        d_2_openedOnce_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (not(d_2_openedOnce_)) and (not(insideConstrainedOut)):
                        d_3_openedGenerated_: _dafny.Seq
                        d_4_openedInside_: bool
                        d_5_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_3_openedGenerated_ = out0_
                        d_4_openedInside_ = out1_
                        d_5_openedCurrent_ = out2_
                        generated = d_3_openedGenerated_
                        insideConstrainedOut = d_4_openedInside_
                        currentConstrainedOut = d_5_openedCurrent_
                        d_2_openedOnce_ = True
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        if insideConstrainedOut:
                            d_6_complete_: bool
                            d_6_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_6_complete_:
                                d_7_closedGenerated_: _dafny.Seq
                                d_8_closedInside_: bool
                                d_9_closedCurrent_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_7_closedGenerated_ = out3_
                                d_8_closedInside_ = out4_
                                d_9_closedCurrent_ = out5_
                                generated = d_7_closedGenerated_
                                insideConstrainedOut = d_8_closedInside_
                                currentConstrainedOut = d_9_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_10_remaining_: int
                                d_10_remaining_ = (maxSteps) - (d_1_steps_)
                                d_11_symbolBudget_: int
                                d_11_symbolBudget_ = d_10_remaining_
                                if (stepTokenBudget) < (d_11_symbolBudget_):
                                    d_11_symbolBudget_ = stepTokenBudget
                                if (d_11_symbolBudget_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_12_stablePrefix_: _dafny.Seq
                                    d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_13_constrainedPrompt_: _dafny.Seq
                                    d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                                    d_14_symbolOut_: _dafny.Seq
                                    d_15_hitEos_: bool
                                    d_16_stepsUsed_: int
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: int
                                    out6_, out7_, out8_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, d_11_symbolBudget_, eosToken)
                                    d_14_symbolOut_ = out6_
                                    d_15_hitEos_ = out7_
                                    d_16_stepsUsed_ = out8_
                                    generated = (d_12_stablePrefix_) + (d_14_symbolOut_)
                                    currentConstrainedOut = d_14_symbolOut_
                                    d_1_steps_ = (d_1_steps_) + (d_16_stepsUsed_)
                                    if d_15_hitEos_:
                                        raise _dafny.Break("0")
                        elif True:
                            d_17_next_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_17_next_ = out9_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_17_next_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

