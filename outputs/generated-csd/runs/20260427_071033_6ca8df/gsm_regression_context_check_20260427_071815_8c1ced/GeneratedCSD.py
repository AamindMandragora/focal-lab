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
                        d_2_isComplete_: bool
                        d_2_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_2_isComplete_:
                            d_3_remainingClose_: int
                            d_3_remainingClose_ = (maxSteps) - (d_1_steps_)
                            if (d_3_remainingClose_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_4_closedGenerated_: _dafny.Seq
                                d_5_closedInside_: bool
                                d_6_closedCurrent_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_4_closedGenerated_ = out0_
                                d_5_closedInside_ = out1_
                                d_6_closedCurrent_ = out2_
                                generated = d_4_closedGenerated_
                                insideConstrainedOut = d_5_closedInside_
                                currentConstrainedOut = d_6_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_remainingConstrained_: int
                            d_7_remainingConstrained_ = (maxSteps) - (d_1_steps_)
                            if ((d_7_remainingConstrained_) == (0)) or ((stepTokenBudget) == (0)):
                                raise _dafny.Break("0")
                            elif True:
                                d_8_stablePrefix_: _dafny.Seq
                                d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_9_constrainedPrompt_: _dafny.Seq
                                d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                                d_10_symbolBudget_: int
                                d_10_symbolBudget_ = stepTokenBudget
                                if (d_7_remainingConstrained_) < (d_10_symbolBudget_):
                                    d_10_symbolBudget_ = d_7_remainingConstrained_
                                d_11_symbolOut_: _dafny.Seq
                                d_12_hitEos_: bool
                                d_13_stepsUsed_: int
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: int
                                out3_, out4_, out5_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, d_10_symbolBudget_, eosToken)
                                d_11_symbolOut_ = out3_
                                d_12_hitEos_ = out4_
                                d_13_stepsUsed_ = out5_
                                generated = (d_8_stablePrefix_) + (d_11_symbolOut_)
                                currentConstrainedOut = d_11_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                                if d_12_hitEos_:
                                    raise _dafny.Break("0")
                    elif True:
                        d_14_remainingOuter_: int
                        d_14_remainingOuter_ = (maxSteps) - (d_1_steps_)
                        if (d_14_remainingOuter_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            if (len(generated)) == (len(generatedPrefix)):
                                d_15_openedGenerated0_: _dafny.Seq
                                d_16_openedInside0_: bool
                                d_17_openedCurrent0_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_15_openedGenerated0_ = out6_
                                d_16_openedInside0_ = out7_
                                d_17_openedCurrent0_ = out8_
                                generated = d_15_openedGenerated0_
                                insideConstrainedOut = d_16_openedInside0_
                                currentConstrainedOut = d_17_openedCurrent0_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_18_next_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_18_next_ = out9_
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_18_next_]))
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_18_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_remainingAfterText_: int
                                    d_19_remainingAfterText_ = (maxSteps) - (d_1_steps_)
                                    if (d_19_remainingAfterText_) > (0):
                                        d_20_openedGenerated1_: _dafny.Seq
                                        d_21_openedInside1_: bool
                                        d_22_openedCurrent1_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                        d_20_openedGenerated1_ = out10_
                                        d_21_openedInside1_ = out11_
                                        d_22_openedCurrent1_ = out12_
                                        generated = d_20_openedGenerated1_
                                        insideConstrainedOut = d_21_openedInside1_
                                        currentConstrainedOut = d_22_openedCurrent1_
                                        d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

