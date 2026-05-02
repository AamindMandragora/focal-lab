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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_forcedSpanOpened_: bool
        d_2_forcedSpanOpened_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_forcedSpanOpened_)) and ((d_1_steps_) == (0)):
                            d_3_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_3_next_ = out0_
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_3_next_) == (eosToken):
                                d_2_forcedSpanOpened_ = True
                                raise _dafny.Break("0")
                        elif not(d_2_forcedSpanOpened_):
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out1_
                            d_5_openedInside_ = out2_
                            d_6_openedCurrent_ = out3_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_2_forcedSpanOpened_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_next2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next2_ = out4_
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next2_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next2_) == (eosToken):
                                raise _dafny.Break("0")
                    elif True:
                        d_8_isComplete_: bool
                        d_8_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_isComplete_:
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out5_
                            d_10_closedInside_ = out6_
                            d_11_closedCurrent_ = out7_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_stablePrefix_: _dafny.Seq
                            d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_13_next3_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_12_stablePrefix_), currentConstrainedOut, eosToken)
                            d_13_next3_ = out8_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next3_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_14_appendedGenerated_: _dafny.Seq
                                d_15_appendedInside_: bool
                                d_16_appendedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next3_)
                                d_14_appendedGenerated_ = out9_
                                d_15_appendedInside_ = out10_
                                d_16_appendedCurrent_ = out11_
                                generated = d_14_appendedGenerated_
                                insideConstrainedOut = d_15_appendedInside_
                                currentConstrainedOut = d_16_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        if (maxSteps) > (0):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

