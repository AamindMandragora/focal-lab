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
        d_2_openedAnySpan_: bool
        d_2_openedAnySpan_ = insideConstrained
        d_3_unconstrainedLeadSteps_: int
        d_3_unconstrainedLeadSteps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_4_completeNow_: bool
                        d_4_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_completeNow_:
                            d_5_closedGenerated_: _dafny.Seq
                            d_6_closedInside_: bool
                            d_7_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_closedGenerated_ = out0_
                            d_6_closedInside_ = out1_
                            d_7_closedCurrent_ = out2_
                            generated = d_5_closedGenerated_
                            insideConstrainedOut = d_6_closedInside_
                            currentConstrainedOut = d_7_closedCurrent_
                            d_2_openedAnySpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if ((d_1_steps_) + (1)) >= (maxSteps):
                                raise _dafny.Break("0")
                            elif True:
                                d_8_next_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_8_next_ = out3_
                                if (d_8_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_9_appendedGenerated_: _dafny.Seq
                                    d_10_appendedInside_: bool
                                    d_11_appendedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                                    d_9_appendedGenerated_ = out4_
                                    d_10_appendedInside_ = out5_
                                    d_11_appendedCurrent_ = out6_
                                    generated = d_9_appendedGenerated_
                                    insideConstrainedOut = d_10_appendedInside_
                                    currentConstrainedOut = d_11_appendedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        if ((not(d_2_openedAnySpan_)) and ((d_3_unconstrainedLeadSteps_) >= (2))) and (((d_1_steps_) + (1)) < (maxSteps)):
                            d_12_openedGenerated_: _dafny.Seq
                            d_13_openedInside_: bool
                            d_14_openedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_12_openedGenerated_ = out7_
                            d_13_openedInside_ = out8_
                            d_14_openedCurrent_ = out9_
                            generated = d_12_openedGenerated_
                            insideConstrainedOut = d_13_openedInside_
                            currentConstrainedOut = d_14_openedCurrent_
                            d_2_openedAnySpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_15_next2_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_15_next2_ = out10_
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_15_next2_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_unconstrainedLeadSteps_ = (d_3_unconstrainedLeadSteps_) + (1)
                            if (d_15_next2_) == (eosToken):
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

