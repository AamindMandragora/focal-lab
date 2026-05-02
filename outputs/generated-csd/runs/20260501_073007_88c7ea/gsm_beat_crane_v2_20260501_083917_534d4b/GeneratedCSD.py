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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_inheritedConstrainedSteps_: int
        d_2_inheritedConstrainedSteps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
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
                            if (d_2_inheritedConstrainedSteps_) >= (1):
                                raise _dafny.Break("0")
                            elif True:
                                d_6_steppedGenerated_: _dafny.Seq
                                d_7_steppedInside_: bool
                                d_8_steppedCurrent_: _dafny.Seq
                                d_9_hitEos_: bool
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out6_: bool
                                out3_, out4_, out5_, out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, currentConstrainedOut, eosToken)
                                d_6_steppedGenerated_ = out3_
                                d_7_steppedInside_ = out4_
                                d_8_steppedCurrent_ = out5_
                                d_9_hitEos_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_inheritedConstrainedSteps_ = (d_2_inheritedConstrainedSteps_) + (1)
                                if d_9_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_6_steppedGenerated_
                                    insideConstrainedOut = d_7_steppedInside_
                                    currentConstrainedOut = d_8_steppedCurrent_
                    elif True:
                        if ((maxSteps) - (d_1_steps_)) == (2):
                            d_10_openedGenerated_: _dafny.Seq
                            d_11_openedInside_: bool
                            d_12_openedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_openedGenerated_ = out7_
                            d_11_openedInside_ = out8_
                            d_12_openedCurrent_ = out9_
                            generated = d_10_openedGenerated_
                            insideConstrainedOut = d_11_openedInside_
                            currentConstrainedOut = d_12_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_next_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_next_ = out10_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

