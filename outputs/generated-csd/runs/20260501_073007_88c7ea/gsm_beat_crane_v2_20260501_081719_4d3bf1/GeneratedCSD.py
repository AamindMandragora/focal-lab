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
        d_2_inheritedCap_: int
        d_2_inheritedCap_ = 3
        d_3_guaranteedSpanDone_: bool
        d_3_guaranteedSpanDone_ = insideConstrained
        d_4_tailCap_: int
        d_4_tailCap_ = 48
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_5_completeNow_: bool
                        d_5_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_completeNow_:
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out0_
                            d_7_closedInside_ = out1_
                            d_8_closedCurrent_ = out2_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_3_guaranteedSpanDone_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (len(currentConstrainedOut)) >= (d_2_inheritedCap_):
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_3_guaranteedSpanDone_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_steppedGenerated_: _dafny.Seq
                            d_10_steppedInside_: bool
                            d_11_steppedCurrent_: _dafny.Seq
                            d_12_hitEos_: bool
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out6_: bool
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, currentConstrainedOut, eosToken)
                            d_9_steppedGenerated_ = out3_
                            d_10_steppedInside_ = out4_
                            d_11_steppedCurrent_ = out5_
                            d_12_hitEos_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_12_hitEos_:
                                raise _dafny.Break("0")
                            elif True:
                                generated = d_9_steppedGenerated_
                                insideConstrainedOut = d_10_steppedInside_
                                currentConstrainedOut = d_11_steppedCurrent_
                    elif True:
                        if not(d_3_guaranteedSpanDone_):
                            d_13_openedGenerated_: _dafny.Seq
                            d_14_openedInside_: bool
                            d_15_openedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_13_openedGenerated_ = out7_
                            d_14_openedInside_ = out8_
                            d_15_openedCurrent_ = out9_
                            generated = d_13_openedGenerated_
                            insideConstrainedOut = d_14_openedInside_
                            currentConstrainedOut = d_15_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) >= (d_4_tailCap_):
                            raise _dafny.Break("0")
                        elif True:
                            d_16_next_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_16_next_ = out10_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_next_]))
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

