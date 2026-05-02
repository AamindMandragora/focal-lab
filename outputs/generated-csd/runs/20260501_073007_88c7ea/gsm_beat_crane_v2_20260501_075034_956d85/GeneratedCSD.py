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
        d_2_sawAnySpan_: bool
        d_2_sawAnySpan_ = insideConstrained
        d_3_lateOpenThreshold_: int
        d_3_lateOpenThreshold_ = 0
        if (maxSteps) > (3):
            d_3_lateOpenThreshold_ = (maxSteps) - (3)
        elif True:
            d_3_lateOpenThreshold_ = maxSteps
        d_4_maxConstrainedLen_: int
        d_4_maxConstrainedLen_ = 8
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
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (len(currentConstrainedOut)) >= (d_4_maxConstrainedLen_):
                            d_9_repaired_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                            d_9_repaired_ = out3_
                            generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_9_repaired_))):])
                            currentConstrainedOut = d_9_repaired_
                            if (len(currentConstrainedOut)) == (0):
                                insideConstrainedOut = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_steppedGenerated_: _dafny.Seq
                            d_11_steppedInside_: bool
                            d_12_steppedCurrent_: _dafny.Seq
                            d_13_hitEos_: bool
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out7_: bool
                            out4_, out5_, out6_, out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, currentConstrainedOut, eosToken)
                            d_10_steppedGenerated_ = out4_
                            d_11_steppedInside_ = out5_
                            d_12_steppedCurrent_ = out6_
                            d_13_hitEos_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_13_hitEos_:
                                raise _dafny.Break("0")
                            elif True:
                                generated = d_10_steppedGenerated_
                                insideConstrainedOut = d_11_steppedInside_
                                currentConstrainedOut = d_12_steppedCurrent_
                    elif True:
                        if ((not(d_2_sawAnySpan_)) and ((d_1_steps_) >= (d_3_lateOpenThreshold_))) and (((d_1_steps_) + (1)) < (maxSteps)):
                            d_14_openedGenerated_: _dafny.Seq
                            d_15_openedInside_: bool
                            d_16_openedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_14_openedGenerated_ = out8_
                            d_15_openedInside_ = out9_
                            d_16_openedCurrent_ = out10_
                            generated = d_14_openedGenerated_
                            insideConstrainedOut = d_15_openedInside_
                            currentConstrainedOut = d_16_openedCurrent_
                            d_2_sawAnySpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_17_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_17_next_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_17_next_]))
                                if VerifiedDecoderAgent.default__.Contains(d_17_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_sawAnySpan_ = True
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

