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
        d_2_sawSpan_: bool
        d_2_sawSpan_ = insideConstrained
        d_3_maxConstrainedLen_: int
        d_3_maxConstrainedLen_ = 6
        d_4_unconstrainedTailBudget_: int
        d_4_unconstrainedTailBudget_ = 4
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
                            d_2_sawSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (len(currentConstrainedOut)) >= (d_3_maxConstrainedLen_):
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
                            d_10_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_10_next_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_11_appendedGenerated_: _dafny.Seq
                                d_12_appendedInside_: bool
                                d_13_appendedCurrent_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                d_11_appendedGenerated_ = out5_
                                d_12_appendedInside_ = out6_
                                d_13_appendedCurrent_ = out7_
                                generated = d_11_appendedGenerated_
                                insideConstrainedOut = d_12_appendedInside_
                                currentConstrainedOut = d_13_appendedCurrent_
                    elif True:
                        if not(d_2_sawSpan_):
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
                            d_2_sawSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif ((d_1_steps_) < (maxSteps)) and ((d_1_steps_) < (((d_4_unconstrainedTailBudget_) + (len(generated))) - (len(generatedPrefix)))):
                            d_17_nextTail_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_17_nextTail_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_nextTail_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_17_nextTail_]))
                        elif True:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

