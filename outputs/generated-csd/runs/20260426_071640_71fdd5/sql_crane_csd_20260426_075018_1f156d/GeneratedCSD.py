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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        insideConstrainedOut: bool = False
        currentConstrainedOut: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        if True:
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
                            d_5_completeNow_: bool
                            d_5_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_5_completeNow_:
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
                                d_9_validCount_: int
                                out6_: int
                                out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_9_validCount_ = out6_
                                d_10_narrow_: bool
                                out7_: bool
                                out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 3)
                                d_10_narrow_ = out7_
                                (lm).GenerateLogits((prompt) + (generated))
                                if d_10_narrow_:
                                    d_11_candsNarrow_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 6, eosToken)
                                    d_11_candsNarrow_ = out8_
                                    (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('7e-1'))
                                    (d_0_helpers_).BoostTokenLogits(lm, d_11_candsNarrow_, _dafny.BigRational('8e0'))
                                elif True:
                                    if (d_9_validCount_) <= (8):
                                        d_12_candsSmall_: _dafny.Seq
                                        out9_: _dafny.Seq
                                        out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 5, eosToken)
                                        d_12_candsSmall_ = out9_
                                        (d_0_helpers_).BoostTokenLogits(lm, d_12_candsSmall_, _dafny.BigRational('3e0'))
                                d_13_next_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_13_next_ = out10_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_13_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_appendedGenerated_: _dafny.Seq
                                    d_15_appendedInside_: bool
                                    d_16_appendedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                    d_14_appendedGenerated_ = out11_
                                    d_15_appendedInside_ = out12_
                                    d_16_appendedCurrent_ = out13_
                                    generated = d_14_appendedGenerated_
                                    insideConstrainedOut = d_15_appendedInside_
                                    currentConstrainedOut = d_16_appendedCurrent_
                        pass
                pass
            cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

