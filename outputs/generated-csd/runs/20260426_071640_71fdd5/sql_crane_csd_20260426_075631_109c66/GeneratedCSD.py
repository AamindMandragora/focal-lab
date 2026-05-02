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
                            d_6_validCount_: int
                            out3_: int
                            out3_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_6_validCount_ = out3_
                            d_7_narrow_: bool
                            out4_: bool
                            out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 4)
                            d_7_narrow_ = out4_
                            if (d_5_completeNow_) and ((d_6_validCount_) <= (2)):
                                d_8_closedGenerated_: _dafny.Seq
                                d_9_closedInside_: bool
                                d_10_closedCurrent_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_8_closedGenerated_ = out5_
                                d_9_closedInside_ = out6_
                                d_10_closedCurrent_ = out7_
                                generated = d_8_closedGenerated_
                                insideConstrainedOut = d_9_closedInside_
                                currentConstrainedOut = d_10_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                (lm).GenerateLogits((prompt) + (generated))
                                if d_7_narrow_:
                                    d_11_candsNarrow_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 8, eosToken)
                                    d_11_candsNarrow_ = out8_
                                    (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('6e-1'))
                                    (d_0_helpers_).BoostTokenLogits(lm, d_11_candsNarrow_, _dafny.BigRational('1e1'))
                                elif True:
                                    if (d_6_validCount_) <= (6):
                                        d_12_candsSmall_: _dafny.Seq
                                        out9_: _dafny.Seq
                                        out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 6, eosToken)
                                        d_12_candsSmall_ = out9_
                                        (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('8e-1'))
                                        (d_0_helpers_).BoostTokenLogits(lm, d_12_candsSmall_, _dafny.BigRational('6e0'))
                                    elif True:
                                        d_13_candsWide_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out10_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 4, eosToken)
                                        d_13_candsWide_ = out10_
                                        (d_0_helpers_).BoostTokenLogits(lm, d_13_candsWide_, _dafny.BigRational('25e-1'))
                                d_14_next_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_14_next_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    if not(d_5_completeNow_):
                                        d_15_appendedGenerated_: _dafny.Seq
                                        d_16_appendedInside_: bool
                                        d_17_appendedCurrent_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: _dafny.Seq
                                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                        d_15_appendedGenerated_ = out12_
                                        d_16_appendedInside_ = out13_
                                        d_17_appendedCurrent_ = out14_
                                        generated = d_15_appendedGenerated_
                                        insideConstrainedOut = d_16_appendedInside_
                                        currentConstrainedOut = d_17_appendedCurrent_
                                    elif True:
                                        raise _dafny.Break("0")
                        pass
                pass
            cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

