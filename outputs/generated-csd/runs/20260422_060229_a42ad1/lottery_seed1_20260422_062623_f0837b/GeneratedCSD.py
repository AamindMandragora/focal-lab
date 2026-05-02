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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        (d_0_helpers_).cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_constrainedLen_: int
        d_2_constrainedLen_ = 0
        if insideConstrainedOut:
            d_2_constrainedLen_ = len(currentConstrainedOut)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_3_completeNow_: bool
                        d_3_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_completeNow_:
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
                            d_2_constrainedLen_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if ((d_1_steps_) + (1)) >= (maxSteps):
                                raise _dafny.Break("0")
                            elif True:
                                d_7_next_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_7_next_ = out3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_7_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_8_appendedGenerated_: _dafny.Seq
                                    d_9_appendedInside_: bool
                                    d_10_appendedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_next_)
                                    d_8_appendedGenerated_ = out4_
                                    d_9_appendedInside_ = out5_
                                    d_10_appendedCurrent_ = out6_
                                    generated = d_8_appendedGenerated_
                                    insideConstrainedOut = d_9_appendedInside_
                                    currentConstrainedOut = d_10_appendedCurrent_
                                    d_2_constrainedLen_ = len(currentConstrainedOut)
                                    d_11_completeAfter_: bool
                                    d_11_completeAfter_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if (d_11_completeAfter_) and ((d_1_steps_) < (maxSteps)):
                                        d_12_closedGenerated2_: _dafny.Seq
                                        d_13_closedInside2_: bool
                                        d_14_closedCurrent2_: _dafny.Seq
                                        out7_: _dafny.Seq
                                        out8_: bool
                                        out9_: _dafny.Seq
                                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_12_closedGenerated2_ = out7_
                                        d_13_closedInside2_ = out8_
                                        d_14_closedCurrent2_ = out9_
                                        generated = d_12_closedGenerated2_
                                        insideConstrainedOut = d_13_closedInside2_
                                        currentConstrainedOut = d_14_closedCurrent2_
                                        d_2_constrainedLen_ = 0
                                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        if ((d_1_steps_) + (2)) <= (maxSteps):
                            d_15_openedGenerated_: _dafny.Seq
                            d_16_openedInside_: bool
                            d_17_openedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_15_openedGenerated_ = out10_
                            d_16_openedInside_ = out11_
                            d_17_openedCurrent_ = out12_
                            generated = d_15_openedGenerated_
                            insideConstrainedOut = d_16_openedInside_
                            currentConstrainedOut = d_17_openedCurrent_
                            d_2_constrainedLen_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                            d_18_next2_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (lm).ChooseNextTokenUnconstrained()
                            d_18_next2_ = out13_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_18_next2_]))
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

