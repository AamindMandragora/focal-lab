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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_hasOpened_: bool
        d_2_hasOpened_ = (insideConstrained) or ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in (generatedPrefix))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_mustOpenSoon_: bool
                        d_3_mustOpenSoon_ = (not(d_2_hasOpened_)) and (((d_1_steps_) + (2)) >= (maxSteps))
                        if d_3_mustOpenSoon_:
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out0_
                            d_5_openedInside_ = out1_
                            d_6_openedCurrent_ = out2_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_2_hasOpened_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                            cost = d_0_helpers_.cost
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            if d_2_hasOpened_:
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                            elif True:
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (lm).ChooseNextTokenUnconstrained()
                            d_7_next_ = out3_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                if (not(d_2_hasOpened_)) and ((d_1_steps_) < (maxSteps)):
                                    d_8_openedGenerated2_: _dafny.Seq
                                    d_9_openedInside2_: bool
                                    d_10_openedCurrent2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_8_openedGenerated2_ = out4_
                                    d_9_openedInside2_ = out5_
                                    d_10_openedCurrent2_ = out6_
                                    generated = d_8_openedGenerated2_
                                    insideConstrainedOut = d_9_openedInside2_
                                    currentConstrainedOut = d_10_openedCurrent2_
                                    d_2_hasOpened_ = True
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    cost = d_0_helpers_.cost
                                elif True:
                                    cost = d_0_helpers_.cost
                                    raise _dafny.Break("0")
                            elif True:
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    if d_2_hasOpened_:
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                    elif True:
                                        d_11_openedGenerated3_: _dafny.Seq
                                        d_12_openedInside3_: bool
                                        d_13_openedCurrent3_: _dafny.Seq
                                        out7_: _dafny.Seq
                                        out8_: bool
                                        out9_: _dafny.Seq
                                        out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                        d_11_openedGenerated3_ = out7_
                                        d_12_openedInside3_ = out8_
                                        d_13_openedCurrent3_ = out9_
                                        generated = d_11_openedGenerated3_
                                        insideConstrainedOut = d_12_openedInside3_
                                        currentConstrainedOut = d_13_openedCurrent3_
                                        d_2_hasOpened_ = True
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                cost = d_0_helpers_.cost
                    elif True:
                        d_14_isComplete_: bool
                        d_14_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_14_isComplete_:
                            d_15_closedGenerated_: _dafny.Seq
                            d_16_closedInside_: bool
                            d_17_closedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_15_closedGenerated_ = out10_
                            d_16_closedInside_ = out11_
                            d_17_closedCurrent_ = out12_
                            generated = d_15_closedGenerated_
                            insideConstrainedOut = d_16_closedInside_
                            currentConstrainedOut = d_17_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            cost = d_0_helpers_.cost
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                            d_18_nextIn_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_18_nextIn_ = out13_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_nextIn_) == (eosToken):
                                cost = d_0_helpers_.cost
                                raise _dafny.Break("0")
                            elif True:
                                d_19_appendedGenerated_: _dafny.Seq
                                d_20_appendedInside_: bool
                                d_21_appendedCurrent_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_nextIn_)
                                d_19_appendedGenerated_ = out14_
                                d_20_appendedInside_ = out15_
                                d_21_appendedCurrent_ = out16_
                                generated = d_19_appendedGenerated_
                                insideConstrainedOut = d_20_appendedInside_
                                currentConstrainedOut = d_21_appendedCurrent_
                                cost = d_0_helpers_.cost
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

