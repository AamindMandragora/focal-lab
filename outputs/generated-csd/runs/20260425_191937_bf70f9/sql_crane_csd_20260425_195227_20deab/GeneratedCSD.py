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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (lm).ChooseNextToken()
                        d_2_next_ = out0_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_3_complete_: bool
                        d_3_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_complete_:
                            d_4_closedGenerated_: _dafny.Seq
                            d_5_closedInside_: bool
                            d_6_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_closedGenerated_ = out1_
                            d_5_closedInside_ = out2_
                            d_6_closedCurrent_ = out3_
                            generated = d_4_closedGenerated_
                            insideConstrainedOut = d_5_closedInside_
                            currentConstrainedOut = d_6_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_validCount_: int
                            out4_: int
                            out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_7_validCount_ = out4_
                            d_8_narrow_: bool
                            out5_: bool
                            out5_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 3)
                            d_8_narrow_ = out5_
                            if (d_8_narrow_) or ((d_7_validCount_) <= (8)):
                                d_9_next2_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_9_next2_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_9_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_10_appendedGenerated1_: _dafny.Seq
                                    d_11_appendedInside1_: bool
                                    d_12_appendedCurrent1_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next2_)
                                    d_10_appendedGenerated1_ = out7_
                                    d_11_appendedInside1_ = out8_
                                    d_12_appendedCurrent1_ = out9_
                                    generated = d_10_appendedGenerated1_
                                    insideConstrainedOut = d_11_appendedInside1_
                                    currentConstrainedOut = d_12_appendedCurrent1_
                            elif True:
                                (lm).GenerateLogits((prompt) + (generated))
                                d_13_argmax_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).GetHighestLogitToken(lm)
                                d_13_argmax_ = out10_
                                d_14_argmaxValid_: bool
                                out11_: bool
                                out11_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_13_argmax_)
                                d_14_argmaxValid_ = out11_
                                if (d_14_argmaxValid_) and ((d_13_argmax_) != (eosToken)):
                                    d_15_appendedGenerated2_: _dafny.Seq
                                    d_16_appendedInside2_: bool
                                    d_17_appendedCurrent2_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_argmax_)
                                    d_15_appendedGenerated2_ = out12_
                                    d_16_appendedInside2_ = out13_
                                    d_17_appendedCurrent2_ = out14_
                                    generated = d_15_appendedGenerated2_
                                    insideConstrainedOut = d_16_appendedInside2_
                                    currentConstrainedOut = d_17_appendedCurrent2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_18_next3_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                    d_18_next3_ = out15_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_18_next3_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_19_appendedGenerated3_: _dafny.Seq
                                        d_20_appendedInside3_: bool
                                        d_21_appendedCurrent3_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: _dafny.Seq
                                        out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next3_)
                                        d_19_appendedGenerated3_ = out16_
                                        d_20_appendedInside3_ = out17_
                                        d_21_appendedCurrent3_ = out18_
                                        generated = d_19_appendedGenerated3_
                                        insideConstrainedOut = d_20_appendedInside3_
                                        currentConstrainedOut = d_21_appendedCurrent3_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

