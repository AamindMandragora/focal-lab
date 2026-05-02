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
        d_2_blocked_: _dafny.Seq
        d_2_blocked_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))])
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
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_narrow_: bool
                            out3_: bool
                            out3_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_7_narrow_ = out3_
                            if d_7_narrow_:
                                d_8_stablePrefix_: _dafny.Seq
                                d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_9_repairedGenerated_: _dafny.Seq
                                d_10_repairedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: _dafny.Seq
                                out4_, out5_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_8_stablePrefix_, generated, currentConstrainedOut)
                                d_9_repairedGenerated_ = out4_
                                d_10_repairedCurrent_ = out5_
                                generated = d_9_repairedGenerated_
                                currentConstrainedOut = d_10_repairedCurrent_
                                insideConstrainedOut = True
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_1_steps_) < (maxSteps):
                                    d_11_repairedComplete_: bool
                                    d_11_repairedComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if d_11_repairedComplete_:
                                        d_12_closedGenerated2_: _dafny.Seq
                                        d_13_closedInside2_: bool
                                        d_14_closedCurrent2_: _dafny.Seq
                                        out6_: _dafny.Seq
                                        out7_: bool
                                        out8_: _dafny.Seq
                                        out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_12_closedGenerated2_ = out6_
                                        d_13_closedInside2_ = out7_
                                        d_14_closedCurrent2_ = out8_
                                        generated = d_12_closedGenerated2_
                                        insideConstrainedOut = d_13_closedInside2_
                                        currentConstrainedOut = d_14_closedCurrent2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_15_next_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, prompt, currentConstrainedOut, d_2_blocked_, _dafny.BigRational('1e2'), eosToken)
                                d_15_next_ = out9_
                                if (d_15_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_valid_: bool
                                    out10_: bool
                                    out10_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_15_next_)
                                    d_16_valid_ = out10_
                                    if d_16_valid_:
                                        d_17_appendedGenerated_: _dafny.Seq
                                        d_18_appendedInside_: bool
                                        d_19_appendedCurrent_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: bool
                                        out13_: _dafny.Seq
                                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                        d_17_appendedGenerated_ = out11_
                                        d_18_appendedInside_ = out12_
                                        d_19_appendedCurrent_ = out13_
                                        generated = d_17_appendedGenerated_
                                        insideConstrainedOut = d_18_appendedInside_
                                        currentConstrainedOut = d_19_appendedCurrent_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        d_20_stablePrefix2_: _dafny.Seq
                                        d_20_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                        d_21_repairedGenerated2_: _dafny.Seq
                                        d_22_repairedCurrent2_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out14_, out15_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_20_stablePrefix2_, generated, currentConstrainedOut)
                                        d_21_repairedGenerated2_ = out14_
                                        d_22_repairedCurrent2_ = out15_
                                        generated = d_21_repairedGenerated2_
                                        currentConstrainedOut = d_22_repairedCurrent2_
                                        insideConstrainedOut = True
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if (d_1_steps_) < (maxSteps):
                                            d_23_repairedComplete2_: bool
                                            d_23_repairedComplete2_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                            if d_23_repairedComplete2_:
                                                d_24_closedGenerated3_: _dafny.Seq
                                                d_25_closedInside3_: bool
                                                d_26_closedCurrent3_: _dafny.Seq
                                                out16_: _dafny.Seq
                                                out17_: bool
                                                out18_: _dafny.Seq
                                                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_24_closedGenerated3_ = out16_
                                                d_25_closedInside3_ = out17_
                                                d_26_closedCurrent3_ = out18_
                                                generated = d_24_closedGenerated3_
                                                insideConstrainedOut = d_25_closedInside3_
                                                currentConstrainedOut = d_26_closedCurrent3_
                                                d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_27_next2_: _dafny.Seq
                        out19_: _dafny.Seq
                        out19_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, prompt, _dafny.SeqWithoutIsStrInference([]), d_2_blocked_, _dafny.BigRational('1e2'), eosToken)
                        d_27_next2_ = out19_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_27_next2_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if (d_27_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_27_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_27_next2_]))
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        cost = d_0_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost

