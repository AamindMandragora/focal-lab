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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_2_completeNow_: bool
                        d_2_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_2_completeNow_:
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
                            d_6_narrow_: bool
                            out3_: bool
                            out3_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_6_narrow_ = out3_
                            if d_6_narrow_:
                                d_7_stablePrefix_: _dafny.Seq
                                d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_8_repairedGenerated_: _dafny.Seq
                                d_9_repairedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: _dafny.Seq
                                out4_, out5_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_7_stablePrefix_, generated, currentConstrainedOut)
                                d_8_repairedGenerated_ = out4_
                                d_9_repairedCurrent_ = out5_
                                generated = d_8_repairedGenerated_
                                currentConstrainedOut = d_9_repairedCurrent_
                                insideConstrainedOut = True
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_1_steps_) < (maxSteps):
                                    d_10_repairedComplete_: bool
                                    d_10_repairedComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if d_10_repairedComplete_:
                                        d_11_closedGenerated2_: _dafny.Seq
                                        d_12_closedInside2_: bool
                                        d_13_closedCurrent2_: _dafny.Seq
                                        out6_: _dafny.Seq
                                        out7_: bool
                                        out8_: _dafny.Seq
                                        out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_11_closedGenerated2_ = out6_
                                        d_12_closedInside2_ = out7_
                                        d_13_closedCurrent2_ = out8_
                                        generated = d_11_closedGenerated2_
                                        insideConstrainedOut = d_12_closedInside2_
                                        currentConstrainedOut = d_13_closedCurrent2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_14_next_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_14_next_ = out9_
                                if (d_14_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_15_valid_: bool
                                    out10_: bool
                                    out10_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_14_next_)
                                    d_15_valid_ = out10_
                                    if d_15_valid_:
                                        d_16_appendedGenerated_: _dafny.Seq
                                        d_17_appendedInside_: bool
                                        d_18_appendedCurrent_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: bool
                                        out13_: _dafny.Seq
                                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                        d_16_appendedGenerated_ = out11_
                                        d_17_appendedInside_ = out12_
                                        d_18_appendedCurrent_ = out13_
                                        generated = d_16_appendedGenerated_
                                        insideConstrainedOut = d_17_appendedInside_
                                        currentConstrainedOut = d_18_appendedCurrent_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        d_19_stablePrefix2_: _dafny.Seq
                                        d_19_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                        d_20_repairedGenerated2_: _dafny.Seq
                                        d_21_repairedCurrent2_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out14_, out15_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_19_stablePrefix2_, generated, currentConstrainedOut)
                                        d_20_repairedGenerated2_ = out14_
                                        d_21_repairedCurrent2_ = out15_
                                        generated = d_20_repairedGenerated2_
                                        currentConstrainedOut = d_21_repairedCurrent2_
                                        insideConstrainedOut = True
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if (d_1_steps_) < (maxSteps):
                                            d_22_repairedComplete2_: bool
                                            d_22_repairedComplete2_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                            if d_22_repairedComplete2_:
                                                d_23_closedGenerated3_: _dafny.Seq
                                                d_24_closedInside3_: bool
                                                d_25_closedCurrent3_: _dafny.Seq
                                                out16_: _dafny.Seq
                                                out17_: bool
                                                out18_: _dafny.Seq
                                                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_23_closedGenerated3_ = out16_
                                                d_24_closedInside3_ = out17_
                                                d_25_closedCurrent3_ = out18_
                                                generated = d_23_closedGenerated3_
                                                insideConstrainedOut = d_24_closedInside3_
                                                currentConstrainedOut = d_25_closedCurrent3_
                                                d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_26_next2_: _dafny.Seq
                        out19_: _dafny.Seq
                        out19_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_26_next2_ = out19_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_26_next2_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_26_next2_]))
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        cost = d_0_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost

