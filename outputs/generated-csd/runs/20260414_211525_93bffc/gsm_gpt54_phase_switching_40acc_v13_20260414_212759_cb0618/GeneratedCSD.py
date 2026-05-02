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
                        d_2_isComplete_: bool
                        d_2_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_2_isComplete_:
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
                            if ((d_1_steps_) + (1)) == (maxSteps):
                                raise _dafny.Break("0")
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
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    insideConstrainedOut = True
                                    d_10_repairedComplete_: bool
                                    d_10_repairedComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if (d_10_repairedComplete_) and ((d_1_steps_) < (maxSteps)):
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
                                        d_15_completeBeforeClose_: bool
                                        d_15_completeBeforeClose_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                        if (d_15_completeBeforeClose_) and (((d_1_steps_) + (1)) < (maxSteps)):
                                            d_16_closedGenerated3_: _dafny.Seq
                                            d_17_closedInside3_: bool
                                            d_18_closedCurrent3_: _dafny.Seq
                                            out10_: _dafny.Seq
                                            out11_: bool
                                            out12_: _dafny.Seq
                                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_16_closedGenerated3_ = out10_
                                            d_17_closedInside3_ = out11_
                                            d_18_closedCurrent3_ = out12_
                                            generated = d_16_closedGenerated3_
                                            insideConstrainedOut = d_17_closedInside3_
                                            currentConstrainedOut = d_18_closedCurrent3_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                        elif True:
                                            raise _dafny.Break("0")
                                    elif True:
                                        d_19_valid_: bool
                                        out13_: bool
                                        out13_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_14_next_)
                                        d_19_valid_ = out13_
                                        if d_19_valid_:
                                            d_20_appendedGenerated_: _dafny.Seq
                                            d_21_appendedInside_: bool
                                            d_22_appendedCurrent_: _dafny.Seq
                                            out14_: _dafny.Seq
                                            out15_: bool
                                            out16_: _dafny.Seq
                                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                            d_20_appendedGenerated_ = out14_
                                            d_21_appendedInside_ = out15_
                                            d_22_appendedCurrent_ = out16_
                                            generated = d_20_appendedGenerated_
                                            insideConstrainedOut = d_21_appendedInside_
                                            currentConstrainedOut = d_22_appendedCurrent_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                        elif True:
                                            d_23_stablePrefix2_: _dafny.Seq
                                            d_23_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                            d_24_repairedGenerated2_: _dafny.Seq
                                            d_25_repairedCurrent2_: _dafny.Seq
                                            out17_: _dafny.Seq
                                            out18_: _dafny.Seq
                                            out17_, out18_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_23_stablePrefix2_, generated, currentConstrainedOut)
                                            d_24_repairedGenerated2_ = out17_
                                            d_25_repairedCurrent2_ = out18_
                                            generated = d_24_repairedGenerated2_
                                            currentConstrainedOut = d_25_repairedCurrent2_
                                            insideConstrainedOut = True
                                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_26_produced_: int
                        d_26_produced_ = (len(generated)) - (len(generatedPrefix))
                        if ((d_26_produced_) >= (8)) and (((d_1_steps_) + (2)) < (maxSteps)):
                            d_27_openedGenerated_: _dafny.Seq
                            d_28_openedInside_: bool
                            d_29_openedCurrent_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_27_openedGenerated_ = out19_
                            d_28_openedInside_ = out20_
                            d_29_openedCurrent_ = out21_
                            generated = d_27_openedGenerated_
                            insideConstrainedOut = d_28_openedInside_
                            currentConstrainedOut = d_29_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_30_next2_: _dafny.Seq
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_30_next2_ = out22_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_30_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_30_next2_]))
                    pass
            pass
        cost = d_0_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost

