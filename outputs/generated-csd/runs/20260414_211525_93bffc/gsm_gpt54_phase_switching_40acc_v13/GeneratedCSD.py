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
                    if not(insideConstrainedOut):
                        if ((len(generated)) > (len(generatedPrefix))) and (((d_1_steps_) + (1)) < (maxSteps)):
                            d_2_lastTok_: _dafny.Seq
                            d_2_lastTok_ = (generated)[(len(generated)) - (1)]
                            if (((d_2_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) or ((d_2_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))):
                                d_3_openedGenerated_: _dafny.Seq
                                d_4_openedInside_: bool
                                d_5_openedCurrent_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_3_openedGenerated_ = out0_
                                d_4_openedInside_ = out1_
                                d_5_openedCurrent_ = out2_
                                generated = d_3_openedGenerated_
                                insideConstrainedOut = d_4_openedInside_
                                currentConstrainedOut = d_5_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_6_next_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_6_next_ = out3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_6_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                        elif True:
                            d_7_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                    elif True:
                        d_8_isComplete_: bool
                        d_8_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_isComplete_:
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out5_
                            d_10_closedInside_ = out6_
                            d_11_closedCurrent_ = out7_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_narrow_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_12_narrow_ = out8_
                            if d_12_narrow_:
                                d_13_stablePrefix_: _dafny.Seq
                                d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_14_repairedGenerated_: _dafny.Seq
                                d_15_repairedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: _dafny.Seq
                                out9_, out10_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_13_stablePrefix_, generated, currentConstrainedOut)
                                d_14_repairedGenerated_ = out9_
                                d_15_repairedCurrent_ = out10_
                                generated = d_14_repairedGenerated_
                                currentConstrainedOut = d_15_repairedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                insideConstrainedOut = True
                            elif True:
                                d_16_next_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_16_next_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_16_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                d_17_valid_: bool
                                out12_: bool
                                out12_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_16_next_)
                                d_17_valid_ = out12_
                                if d_17_valid_:
                                    d_18_appendedGenerated_: _dafny.Seq
                                    d_19_appendedInside_: bool
                                    d_20_appendedCurrent_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                    d_18_appendedGenerated_ = out13_
                                    d_19_appendedInside_ = out14_
                                    d_20_appendedCurrent_ = out15_
                                    generated = d_18_appendedGenerated_
                                    insideConstrainedOut = d_19_appendedInside_
                                    currentConstrainedOut = d_20_appendedCurrent_
                                elif True:
                                    d_21_stablePrefix_: _dafny.Seq
                                    d_21_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_22_repairedGenerated_: _dafny.Seq
                                    d_23_repairedCurrent_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out16_, out17_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_21_stablePrefix_, generated, currentConstrainedOut)
                                    d_22_repairedGenerated_ = out16_
                                    d_23_repairedCurrent_ = out17_
                                    generated = d_22_repairedGenerated_
                                    currentConstrainedOut = d_23_repairedCurrent_
                                    insideConstrainedOut = True
                    pass
            pass
        cost = d_0_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost

