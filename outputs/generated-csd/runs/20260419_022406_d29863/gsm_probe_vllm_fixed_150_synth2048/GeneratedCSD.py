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
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_3_closedGenerated_: _dafny.Seq
                            d_4_closedInside_: bool
                            d_5_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_3_closedGenerated_ = out1_
                            d_4_closedInside_ = out2_
                            d_5_closedCurrent_ = out3_
                            generated = d_3_closedGenerated_
                            insideConstrainedOut = d_4_closedInside_
                            currentConstrainedOut = d_5_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_narrow_: bool
                            out4_: bool
                            out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_6_narrow_ = out4_
                            if d_6_narrow_:
                                d_7_stablePrefix_: _dafny.Seq
                                d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_8_repairedGenerated_: _dafny.Seq
                                d_9_repairedCurrent_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: _dafny.Seq
                                out5_, out6_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_7_stablePrefix_, generated, currentConstrainedOut)
                                d_8_repairedGenerated_ = out5_
                                d_9_repairedCurrent_ = out6_
                                generated = d_8_repairedGenerated_
                                currentConstrainedOut = d_9_repairedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    d_10_closedGenerated2_: _dafny.Seq
                                    d_11_closedInside2_: bool
                                    d_12_closedCurrent2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_10_closedGenerated2_ = out7_
                                    d_11_closedInside2_ = out8_
                                    d_12_closedCurrent2_ = out9_
                                    generated = d_10_closedGenerated2_
                                    insideConstrainedOut = d_11_closedInside2_
                                    currentConstrainedOut = d_12_closedCurrent2_
                                elif True:
                                    insideConstrainedOut = True
                            elif True:
                                d_13_constrainedPrompt_: _dafny.Seq
                                d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_14_next_: _dafny.Seq
                                d_15_isValid_: bool
                                out10_: _dafny.Seq
                                out11_: bool
                                out10_, out11_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('3e0'), eosToken)
                                d_14_next_ = out10_
                                d_15_isValid_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    if d_15_isValid_:
                                        d_16_appendedGenerated_: _dafny.Seq
                                        d_17_appendedInside_: bool
                                        d_18_appendedCurrent_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: _dafny.Seq
                                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                        d_16_appendedGenerated_ = out12_
                                        d_17_appendedInside_ = out13_
                                        d_18_appendedCurrent_ = out14_
                                        generated = d_16_appendedGenerated_
                                        insideConstrainedOut = d_17_appendedInside_
                                        currentConstrainedOut = d_18_appendedCurrent_
                                    elif True:
                                        d_19_stablePrefix2_: _dafny.Seq
                                        d_19_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                        d_20_repairedGenerated2_: _dafny.Seq
                                        d_21_repairedCurrent2_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out15_, out16_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_19_stablePrefix2_, generated, currentConstrainedOut)
                                        d_20_repairedGenerated2_ = out15_
                                        d_21_repairedCurrent2_ = out16_
                                        generated = d_20_repairedGenerated2_
                                        currentConstrainedOut = d_21_repairedCurrent2_
                                        insideConstrainedOut = True
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

