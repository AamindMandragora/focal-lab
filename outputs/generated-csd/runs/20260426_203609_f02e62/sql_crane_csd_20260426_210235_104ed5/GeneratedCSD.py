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
                            if VerifiedDecoderAgent.default__.Contains(d_2_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_3_narrow_: bool
                        out1_: bool
                        out1_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_3_narrow_ = out1_
                        if not(d_3_narrow_):
                            d_4_rolled_: _dafny.Seq
                            out2_: _dafny.Seq
                            out2_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                            d_4_rolled_ = out2_
                            d_5_stablePrefix_: _dafny.Seq
                            d_5_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_6_rolledGenerated_: _dafny.Seq
                            d_7_rolledCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: _dafny.Seq
                            out3_, out4_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_5_stablePrefix_, generated, currentConstrainedOut)
                            d_6_rolledGenerated_ = out3_
                            d_7_rolledCurrent_ = out4_
                            generated = d_6_rolledGenerated_
                            currentConstrainedOut = d_7_rolledCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_isComplete_: bool
                            d_8_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_8_isComplete_:
                                d_9_validCount_: int
                                out5_: int
                                out5_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_9_validCount_ = out5_
                                if (d_9_validCount_) == (0):
                                    d_10_closedGenerated_: _dafny.Seq
                                    d_11_closedInside_: bool
                                    d_12_closedCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_10_closedGenerated_ = out6_
                                    d_11_closedInside_ = out7_
                                    d_12_closedCurrent_ = out8_
                                    generated = d_10_closedGenerated_
                                    insideConstrainedOut = d_11_closedInside_
                                    currentConstrainedOut = d_12_closedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_13_constrainedPrompt_: _dafny.Seq
                                    d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_14_next2_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_14_next2_ = out9_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_14_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_15_appendedGenerated2_: _dafny.Seq
                                        d_16_appendedInside2_: bool
                                        d_17_appendedCurrent2_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next2_)
                                        d_15_appendedGenerated2_ = out10_
                                        d_16_appendedInside2_ = out11_
                                        d_17_appendedCurrent2_ = out12_
                                        generated = d_15_appendedGenerated2_
                                        insideConstrainedOut = d_16_appendedInside2_
                                        currentConstrainedOut = d_17_appendedCurrent2_
                            elif True:
                                d_18_constrainedPrompt2_: _dafny.Seq
                                d_18_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_19_next3_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt2_, currentConstrainedOut, eosToken)
                                d_19_next3_ = out13_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_19_next3_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_20_appendedGenerated3_: _dafny.Seq
                                    d_21_appendedInside3_: bool
                                    d_22_appendedCurrent3_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next3_)
                                    d_20_appendedGenerated3_ = out14_
                                    d_21_appendedInside3_ = out15_
                                    d_22_appendedCurrent3_ = out16_
                                    generated = d_20_appendedGenerated3_
                                    insideConstrainedOut = d_21_appendedInside3_
                                    currentConstrainedOut = d_22_appendedCurrent3_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

