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
                        d_2_chunkLimit_: int
                        d_2_chunkLimit_ = (maxSteps) - (d_1_steps_)
                        if (d_2_chunkLimit_) > (8):
                            d_2_chunkLimit_ = 8
                        d_3_chunkGenerated_: _dafny.Seq
                        d_4_stoppedOnOpenSpan_: bool
                        d_5_stoppedOnEos_: bool
                        d_6_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkLimit_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_3_chunkGenerated_ = out0_
                        d_4_stoppedOnOpenSpan_ = out1_
                        d_5_stoppedOnEos_ = out2_
                        d_6_stepsUsed_ = out3_
                        generated = d_3_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                        if d_5_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_4_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_7_closedGenerated_: _dafny.Seq
                            d_8_closedInside_: bool
                            d_9_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_closedGenerated_ = out4_
                            d_8_closedInside_ = out5_
                            d_9_closedCurrent_ = out6_
                            generated = d_7_closedGenerated_
                            insideConstrainedOut = d_8_closedInside_
                            currentConstrainedOut = d_9_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_stablePrefix_: _dafny.Seq
                            d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                            d_12_narrow_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_12_narrow_ = out7_
                            d_13_validCount_: int
                            out8_: int
                            out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_13_validCount_ = out8_
                            if (d_12_narrow_) or ((d_13_validCount_) <= (2)):
                                d_14_next_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_14_next_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
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
                                    elif True:
                                        d_19_repairedGenerated_: _dafny.Seq
                                        d_20_repairedCurrent_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out14_, out15_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_10_stablePrefix_, generated, currentConstrainedOut)
                                        d_19_repairedGenerated_ = out14_
                                        d_20_repairedCurrent_ = out15_
                                        generated = d_19_repairedGenerated_
                                        currentConstrainedOut = d_20_repairedCurrent_
                                        if (parser).IsValidPrefix(currentConstrainedOut):
                                            insideConstrainedOut = True
                                        elif True:
                                            insideConstrainedOut = False
                                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                d_21_next_: _dafny.Seq
                                d_22_isValid_: bool
                                out16_: _dafny.Seq
                                out17_: bool
                                out16_, out17_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('3e0'), eosToken)
                                d_21_next_ = out16_
                                d_22_isValid_ = out17_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    if d_22_isValid_:
                                        d_23_appendedGenerated_: _dafny.Seq
                                        d_24_appendedInside_: bool
                                        d_25_appendedCurrent_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                        d_23_appendedGenerated_ = out18_
                                        d_24_appendedInside_ = out19_
                                        d_25_appendedCurrent_ = out20_
                                        generated = d_23_appendedGenerated_
                                        insideConstrainedOut = d_24_appendedInside_
                                        currentConstrainedOut = d_25_appendedCurrent_
                                    elif True:
                                        d_26_repairedGenerated2_: _dafny.Seq
                                        d_27_repairedCurrent2_: _dafny.Seq
                                        out21_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out21_, out22_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_10_stablePrefix_, generated, currentConstrainedOut)
                                        d_26_repairedGenerated2_ = out21_
                                        d_27_repairedCurrent2_ = out22_
                                        generated = d_26_repairedGenerated2_
                                        currentConstrainedOut = d_27_repairedCurrent2_
                                        if (parser).IsValidPrefix(currentConstrainedOut):
                                            insideConstrainedOut = True
                                        elif True:
                                            insideConstrainedOut = False
                                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

