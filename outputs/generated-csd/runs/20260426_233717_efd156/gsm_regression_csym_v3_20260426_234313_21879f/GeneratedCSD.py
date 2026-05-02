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
                        if ((d_1_steps_) + (2)) < (maxSteps):
                            (lm).GenerateLogits((prompt) + (generated))
                            d_2_top_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).GetHighestLogitToken(lm)
                            d_2_top_ = out0_
                            if (d_2_top_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_3_openedGenerated_: _dafny.Seq
                                d_4_openedInside_: bool
                                d_5_openedCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_3_openedGenerated_ = out1_
                                d_4_openedInside_ = out2_
                                d_5_openedCurrent_ = out3_
                                generated = d_3_openedGenerated_
                                insideConstrainedOut = d_4_openedInside_
                                currentConstrainedOut = d_5_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_6_next_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_6_next_ = out4_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_6_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                    if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                        insideConstrainedOut = True
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_7_next2_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next2_ = out5_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next2_]))
                                if (d_7_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_8_complete_: bool
                        d_8_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_complete_:
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out6_
                            d_10_closedInside_ = out7_
                            d_11_closedCurrent_ = out8_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_narrow_: bool
                            out9_: bool
                            out9_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_12_narrow_ = out9_
                            if d_12_narrow_:
                                d_13_rolled_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_13_rolled_ = out10_
                                d_14_stablePrefix_: _dafny.Seq
                                d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_15_rolledGenerated_: _dafny.Seq
                                d_16_rolledCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: _dafny.Seq
                                out11_, out12_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_14_stablePrefix_, generated, currentConstrainedOut)
                                d_15_rolledGenerated_ = out11_
                                d_16_rolledCurrent_ = out12_
                                generated = d_15_rolledGenerated_
                                currentConstrainedOut = d_16_rolledCurrent_
                                if (len(d_13_rolled_)) < (len(currentConstrainedOut)):
                                    currentConstrainedOut = d_13_rolled_
                                    generated = (d_14_stablePrefix_) + (d_13_rolled_)
                                d_17_completeAfterRollback_: bool
                                d_17_completeAfterRollback_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if (d_17_completeAfterRollback_) and ((d_1_steps_) < (maxSteps)):
                                    d_18_closedGenerated2_: _dafny.Seq
                                    d_19_closedInside2_: bool
                                    d_20_closedCurrent2_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_18_closedGenerated2_ = out13_
                                    d_19_closedInside2_ = out14_
                                    d_20_closedCurrent2_ = out15_
                                    generated = d_18_closedGenerated2_
                                    insideConstrainedOut = d_19_closedInside2_
                                    currentConstrainedOut = d_20_closedCurrent2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_21_stablePrefix2_: _dafny.Seq
                                d_21_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_22_constrainedPrompt_: _dafny.Seq
                                d_22_constrainedPrompt_ = (prompt) + (d_21_stablePrefix2_)
                                d_23_next3_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_23_next3_ = out16_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_23_next3_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_24_appendedGenerated_: _dafny.Seq
                                    d_25_appendedInside_: bool
                                    d_26_appendedCurrent_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next3_)
                                    d_24_appendedGenerated_ = out17_
                                    d_25_appendedInside_ = out18_
                                    d_26_appendedCurrent_ = out19_
                                    generated = d_24_appendedGenerated_
                                    insideConstrainedOut = d_25_appendedInside_
                                    currentConstrainedOut = d_26_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

