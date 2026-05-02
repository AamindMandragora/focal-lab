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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_canClose_: bool
        d_2_canClose_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))) in ((lm).Tokens)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkLimit_: int
                        if (d_3_remaining_) < (8):
                            d_4_chunkLimit_ = d_3_remaining_
                        elif True:
                            d_4_chunkLimit_ = 8
                        if (d_4_chunkLimit_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            d_5_chunkGenerated_: _dafny.Seq
                            d_6_stoppedOnOpenSpan_: bool
                            d_7_stoppedOnEos_: bool
                            d_8_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkLimit_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_5_chunkGenerated_ = out0_
                            d_6_stoppedOnOpenSpan_ = out1_
                            d_7_stoppedOnEos_ = out2_
                            d_8_stepsUsed_ = out3_
                            generated = d_5_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                            if (d_7_stoppedOnEos_) or (d_6_stoppedOnOpenSpan_):
                                raise _dafny.Break("0")
                    elif True:
                        d_9_completeNow_: bool
                        d_9_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_completeNow_:
                            if d_2_canClose_:
                                d_10_closedGenerated_: _dafny.Seq
                                d_11_closedInside_: bool
                                d_12_closedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_10_closedGenerated_ = out4_
                                d_11_closedInside_ = out5_
                                d_12_closedCurrent_ = out6_
                                generated = d_10_closedGenerated_
                                insideConstrainedOut = d_11_closedInside_
                                currentConstrainedOut = d_12_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_13_stablePrefix_: _dafny.Seq
                            d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_14_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_13_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_14_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_15_appendedGenerated_: _dafny.Seq
                                d_16_appendedInside_: bool
                                d_17_appendedCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                d_15_appendedGenerated_ = out8_
                                d_16_appendedInside_ = out9_
                                d_17_appendedCurrent_ = out10_
                                generated = d_15_appendedGenerated_
                                insideConstrainedOut = d_16_appendedInside_
                                currentConstrainedOut = d_17_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

