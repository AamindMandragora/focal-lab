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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_chunkSize_: int
        d_2_chunkSize_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_3_completeNow_: bool
                        d_3_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_completeNow_:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                d_4_gClose_: _dafny.Seq
                                d_5_iClose_: bool
                                d_6_cClose_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_4_gClose_ = out0_
                                d_5_iClose_ = out1_
                                d_6_cClose_ = out2_
                                generated = d_4_gClose_
                                insideConstrainedOut = d_5_iClose_
                                currentConstrainedOut = d_6_cClose_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                d_7_gStep_: _dafny.Seq
                                d_8_iStep_: bool
                                d_9_cStep_: _dafny.Seq
                                d_10_hitEos_: bool
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out6_: bool
                                out3_, out4_, out5_, out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, currentConstrainedOut, eosToken)
                                d_7_gStep_ = out3_
                                d_8_iStep_ = out4_
                                d_9_cStep_ = out5_
                                d_10_hitEos_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_10_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_7_gStep_
                                    insideConstrainedOut = d_8_iStep_
                                    currentConstrainedOut = d_9_cStep_
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        d_11_remaining_: int
                        d_11_remaining_ = (maxSteps) - (d_1_steps_)
                        d_12_shouldLateOpen_: bool
                        d_12_shouldLateOpen_ = False
                        if (((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)) and ((d_11_remaining_) <= (3))) and ((d_11_remaining_) >= (2)):
                            d_12_shouldLateOpen_ = True
                        if d_12_shouldLateOpen_:
                            d_13_gOpen_: _dafny.Seq
                            d_14_iOpen_: bool
                            d_15_cOpen_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_13_gOpen_ = out7_
                            d_14_iOpen_ = out8_
                            d_15_cOpen_ = out9_
                            generated = d_13_gOpen_
                            insideConstrainedOut = d_14_iOpen_
                            currentConstrainedOut = d_15_cOpen_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_16_maxChunk_: int
                            d_16_maxChunk_ = d_2_chunkSize_
                            if (d_11_remaining_) < (d_16_maxChunk_):
                                d_16_maxChunk_ = d_11_remaining_
                            d_17_gChunk_: _dafny.Seq
                            d_18_stoppedOnOpenSpan_: bool
                            d_19_stoppedOnEos_: bool
                            d_20_stepsUsed_: int
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: bool
                            out13_: int
                            out10_, out11_, out12_, out13_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_16_maxChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_17_gChunk_ = out10_
                            d_18_stoppedOnOpenSpan_ = out11_
                            d_19_stoppedOnEos_ = out12_
                            d_20_stepsUsed_ = out13_
                            generated = d_17_gChunk_
                            d_1_steps_ = (d_1_steps_) + (d_20_stepsUsed_)
                            if d_19_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif True:
                                if d_18_stoppedOnOpenSpan_:
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

