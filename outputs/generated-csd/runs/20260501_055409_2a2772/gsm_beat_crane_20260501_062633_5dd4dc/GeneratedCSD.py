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
        d_2_chunkSize_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_3_completeNow_: bool
                        d_3_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_3_completeNow_) and (((d_1_steps_) + (1)) <= (maxSteps)):
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
                            d_7_narrow_: bool
                            out3_: bool
                            out3_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_7_narrow_ = out3_
                            if d_7_narrow_:
                                d_8_repaired_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_8_repaired_ = out4_
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_8_repaired_))):])
                                currentConstrainedOut = d_8_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_9_gStep_: _dafny.Seq
                                d_10_iStep_: bool
                                d_11_cStep_: _dafny.Seq
                                d_12_hitEos_: bool
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out8_: bool
                                out5_, out6_, out7_, out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, currentConstrainedOut, eosToken)
                                d_9_gStep_ = out5_
                                d_10_iStep_ = out6_
                                d_11_cStep_ = out7_
                                d_12_hitEos_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_12_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_9_gStep_
                                    insideConstrainedOut = d_10_iStep_
                                    currentConstrainedOut = d_11_cStep_
                    elif True:
                        d_13_doFallbackOpen_: bool
                        d_13_doFallbackOpen_ = False
                        if (((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)) and (((d_1_steps_) + (2)) <= (maxSteps))) and ((generated) == (generatedPrefix)):
                            d_13_doFallbackOpen_ = True
                        if d_13_doFallbackOpen_:
                            d_14_gOpen_: _dafny.Seq
                            d_15_iOpen_: bool
                            d_16_cOpen_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_14_gOpen_ = out9_
                            d_15_iOpen_ = out10_
                            d_16_cOpen_ = out11_
                            generated = d_14_gOpen_
                            insideConstrainedOut = d_15_iOpen_
                            currentConstrainedOut = d_16_cOpen_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_17_remaining_: int
                            d_17_remaining_ = (maxSteps) - (d_1_steps_)
                            d_18_maxChunk_: int
                            d_18_maxChunk_ = d_2_chunkSize_
                            if (d_17_remaining_) < (d_18_maxChunk_):
                                d_18_maxChunk_ = d_17_remaining_
                            d_19_gChunk_: _dafny.Seq
                            d_20_stoppedOnOpenSpan_: bool
                            d_21_stoppedOnEos_: bool
                            d_22_stepsUsed_: int
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: bool
                            out15_: int
                            out12_, out13_, out14_, out15_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_18_maxChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_19_gChunk_ = out12_
                            d_20_stoppedOnOpenSpan_ = out13_
                            d_21_stoppedOnEos_ = out14_
                            d_22_stepsUsed_ = out15_
                            generated = d_19_gChunk_
                            d_1_steps_ = (d_1_steps_) + (d_22_stepsUsed_)
                            if d_21_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif True:
                                if d_20_stoppedOnOpenSpan_:
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

