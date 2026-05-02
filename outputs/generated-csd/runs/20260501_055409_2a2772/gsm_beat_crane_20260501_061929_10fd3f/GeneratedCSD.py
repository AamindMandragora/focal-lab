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
                    if not(insideConstrainedOut):
                        d_3_canTryOpen_: bool
                        d_3_canTryOpen_ = False
                        if ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)) and ((len(generated)) > (len(generatedPrefix))):
                            d_4_prevTok_: _dafny.Seq
                            d_5_foundEq_: bool
                            out0_: _dafny.Seq
                            out1_: bool
                            out0_, out1_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                            d_4_prevTok_ = out0_
                            d_5_foundEq_ = out1_
                            if d_5_foundEq_:
                                d_3_canTryOpen_ = True
                        if (d_3_canTryOpen_) and (((d_1_steps_) + (1)) <= (maxSteps)):
                            d_6_gOpen_: _dafny.Seq
                            d_7_iOpen_: bool
                            d_8_cOpen_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_gOpen_ = out2_
                            d_7_iOpen_ = out3_
                            d_8_cOpen_ = out4_
                            generated = d_6_gOpen_
                            insideConstrainedOut = d_7_iOpen_
                            currentConstrainedOut = d_8_cOpen_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_remaining_: int
                            d_9_remaining_ = (maxSteps) - (d_1_steps_)
                            d_10_maxChunk_: int
                            d_10_maxChunk_ = d_2_chunkSize_
                            if (d_9_remaining_) < (d_10_maxChunk_):
                                d_10_maxChunk_ = d_9_remaining_
                            d_11_gChunk_: _dafny.Seq
                            d_12_stoppedOnOpenSpan_: bool
                            d_13_stoppedOnEos_: bool
                            d_14_stepsUsed_: int
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: bool
                            out8_: int
                            out5_, out6_, out7_, out8_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_maxChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_11_gChunk_ = out5_
                            d_12_stoppedOnOpenSpan_ = out6_
                            d_13_stoppedOnEos_ = out7_
                            d_14_stepsUsed_ = out8_
                            generated = d_11_gChunk_
                            d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                            if d_13_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif True:
                                if d_12_stoppedOnOpenSpan_:
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_15_completeNow_: bool
                        d_15_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_15_completeNow_:
                            d_16_gClose_: _dafny.Seq
                            d_17_iClose_: bool
                            d_18_cClose_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_gClose_ = out9_
                            d_17_iClose_ = out10_
                            d_18_cClose_ = out11_
                            generated = d_16_gClose_
                            insideConstrainedOut = d_17_iClose_
                            currentConstrainedOut = d_18_cClose_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_19_narrow_: bool
                            out12_: bool
                            out12_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_19_narrow_ = out12_
                            if d_19_narrow_:
                                d_20_repaired_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_20_repaired_ = out13_
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_20_repaired_))):])
                                currentConstrainedOut = d_20_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_21_stablePrefix_: _dafny.Seq
                                d_21_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_22_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_21_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_22_next_ = out14_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_22_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_23_gApp_: _dafny.Seq
                                    d_24_iApp_: bool
                                    d_25_cApp_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                    d_23_gApp_ = out15_
                                    d_24_iApp_ = out16_
                                    d_25_cApp_ = out17_
                                    generated = d_23_gApp_
                                    insideConstrainedOut = d_24_iApp_
                                    currentConstrainedOut = d_25_cApp_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

