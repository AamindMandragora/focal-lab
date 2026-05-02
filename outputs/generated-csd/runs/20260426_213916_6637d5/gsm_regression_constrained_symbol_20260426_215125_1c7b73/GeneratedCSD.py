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
        d_2_didWarmup_: bool
        d_2_didWarmup_ = insideConstrained
        d_3_openedSpan_: bool
        d_3_openedSpan_ = insideConstrained
        d_4_warmupBudget_: int
        d_4_warmupBudget_ = 3
        if (maxSteps) < (d_4_warmupBudget_):
            d_4_warmupBudget_ = maxSteps
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_5_completeNow_: bool
                        d_5_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_completeNow_:
                            d_6_gClosed_: _dafny.Seq
                            d_7_inClosed_: bool
                            d_8_cClosed_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_gClosed_ = out0_
                            d_7_inClosed_ = out1_
                            d_8_cClosed_ = out2_
                            generated = d_6_gClosed_
                            insideConstrainedOut = d_7_inClosed_
                            currentConstrainedOut = d_8_cClosed_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_openedSpan_ = True
                            d_2_didWarmup_ = True
                        elif True:
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_10_nextC_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_10_nextC_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_nextC_) == (eosToken):
                                d_11_stablePrefix_: _dafny.Seq
                                d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_12_gRb_: _dafny.Seq
                                d_13_cRb_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: _dafny.Seq
                                out4_, out5_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_11_stablePrefix_, generated, currentConstrainedOut)
                                d_12_gRb_ = out4_
                                d_13_cRb_ = out5_
                                generated = d_12_gRb_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_openedSpan_ = True
                                raise _dafny.Break("0")
                            elif True:
                                d_14_gApp_: _dafny.Seq
                                d_15_inApp_: bool
                                d_16_cApp_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_nextC_)
                                d_14_gApp_ = out6_
                                d_15_inApp_ = out7_
                                d_16_cApp_ = out8_
                                generated = d_14_gApp_
                                insideConstrainedOut = d_15_inApp_
                                currentConstrainedOut = d_16_cApp_
                    elif True:
                        if not(d_2_didWarmup_):
                            d_17_remaining_: int
                            d_17_remaining_ = (maxSteps) - (d_1_steps_)
                            d_18_chunk_: int
                            d_18_chunk_ = d_4_warmupBudget_
                            if (d_17_remaining_) < (d_18_chunk_):
                                d_18_chunk_ = d_17_remaining_
                            if (d_18_chunk_) == (0):
                                d_2_didWarmup_ = True
                            elif True:
                                d_19_gChunk_: _dafny.Seq
                                d_20_stoppedOnOpenSpan_: bool
                                d_21_stoppedOnEos_: bool
                                d_22_used_: int
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: bool
                                out12_: int
                                out9_, out10_, out11_, out12_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_18_chunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_19_gChunk_ = out9_
                                d_20_stoppedOnOpenSpan_ = out10_
                                d_21_stoppedOnEos_ = out11_
                                d_22_used_ = out12_
                                generated = d_19_gChunk_
                                d_1_steps_ = (d_1_steps_) + (d_22_used_)
                                d_2_didWarmup_ = True
                                if d_21_stoppedOnEos_:
                                    raise _dafny.Break("0")
                        elif True:
                            if (not(d_3_openedSpan_)) and ((d_1_steps_) < (maxSteps)):
                                d_23_gOpen_: _dafny.Seq
                                d_24_inOpen_: bool
                                d_25_cOpen_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_23_gOpen_ = out13_
                                d_24_inOpen_ = out14_
                                d_25_cOpen_ = out15_
                                generated = d_23_gOpen_
                                insideConstrainedOut = d_24_inOpen_
                                currentConstrainedOut = d_25_cOpen_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_openedSpan_ = True
                            elif True:
                                d_26_nextU_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_26_nextU_ = out16_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_26_nextU_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_26_nextU_]))
                    pass
            pass
        if insideConstrainedOut:
            d_27_completeEnd_: bool
            d_27_completeEnd_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if (d_27_completeEnd_) and ((d_1_steps_) < (maxSteps)):
                d_28_gFinal_: _dafny.Seq
                d_29_inFinal_: bool
                d_30_cFinal_: _dafny.Seq
                out17_: _dafny.Seq
                out18_: bool
                out19_: _dafny.Seq
                out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_28_gFinal_ = out17_
                d_29_inFinal_ = out18_
                d_30_cFinal_ = out19_
                generated = d_28_gFinal_
                insideConstrainedOut = d_29_inFinal_
                currentConstrainedOut = d_30_cFinal_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_31_stableEnd_: _dafny.Seq
                d_31_stableEnd_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_32_gEndRb_: _dafny.Seq
                d_33_cEndRb_: _dafny.Seq
                out20_: _dafny.Seq
                out21_: _dafny.Seq
                out20_, out21_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_31_stableEnd_, generated, currentConstrainedOut)
                d_32_gEndRb_ = out20_
                d_33_cEndRb_ = out21_
                generated = d_32_gEndRb_
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

