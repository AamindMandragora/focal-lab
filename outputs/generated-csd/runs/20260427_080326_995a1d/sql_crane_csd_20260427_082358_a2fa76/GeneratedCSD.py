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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        d_2_openedSeen_: bool
        d_2_openedSeen_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_openedSeen_:
                            raise _dafny.Break("0")
                        elif True:
                            d_3_remainingChunk_: int
                            d_3_remainingChunk_ = (maxSteps) - (d_1_steps_)
                            if (d_3_remainingChunk_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_4_gChunk_: _dafny.Seq
                                d_5_stoppedOnOpenSpan_: bool
                                d_6_stoppedOnEos_: bool
                                d_7_usedChunk_: int
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: bool
                                out3_: int
                                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_remainingChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_4_gChunk_ = out0_
                                d_5_stoppedOnOpenSpan_ = out1_
                                d_6_stoppedOnEos_ = out2_
                                d_7_usedChunk_ = out3_
                                if (d_7_usedChunk_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_4_gChunk_
                                    d_1_steps_ = (d_1_steps_) + (d_7_usedChunk_)
                                    if d_6_stoppedOnEos_:
                                        raise _dafny.Break("0")
                                    elif True:
                                        if d_5_stoppedOnOpenSpan_:
                                            insideConstrainedOut = True
                                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                            d_2_openedSeen_ = True
                                        elif True:
                                            raise _dafny.Break("0")
                    elif True:
                        d_8_completeNow_: bool
                        d_8_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_completeNow_:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                d_9_gClose_: _dafny.Seq
                                d_10_inClose_: bool
                                d_11_cClose_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_9_gClose_ = out4_
                                d_10_inClose_ = out5_
                                d_11_cClose_ = out6_
                                generated = d_9_gClose_
                                insideConstrainedOut = d_10_inClose_
                                currentConstrainedOut = d_11_cClose_
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_12_dead_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_12_dead_ = out7_
                            if d_12_dead_:
                                d_13_stableDead_: _dafny.Seq
                                d_13_stableDead_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_14_gRoll_: _dafny.Seq
                                d_15_cRoll_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: _dafny.Seq
                                out8_, out9_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_13_stableDead_, generated, currentConstrainedOut)
                                d_14_gRoll_ = out8_
                                d_15_cRoll_ = out9_
                                generated = d_14_gRoll_
                                currentConstrainedOut = d_15_cRoll_
                                insideConstrainedOut = True
                            elif True:
                                if ((d_1_steps_) + (1)) <= (maxSteps):
                                    d_16_next_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                    d_16_next_ = out10_
                                    if (d_16_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_17_gApp_: _dafny.Seq
                                        d_18_inApp_: bool
                                        d_19_cApp_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: bool
                                        out13_: _dafny.Seq
                                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                        d_17_gApp_ = out11_
                                        d_18_inApp_ = out12_
                                        d_19_cApp_ = out13_
                                        generated = d_17_gApp_
                                        insideConstrainedOut = d_18_inApp_
                                        currentConstrainedOut = d_19_cApp_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

