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
        d_2_deadEndThreshold_: int
        d_2_deadEndThreshold_ = 1
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_chunkBudget_: int
                        d_3_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkedGenerated_: _dafny.Seq
                        d_5_stoppedOpen_: bool
                        d_6_stoppedEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkedGenerated_ = out0_
                        d_5_stoppedOpen_ = out1_
                        d_6_stoppedEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        generated = d_4_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_8_isComplete_: bool
                        d_8_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_isComplete_:
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out4_
                            d_10_closedInside_ = out5_
                            d_11_closedCurrent_ = out6_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_narrow_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, d_2_deadEndThreshold_)
                            d_12_narrow_ = out7_
                            if d_12_narrow_:
                                d_13_stablePrefix_: _dafny.Seq
                                d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_14_repairedGenerated_: _dafny.Seq
                                d_15_repairedCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: _dafny.Seq
                                out8_, out9_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_13_stablePrefix_, generated, currentConstrainedOut)
                                d_14_repairedGenerated_ = out8_
                                d_15_repairedCurrent_ = out9_
                                generated = d_14_repairedGenerated_
                                currentConstrainedOut = d_15_repairedCurrent_
                            elif True:
                                d_16_stablePrefix_: _dafny.Seq
                                d_16_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_17_constrainedPrompt_: _dafny.Seq
                                d_17_constrainedPrompt_ = (prompt) + (d_16_stablePrefix_)
                                d_18_remaining_: int
                                d_18_remaining_ = (maxSteps) - (d_1_steps_)
                                d_19_symbolBudget_: int
                                d_19_symbolBudget_ = stepTokenBudget
                                if (d_18_remaining_) < (d_19_symbolBudget_):
                                    d_19_symbolBudget_ = d_18_remaining_
                                if (d_19_symbolBudget_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_20_currentOut_: _dafny.Seq
                                    d_21_hitEos_: bool
                                    d_22_stepsUsed_: int
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: int
                                    out10_, out11_, out12_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, d_19_symbolBudget_, eosToken)
                                    d_20_currentOut_ = out10_
                                    d_21_hitEos_ = out11_
                                    d_22_stepsUsed_ = out12_
                                    generated = (d_16_stablePrefix_) + (d_20_currentOut_)
                                    currentConstrainedOut = d_20_currentOut_
                                    d_1_steps_ = (d_1_steps_) + (d_22_stepsUsed_)
                                    if d_21_hitEos_:
                                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

