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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_chunkBudget_: int
                        d_2_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_3_chunkedGenerated_: _dafny.Seq
                        d_4_stoppedOnOpenSpan_: bool
                        d_5_stoppedOnEos_: bool
                        d_6_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_3_chunkedGenerated_ = out0_
                        d_4_stoppedOnOpenSpan_ = out1_
                        d_5_stoppedOnEos_ = out2_
                        d_6_stepsUsed_ = out3_
                        generated = d_3_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                        if d_5_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_4_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        d_7_completeNow_: bool
                        d_7_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_completeNow_:
                            if (d_1_steps_) < (maxSteps):
                                d_8_closedGenerated_: _dafny.Seq
                                d_9_closedInside_: bool
                                d_10_closedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_8_closedGenerated_ = out4_
                                d_9_closedInside_ = out5_
                                d_10_closedCurrent_ = out6_
                                generated = d_8_closedGenerated_
                                insideConstrainedOut = d_9_closedInside_
                                currentConstrainedOut = d_10_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_11_deadEnd_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_11_deadEnd_ = out7_
                            if d_11_deadEnd_:
                                d_12_stablePrefixDead_: _dafny.Seq
                                d_12_stablePrefixDead_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_13_rolledGenerated_: _dafny.Seq
                                d_14_rolledCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: _dafny.Seq
                                out8_, out9_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_12_stablePrefixDead_, generated, currentConstrainedOut)
                                d_13_rolledGenerated_ = out8_
                                d_14_rolledCurrent_ = out9_
                                generated = d_13_rolledGenerated_
                                currentConstrainedOut = d_14_rolledCurrent_
                                d_15_completeAfterRollback_: bool
                                d_15_completeAfterRollback_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_15_completeAfterRollback_:
                                    if (d_1_steps_) < (maxSteps):
                                        d_16_closedGenerated2_: _dafny.Seq
                                        d_17_closedInside2_: bool
                                        d_18_closedCurrent2_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_16_closedGenerated2_ = out10_
                                        d_17_closedInside2_ = out11_
                                        d_18_closedCurrent2_ = out12_
                                        generated = d_16_closedGenerated2_
                                        insideConstrainedOut = d_17_closedInside2_
                                        currentConstrainedOut = d_18_closedCurrent2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    d_19_remainingAfterRollback_: int
                                    d_19_remainingAfterRollback_ = (maxSteps) - (d_1_steps_)
                                    if (d_19_remainingAfterRollback_) == (0):
                                        raise _dafny.Break("0")
                                    elif True:
                                        if (stepTokenBudget) == (0):
                                            raise _dafny.Break("0")
                                        elif True:
                                            d_20_stablePrefix2_: _dafny.Seq
                                            d_20_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                            d_21_constrainedPrompt2_: _dafny.Seq
                                            d_21_constrainedPrompt2_ = (prompt) + (d_20_stablePrefix2_)
                                            d_22_oldLen2_: int
                                            d_22_oldLen2_ = len(currentConstrainedOut)
                                            d_23_currentSym2_: _dafny.Seq
                                            d_24_hitEos2_: bool
                                            d_25_stepsUsed2_: int
                                            out13_: _dafny.Seq
                                            out14_: bool
                                            out15_: int
                                            out13_, out14_, out15_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_21_constrainedPrompt2_, currentConstrainedOut, stepTokenBudget, eosToken)
                                            d_23_currentSym2_ = out13_
                                            d_24_hitEos2_ = out14_
                                            d_25_stepsUsed2_ = out15_
                                            if (d_24_hitEos2_) or ((d_25_stepsUsed2_) == (0)):
                                                raise _dafny.Break("0")
                                            elif True:
                                                if ((d_1_steps_) + (d_25_stepsUsed2_)) > (maxSteps):
                                                    raise _dafny.Break("0")
                                                elif True:
                                                    generated = (d_20_stablePrefix2_) + (d_23_currentSym2_)
                                                    currentConstrainedOut = d_23_currentSym2_
                                                    insideConstrainedOut = True
                                                    d_1_steps_ = (d_1_steps_) + (d_25_stepsUsed2_)
                                                    if (len(currentConstrainedOut)) < (d_22_oldLen2_):
                                                        raise _dafny.Break("0")
                            elif True:
                                d_26_remaining_: int
                                d_26_remaining_ = (maxSteps) - (d_1_steps_)
                                if (d_26_remaining_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    if (stepTokenBudget) == (0):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_27_stablePrefix_: _dafny.Seq
                                        d_27_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                        d_28_constrainedPrompt_: _dafny.Seq
                                        d_28_constrainedPrompt_ = (prompt) + (d_27_stablePrefix_)
                                        d_29_oldLen_: int
                                        d_29_oldLen_ = len(currentConstrainedOut)
                                        d_30_currentSym_: _dafny.Seq
                                        d_31_hitEos_: bool
                                        d_32_stepsUsedSym_: int
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: int
                                        out16_, out17_, out18_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, stepTokenBudget, eosToken)
                                        d_30_currentSym_ = out16_
                                        d_31_hitEos_ = out17_
                                        d_32_stepsUsedSym_ = out18_
                                        if (d_31_hitEos_) or ((d_32_stepsUsedSym_) == (0)):
                                            raise _dafny.Break("0")
                                        elif True:
                                            if ((d_1_steps_) + (d_32_stepsUsedSym_)) > (maxSteps):
                                                raise _dafny.Break("0")
                                            elif True:
                                                generated = (d_27_stablePrefix_) + (d_30_currentSym_)
                                                currentConstrainedOut = d_30_currentSym_
                                                insideConstrainedOut = True
                                                d_1_steps_ = (d_1_steps_) + (d_32_stepsUsedSym_)
                                                if (len(currentConstrainedOut)) < (d_29_oldLen_):
                                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

