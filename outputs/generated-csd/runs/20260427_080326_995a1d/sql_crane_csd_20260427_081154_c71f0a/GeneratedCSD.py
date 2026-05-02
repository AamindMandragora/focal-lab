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
                        if (d_1_steps_) < (maxSteps):
                            d_2_openedGenerated_: _dafny.Seq
                            d_3_openedInside_: bool
                            d_4_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_2_openedGenerated_ = out0_
                            d_3_openedInside_ = out1_
                            d_4_openedCurrent_ = out2_
                            generated = d_2_openedGenerated_
                            insideConstrainedOut = d_3_openedInside_
                            currentConstrainedOut = d_4_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_5_completeNow_: bool
                        d_5_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        d_6_fromTail_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))
                        d_6_fromTail_ = out3_
                        d_7_canCloseNow_: bool
                        d_7_canCloseNow_ = (d_5_completeNow_) and (((len(d_6_fromTail_)) > (0)) or ((len(currentConstrainedOut)) >= (20)))
                        if (d_7_canCloseNow_) and (((d_1_steps_) + (1)) <= (maxSteps)):
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
                                insideConstrainedOut = True
                                d_15_completeAfterRollback_: bool
                                d_15_completeAfterRollback_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                d_16_fromTailAfterRollback_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))
                                d_16_fromTailAfterRollback_ = out10_
                                d_17_canCloseAfterRollback_: bool
                                d_17_canCloseAfterRollback_ = (d_15_completeAfterRollback_) and (((len(d_16_fromTailAfterRollback_)) > (0)) or ((len(currentConstrainedOut)) >= (20)))
                                if (d_17_canCloseAfterRollback_) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                    d_18_closedGenerated2_: _dafny.Seq
                                    d_19_closedInside2_: bool
                                    d_20_closedCurrent2_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_18_closedGenerated2_ = out11_
                                    d_19_closedInside2_ = out12_
                                    d_20_closedCurrent2_ = out13_
                                    generated = d_18_closedGenerated2_
                                    insideConstrainedOut = d_19_closedInside2_
                                    currentConstrainedOut = d_20_closedCurrent2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_21_remainingAfterRollback_: int
                                    d_21_remainingAfterRollback_ = (maxSteps) - (d_1_steps_)
                                    if ((d_21_remainingAfterRollback_) == (0)) or ((stepTokenBudget) == (0)):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_22_stablePrefix2_: _dafny.Seq
                                        d_22_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                        d_23_constrainedPrompt2_: _dafny.Seq
                                        d_23_constrainedPrompt2_ = (prompt) + (d_22_stablePrefix2_)
                                        d_24_currentSym2_: _dafny.Seq
                                        d_25_hitEos2_: bool
                                        d_26_stepsUsed2_: int
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: int
                                        out14_, out15_, out16_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_23_constrainedPrompt2_, currentConstrainedOut, stepTokenBudget, eosToken)
                                        d_24_currentSym2_ = out14_
                                        d_25_hitEos2_ = out15_
                                        d_26_stepsUsed2_ = out16_
                                        if (d_25_hitEos2_) or ((d_26_stepsUsed2_) == (0)):
                                            raise _dafny.Break("0")
                                        elif True:
                                            if ((d_1_steps_) + (d_26_stepsUsed2_)) > (maxSteps):
                                                raise _dafny.Break("0")
                                            elif True:
                                                generated = (d_22_stablePrefix2_) + (d_24_currentSym2_)
                                                currentConstrainedOut = d_24_currentSym2_
                                                insideConstrainedOut = True
                                                d_1_steps_ = (d_1_steps_) + (d_26_stepsUsed2_)
                            elif True:
                                d_27_remaining_: int
                                d_27_remaining_ = (maxSteps) - (d_1_steps_)
                                if ((d_27_remaining_) == (0)) or ((stepTokenBudget) == (0)):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_28_stablePrefix_: _dafny.Seq
                                    d_28_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_29_constrainedPrompt_: _dafny.Seq
                                    d_29_constrainedPrompt_ = (prompt) + (d_28_stablePrefix_)
                                    d_30_currentSym_: _dafny.Seq
                                    d_31_hitEos_: bool
                                    d_32_stepsUsedSym_: int
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: int
                                    out17_, out18_, out19_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, stepTokenBudget, eosToken)
                                    d_30_currentSym_ = out17_
                                    d_31_hitEos_ = out18_
                                    d_32_stepsUsedSym_ = out19_
                                    if (d_31_hitEos_) or ((d_32_stepsUsedSym_) == (0)):
                                        raise _dafny.Break("0")
                                    elif True:
                                        if ((d_1_steps_) + (d_32_stepsUsedSym_)) > (maxSteps):
                                            raise _dafny.Break("0")
                                        elif True:
                                            generated = (d_28_stablePrefix_) + (d_30_currentSym_)
                                            currentConstrainedOut = d_30_currentSym_
                                            insideConstrainedOut = True
                                            d_1_steps_ = (d_1_steps_) + (d_32_stepsUsedSym_)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

