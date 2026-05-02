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
                        d_8_hasFromNow_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))
                        d_8_hasFromNow_ = out4_
                        d_9_canCloseNow_: bool
                        d_9_canCloseNow_ = (d_7_completeNow_) and (((len(d_8_hasFromNow_)) > (0)) or ((len(currentConstrainedOut)) >= (12)))
                        if d_9_canCloseNow_:
                            d_10_closedGenerated_: _dafny.Seq
                            d_11_closedInside_: bool
                            d_12_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_closedGenerated_ = out5_
                            d_11_closedInside_ = out6_
                            d_12_closedCurrent_ = out7_
                            generated = d_10_closedGenerated_
                            insideConstrainedOut = d_11_closedInside_
                            currentConstrainedOut = d_12_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_deadEnd_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_13_deadEnd_ = out8_
                            if d_13_deadEnd_:
                                d_14_stablePrefixDead_: _dafny.Seq
                                d_14_stablePrefixDead_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_15_rolledGenerated_: _dafny.Seq
                                d_16_rolledCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: _dafny.Seq
                                out9_, out10_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_14_stablePrefixDead_, generated, currentConstrainedOut)
                                d_15_rolledGenerated_ = out9_
                                d_16_rolledCurrent_ = out10_
                                generated = d_15_rolledGenerated_
                                currentConstrainedOut = d_16_rolledCurrent_
                                insideConstrainedOut = True
                                d_17_completeAfterRollback_: bool
                                d_17_completeAfterRollback_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                d_18_hasFromAfterRollback_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))
                                d_18_hasFromAfterRollback_ = out11_
                                d_19_canCloseAfterRollback_: bool
                                d_19_canCloseAfterRollback_ = (d_17_completeAfterRollback_) and (((len(d_18_hasFromAfterRollback_)) > (0)) or ((len(currentConstrainedOut)) >= (12)))
                                if d_19_canCloseAfterRollback_:
                                    if (d_1_steps_) < (maxSteps):
                                        d_20_closedGenerated2_: _dafny.Seq
                                        d_21_closedInside2_: bool
                                        d_22_closedCurrent2_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: _dafny.Seq
                                        out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_20_closedGenerated2_ = out12_
                                        d_21_closedInside2_ = out13_
                                        d_22_closedCurrent2_ = out14_
                                        generated = d_20_closedGenerated2_
                                        insideConstrainedOut = d_21_closedInside2_
                                        currentConstrainedOut = d_22_closedCurrent2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    d_23_remainingAfterRollback_: int
                                    d_23_remainingAfterRollback_ = (maxSteps) - (d_1_steps_)
                                    if ((d_23_remainingAfterRollback_) == (0)) or ((stepTokenBudget) == (0)):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_24_stablePrefix2_: _dafny.Seq
                                        d_24_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                        d_25_constrainedPrompt2_: _dafny.Seq
                                        d_25_constrainedPrompt2_ = (prompt) + (d_24_stablePrefix2_)
                                        d_26_oldLen2_: int
                                        d_26_oldLen2_ = len(currentConstrainedOut)
                                        d_27_currentSym2_: _dafny.Seq
                                        d_28_hitEos2_: bool
                                        d_29_stepsUsed2_: int
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: int
                                        out15_, out16_, out17_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_25_constrainedPrompt2_, currentConstrainedOut, stepTokenBudget, eosToken)
                                        d_27_currentSym2_ = out15_
                                        d_28_hitEos2_ = out16_
                                        d_29_stepsUsed2_ = out17_
                                        if (d_28_hitEos2_) or ((d_29_stepsUsed2_) == (0)):
                                            raise _dafny.Break("0")
                                        elif True:
                                            if ((d_1_steps_) + (d_29_stepsUsed2_)) > (maxSteps):
                                                raise _dafny.Break("0")
                                            elif True:
                                                generated = (d_24_stablePrefix2_) + (d_27_currentSym2_)
                                                currentConstrainedOut = d_27_currentSym2_
                                                insideConstrainedOut = True
                                                d_1_steps_ = (d_1_steps_) + (d_29_stepsUsed2_)
                                                if (len(currentConstrainedOut)) < (d_26_oldLen2_):
                                                    raise _dafny.Break("0")
                            elif True:
                                d_30_remaining_: int
                                d_30_remaining_ = (maxSteps) - (d_1_steps_)
                                if ((d_30_remaining_) == (0)) or ((stepTokenBudget) == (0)):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_31_stablePrefix_: _dafny.Seq
                                    d_31_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_32_constrainedPrompt_: _dafny.Seq
                                    d_32_constrainedPrompt_ = (prompt) + (d_31_stablePrefix_)
                                    d_33_oldLen_: int
                                    d_33_oldLen_ = len(currentConstrainedOut)
                                    d_34_currentSym_: _dafny.Seq
                                    d_35_hitEos_: bool
                                    d_36_stepsUsedSym_: int
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: int
                                    out18_, out19_, out20_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_32_constrainedPrompt_, currentConstrainedOut, stepTokenBudget, eosToken)
                                    d_34_currentSym_ = out18_
                                    d_35_hitEos_ = out19_
                                    d_36_stepsUsedSym_ = out20_
                                    if (d_35_hitEos_) or ((d_36_stepsUsedSym_) == (0)):
                                        raise _dafny.Break("0")
                                    elif True:
                                        if ((d_1_steps_) + (d_36_stepsUsedSym_)) > (maxSteps):
                                            raise _dafny.Break("0")
                                        elif True:
                                            generated = (d_31_stablePrefix_) + (d_34_currentSym_)
                                            currentConstrainedOut = d_34_currentSym_
                                            insideConstrainedOut = True
                                            d_1_steps_ = (d_1_steps_) + (d_36_stepsUsedSym_)
                                            if (len(currentConstrainedOut)) < (d_33_oldLen_):
                                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

