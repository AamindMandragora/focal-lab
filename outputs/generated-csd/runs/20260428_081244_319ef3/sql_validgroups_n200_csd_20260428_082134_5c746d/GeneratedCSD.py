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
        d_2_openTok_: _dafny.Seq
        d_2_openTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_3_fromTok_: _dafny.Seq
        d_3_fromTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_4_whereTok_: _dafny.Seq
        d_4_whereTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE"))
        d_5_selectTok_: _dafny.Seq
        d_5_selectTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_chunkBudget_: int
                        d_6_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_7_chunkedGenerated_: _dafny.Seq
                        d_8_stoppedOnOpenSpan_: bool
                        d_9_stoppedOnEos_: bool
                        d_10_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, d_2_openTok_, eosToken)
                        d_7_chunkedGenerated_ = out0_
                        d_8_stoppedOnOpenSpan_ = out1_
                        d_9_stoppedOnEos_ = out2_
                        d_10_stepsUsed_ = out3_
                        generated = d_7_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                        if d_9_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_8_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_11_completeNow_: bool
                        d_11_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_11_completeNow_:
                            d_12_closedGenerated_: _dafny.Seq
                            d_13_closedInside_: bool
                            d_14_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_12_closedGenerated_ = out4_
                            d_13_closedInside_ = out5_
                            d_14_closedCurrent_ = out6_
                            generated = d_12_closedGenerated_
                            insideConstrainedOut = d_13_closedInside_
                            currentConstrainedOut = d_14_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_15_deadEnd_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_15_deadEnd_ = out7_
                            if d_15_deadEnd_:
                                d_16_repaired_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_3_fromTok_)
                                d_16_repaired_ = out8_
                                if (len(d_16_repaired_)) == (len(currentConstrainedOut)):
                                    d_17_repaired2_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out9_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_4_whereTok_)
                                    d_17_repaired2_ = out9_
                                    if (len(d_17_repaired2_)) < (len(d_16_repaired_)):
                                        d_16_repaired_ = d_17_repaired2_
                                    elif True:
                                        d_18_repaired3_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out10_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_5_selectTok_)
                                        d_18_repaired3_ = out10_
                                        if (len(d_18_repaired3_)) < (len(d_16_repaired_)):
                                            d_16_repaired_ = d_18_repaired3_
                                d_19_stablePrefix_: _dafny.Seq
                                d_19_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                generated = (d_19_stablePrefix_) + (d_16_repaired_)
                                currentConstrainedOut = d_16_repaired_
                            elif True:
                                d_20_stablePrefix2_: _dafny.Seq
                                d_20_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_21_constrainedPrompt_: _dafny.Seq
                                d_21_constrainedPrompt_ = (prompt) + (d_20_stablePrefix2_)
                                d_22_budget_: int
                                d_22_budget_ = stepTokenBudget
                                if (d_22_budget_) > ((maxSteps) - (d_1_steps_)):
                                    d_22_budget_ = (maxSteps) - (d_1_steps_)
                                if (d_22_budget_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_23_currentOut_: _dafny.Seq
                                    d_24_hitEos_: bool
                                    d_25_stepsUsed2_: int
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: int
                                    out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, d_22_budget_, eosToken)
                                    d_23_currentOut_ = out11_
                                    d_24_hitEos_ = out12_
                                    d_25_stepsUsed2_ = out13_
                                    d_1_steps_ = (d_1_steps_) + (d_25_stepsUsed2_)
                                    if d_24_hitEos_:
                                        raise _dafny.Break("0")
                                    elif True:
                                        currentConstrainedOut = d_23_currentOut_
                                        generated = (d_20_stablePrefix2_) + (currentConstrainedOut)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

