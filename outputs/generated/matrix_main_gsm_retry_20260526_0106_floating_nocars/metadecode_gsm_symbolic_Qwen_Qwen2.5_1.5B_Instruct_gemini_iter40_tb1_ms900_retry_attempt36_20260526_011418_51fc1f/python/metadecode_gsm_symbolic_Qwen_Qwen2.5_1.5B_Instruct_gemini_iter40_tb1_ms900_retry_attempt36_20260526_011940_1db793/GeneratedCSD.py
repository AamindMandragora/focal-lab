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
        d_1_narrowThreshold_: int
        d_1_narrowThreshold_ = 10
        d_2_rollbackLimit_: int
        d_2_rollbackLimit_ = 30
        d_3_penaltyTokens_: _dafny.Seq
        d_3_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/"))])
        d_4_steps_: int
        d_4_steps_ = 0
        with _dafny.label("0"):
            while (d_4_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remainingBudget_: int
                        d_5_remainingBudget_ = (maxSteps) - (d_4_steps_)
                        if (d_5_remainingBudget_) > (0):
                            d_6_chunkBudget_: int
                            d_6_chunkBudget_ = 15
                            if (d_6_chunkBudget_) > (d_5_remainingBudget_):
                                d_6_chunkBudget_ = d_5_remainingBudget_
                            d_7_chunkedG_: _dafny.Seq
                            d_8_stoppedOpen_: bool
                            d_9_stoppedEos_: bool
                            d_10_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_7_chunkedG_ = out0_
                            d_8_stoppedOpen_ = out1_
                            d_9_stoppedEos_ = out2_
                            d_10_stepsUsed_ = out3_
                            generated = d_7_chunkedG_
                            d_4_steps_ = (d_4_steps_) + (d_10_stepsUsed_)
                            if d_9_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_8_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif (d_4_steps_) < (maxSteps):
                                d_11_openedGenerated_: _dafny.Seq
                                d_12_openedInside_: bool
                                d_13_openedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_11_openedGenerated_ = out4_
                                d_12_openedInside_ = out5_
                                d_13_openedCurrent_ = out6_
                                generated = d_11_openedGenerated_
                                insideConstrainedOut = d_12_openedInside_
                                currentConstrainedOut = d_13_openedCurrent_
                                d_4_steps_ = (d_4_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_14_closedGenerated_: _dafny.Seq
                            d_15_closedInside_: bool
                            d_16_closedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_14_closedGenerated_ = out7_
                            d_15_closedInside_ = out8_
                            d_16_closedCurrent_ = out9_
                            generated = d_14_closedGenerated_
                            insideConstrainedOut = d_15_closedInside_
                            currentConstrainedOut = d_16_closedCurrent_
                            d_4_steps_ = (d_4_steps_) + (1)
                        elif (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
                            d_17_rolledGenerated_: _dafny.Seq
                            d_18_rolledCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: _dafny.Seq
                            out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_17_rolledGenerated_ = out10_
                            d_18_rolledCurrent_ = out11_
                            generated = d_17_rolledGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_18_rolledCurrent_
                            d_4_steps_ = (d_4_steps_) + (1)
                        elif True:
                            d_19_constrainedPrompt_: _dafny.Seq
                            d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_20_validCount_: int
                            out12_: int
                            out12_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_20_validCount_ = out12_
                            if (d_20_validCount_) <= (d_1_narrowThreshold_):
                                d_21_next_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_penaltyTokens_, _dafny.BigRational('1e1'), d_1_narrowThreshold_, eosToken)
                                d_21_next_ = out13_
                                d_4_steps_ = (d_4_steps_) + (1)
                                if (d_21_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_appendedGenerated_: _dafny.Seq
                                    d_23_appendedInside_: bool
                                    d_24_appendedCurrent_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_22_appendedGenerated_ = out14_
                                    d_23_appendedInside_ = out15_
                                    d_24_appendedCurrent_ = out16_
                                    generated = d_22_appendedGenerated_
                                    insideConstrainedOut = d_23_appendedInside_
                                    currentConstrainedOut = d_24_appendedCurrent_
                            elif True:
                                d_25_remaining_: int
                                d_25_remaining_ = (maxSteps) - (d_4_steps_)
                                d_26_symbolBudget_: int
                                if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_25_remaining_)):
                                    d_26_symbolBudget_ = d_25_remaining_
                                elif True:
                                    d_26_symbolBudget_ = stepTokenBudget
                                d_27_symbolGenerated_: _dafny.Seq
                                d_28_symbolOut_: _dafny.Seq
                                d_29_hitEos_: bool
                                d_30_stepsUsed_: int
                                out17_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: int
                                out17_, out18_, out19_, out20_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_19_constrainedPrompt_, generated, currentConstrainedOut, d_26_symbolBudget_, eosToken)
                                d_27_symbolGenerated_ = out17_
                                d_28_symbolOut_ = out18_
                                d_29_hitEos_ = out19_
                                d_30_stepsUsed_ = out20_
                                generated = d_27_symbolGenerated_
                                currentConstrainedOut = d_28_symbolOut_
                                d_4_steps_ = (d_4_steps_) + (d_30_stepsUsed_)
                                if d_29_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

