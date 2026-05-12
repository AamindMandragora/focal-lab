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
        d_2_rollbackLimit_: int
        d_2_rollbackLimit_ = 24
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 10
        d_4_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_4_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_openedGenerated_: _dafny.Seq
                        d_6_openedInside_: bool
                        d_7_openedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_5_openedGenerated_ = out1_
                        d_6_openedInside_ = out2_
                        d_7_openedCurrent_ = out3_
                        generated = d_5_openedGenerated_
                        insideConstrainedOut = d_6_openedInside_
                        currentConstrainedOut = d_7_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
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
                        d_11_shouldRollback_: bool
                        d_11_shouldRollback_ = False
                        if (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_11_shouldRollback_ = out7_
                        if d_11_shouldRollback_:
                            d_12_rolledGenerated_: _dafny.Seq
                            d_13_rolledCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: _dafny.Seq
                            out8_, out9_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_12_rolledGenerated_ = out8_
                            d_13_rolledCurrent_ = out9_
                            generated = d_12_rolledGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_13_rolledCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                            d_16_validCount_: int
                            out10_: int
                            out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_16_validCount_ = out10_
                            d_17_next_: _dafny.Seq
                            d_17_next_ = eosToken
                            if (len(currentConstrainedOut)) == (0):
                                if (d_16_validCount_) <= (d_3_narrowThreshold_):
                                    out11_: _dafny.Seq
                                    out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                    d_17_next_ = out11_
                                elif True:
                                    out12_: _dafny.Seq
                                    out12_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                    d_17_next_ = out12_
                            elif (d_16_validCount_) <= (d_3_narrowThreshold_):
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), currentConstrainedOut, _dafny.BigRational('3e0'), d_3_narrowThreshold_, eosToken)
                                d_17_next_ = out13_
                            elif (len(d_4_flatGroups_)) > (0):
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, currentConstrainedOut, _dafny.BigRational('2e0'), eosToken)
                                d_17_next_ = out14_
                            elif True:
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_17_next_ = out15_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_18_appendedGenerated_: _dafny.Seq
                                d_19_appendedInside_: bool
                                d_20_appendedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                d_18_appendedGenerated_ = out16_
                                d_19_appendedInside_ = out17_
                                d_20_appendedCurrent_ = out18_
                                generated = d_18_appendedGenerated_
                                insideConstrainedOut = d_19_appendedInside_
                                currentConstrainedOut = d_20_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

