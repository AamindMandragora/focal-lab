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
        d_2_openedOnce_: bool
        d_2_openedOnce_ = insideConstrained
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_rollbackLimit_: int
        d_4_rollbackLimit_ = 64
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_2_openedOnce_):
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out0_
                            d_6_openedInside_ = out1_
                            d_7_openedCurrent_ = out2_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_2_openedOnce_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_nextOutside_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_nextOutside_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_nextOutside_]))
                                if (d_8_nextOutside_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_9_enteredGenerated_: _dafny.Seq
                                    d_10_enteredInside_: bool
                                    d_11_enteredCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_9_enteredGenerated_ = out4_
                                    d_10_enteredInside_ = out5_
                                    d_11_enteredCurrent_ = out6_
                                    generated = d_9_enteredGenerated_
                                    insideConstrainedOut = d_10_enteredInside_
                                    currentConstrainedOut = d_11_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out7_
                        d_13_closedInside_ = out8_
                        d_14_closedCurrent_ = out9_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_deadEnd_: bool
                        out10_: bool
                        out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_15_deadEnd_ = out10_
                        d_16_shouldRollback_: bool
                        d_16_shouldRollback_ = d_15_deadEnd_
                        if not(d_16_shouldRollback_):
                            if (len(currentConstrainedOut)) >= (d_4_rollbackLimit_):
                                d_16_shouldRollback_ = True
                        if d_16_shouldRollback_:
                            d_17_rolledGenerated_: _dafny.Seq
                            d_18_rolledCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_17_rolledGenerated_ = out11_
                            d_18_rolledCurrent_ = out12_
                            generated = d_17_rolledGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_18_rolledCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_19_stablePrefix_: _dafny.Seq
                            d_19_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_20_constrainedPrompt_: _dafny.Seq
                            d_20_constrainedPrompt_ = (prompt) + (d_19_stablePrefix_)
                            d_21_validCount_: int
                            out13_: int
                            out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_21_validCount_ = out13_
                            d_22_next_: _dafny.Seq
                            d_22_next_ = eosToken
                            if (len(currentConstrainedOut)) > (0):
                                d_23_lastTok_: _dafny.Seq
                                d_23_lastTok_ = (generated)[(len(generated)) - (1)]
                                d_24_since_: int
                                out14_: int
                                out14_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_23_lastTok_)
                                d_24_since_ = out14_
                                if (d_24_since_) < (3):
                                    out15_: _dafny.Seq
                                    out15_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                    d_22_next_ = out15_
                                elif (d_21_validCount_) <= (4):
                                    out16_: _dafny.Seq
                                    out16_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_22_next_ = out16_
                                elif ((len(validTokenGroups)) > (0)) and ((d_21_validCount_) <= (d_3_narrowThreshold_)):
                                    out17_: _dafny.Seq
                                    out17_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                    d_22_next_ = out17_
                                elif (len(validTokenGroups)) > (0):
                                    out18_: _dafny.Seq
                                    out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                    d_22_next_ = out18_
                                elif True:
                                    out19_: _dafny.Seq
                                    out19_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_22_next_ = out19_
                            elif True:
                                if (len(validTokenGroups)) > (0):
                                    out20_: _dafny.Seq
                                    out20_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                    d_22_next_ = out20_
                                elif True:
                                    out21_: _dafny.Seq
                                    out21_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_22_next_ = out21_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_25_appendedGenerated_: _dafny.Seq
                                d_26_appendedInside_: bool
                                d_27_appendedCurrent_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_25_appendedGenerated_ = out22_
                                d_26_appendedInside_ = out23_
                                d_27_appendedCurrent_ = out24_
                                generated = d_25_appendedGenerated_
                                insideConstrainedOut = d_26_appendedInside_
                                currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

