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
        d_3_openCount_: int
        d_3_openCount_ = 0
        if not(d_2_openedOnce_):
            out0_: int
            out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generatedPrefix, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            d_3_openCount_ = out0_
            if (d_3_openCount_) > (0):
                d_2_openedOnce_ = True
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_2_openedOnce_):
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out1_
                            d_5_openedInside_ = out2_
                            d_6_openedCurrent_ = out3_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_2_openedOnce_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_openedOnce_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedGenerated_: _dafny.Seq
                        d_9_closedInside_: bool
                        d_10_closedCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedGenerated_ = out5_
                        d_9_closedInside_ = out6_
                        d_10_closedCurrent_ = out7_
                        generated = d_8_closedGenerated_
                        insideConstrainedOut = d_9_closedInside_
                        currentConstrainedOut = d_10_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_11_deadEnd_: bool
                        out8_: bool
                        out8_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_11_deadEnd_ = out8_
                        if d_11_deadEnd_:
                            d_12_rolledGenerated_: _dafny.Seq
                            d_13_rolledCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: _dafny.Seq
                            out9_, out10_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_12_rolledGenerated_ = out9_
                            d_13_rolledCurrent_ = out10_
                            generated = d_12_rolledGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_13_rolledCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_15_validCount_: int
                            out11_: int
                            out11_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_15_validCount_ = out11_
                            d_16_next_: _dafny.Seq
                            d_16_next_ = eosToken
                            d_17_recentRepeat_: bool
                            d_17_recentRepeat_ = False
                            if (len(currentConstrainedOut)) > (0):
                                d_18_lastTok_: _dafny.Seq
                                d_18_lastTok_ = (generated)[(len(generated)) - (1)]
                                d_19_since_: int
                                out12_: int
                                out12_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, d_18_lastTok_)
                                d_19_since_ = out12_
                                if (d_19_since_) < (3):
                                    d_17_recentRepeat_ = True
                            if d_17_recentRepeat_:
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_16_next_ = out13_
                            elif (d_15_validCount_) <= (4):
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_16_next_ = out14_
                            elif (d_15_validCount_) <= (12):
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_16_next_ = out15_
                            elif True:
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_16_next_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_20_appendedGenerated_: _dafny.Seq
                                d_21_appendedInside_: bool
                                d_22_appendedCurrent_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                d_20_appendedGenerated_ = out17_
                                d_21_appendedInside_ = out18_
                                d_22_appendedCurrent_ = out19_
                                generated = d_20_appendedGenerated_
                                insideConstrainedOut = d_21_appendedInside_
                                currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

