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
        if (maxSteps) == (0):
            cost = 0
        elif True:
            (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output a valid SMILES string for the requested molecular class. Use one constrained << >> span for the SMILES itself; once the span opens, continue the molecule until complete, then close it.")))
            d_1_steps_: int
            d_1_steps_ = 1
            d_2_openedAnySpan_: bool
            d_2_openedAnySpan_ = insideConstrained
            d_3_rollbackLimit_: int
            d_3_rollbackLimit_ = 64
            d_4_flatGroups_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
            d_4_flatGroups_ = out0_
            with _dafny.label("1_0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("1_0"):
                        if not(insideConstrainedOut):
                            if d_2_openedAnySpan_:
                                raise _dafny.Break("1_0")
                            elif True:
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
                                d_2_openedAnySpan_ = True
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
                        elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                            d_11_rolledGenerated_: _dafny.Seq
                            d_12_rolledCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: _dafny.Seq
                            out7_, out8_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_11_rolledGenerated_ = out7_
                            d_12_rolledCurrent_ = out8_
                            generated = d_11_rolledGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_12_rolledCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_stablePrefix_: _dafny.Seq
                            d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                            d_15_next_: _dafny.Seq
                            d_15_next_ = eosToken
                            d_16_validCount_: int
                            out9_: int
                            out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_16_validCount_ = out9_
                            d_17_recentRepeat_: bool
                            d_17_recentRepeat_ = False
                            if ((len(currentConstrainedOut)) > (0)) and ((len(generated)) > (0)):
                                d_18_lastTok_: _dafny.Seq
                                d_18_lastTok_ = (generated)[(len(generated)) - (1)]
                                d_19_lastTokCount_: int = int(0)
                                out10_: int
                                out10_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, d_18_lastTok_)
                                d_19_lastTokCount_ = out10_
                                d_17_recentRepeat_ = (d_19_lastTokCount_) >= (2)
                            if d_17_recentRepeat_:
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_15_next_ = out11_
                            elif ((d_16_validCount_) <= (8)) and ((len(validTokenGroups)) > (0)):
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_15_next_ = out12_
                            elif ((d_16_validCount_) <= (12)) and ((len(d_4_flatGroups_)) > (0)):
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_15_next_ = out13_
                            elif True:
                                d_20_nextSoft_: _dafny.Seq
                                d_21_usedFallback_: bool
                                out14_: _dafny.Seq
                                out15_: bool
                                out14_, out15_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                                d_20_nextSoft_ = out14_
                                d_21_usedFallback_ = out15_
                                d_15_next_ = d_20_nextSoft_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_15_next_) == (eosToken):
                                raise _dafny.Break("1_0")
                            elif True:
                                d_22_appendedGenerated_: _dafny.Seq
                                d_23_appendedInside_: bool
                                d_24_appendedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                d_22_appendedGenerated_ = out16_
                                d_23_appendedInside_ = out17_
                                d_24_appendedCurrent_ = out18_
                                generated = d_22_appendedGenerated_
                                insideConstrainedOut = d_23_appendedInside_
                                currentConstrainedOut = d_24_appendedCurrent_
                        pass
                pass
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

