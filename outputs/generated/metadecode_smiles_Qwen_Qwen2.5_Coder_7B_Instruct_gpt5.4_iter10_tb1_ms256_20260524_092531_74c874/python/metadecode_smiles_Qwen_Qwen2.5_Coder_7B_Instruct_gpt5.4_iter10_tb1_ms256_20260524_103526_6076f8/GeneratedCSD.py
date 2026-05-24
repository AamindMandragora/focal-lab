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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES string for the requested molecular class. Prefer chemically plausible continuations. If a << >> span is used, place the SMILES content inside that span and continue until the molecule is complete before closing.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_rollbackLimit_: int
        d_2_rollbackLimit_ = 64
        d_3_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_3_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out1_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_5_closedGenerated_: _dafny.Seq
                        d_6_closedInside_: bool
                        d_7_closedCurrent_: _dafny.Seq
                        out2_: _dafny.Seq
                        out3_: bool
                        out4_: _dafny.Seq
                        out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_5_closedGenerated_ = out2_
                        d_6_closedInside_ = out3_
                        d_7_closedCurrent_ = out4_
                        generated = d_5_closedGenerated_
                        insideConstrainedOut = d_6_closedInside_
                        currentConstrainedOut = d_7_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
                        d_8_rolledGenerated_: _dafny.Seq
                        d_9_rolledCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: _dafny.Seq
                        out5_, out6_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_8_rolledGenerated_ = out5_
                        d_9_rolledCurrent_ = out6_
                        generated = d_8_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_9_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_10_stablePrefix_: _dafny.Seq
                        d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                        d_12_next_: _dafny.Seq
                        d_12_next_ = eosToken
                        d_13_validCount_: int
                        out7_: int
                        out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_13_validCount_ = out7_
                        d_14_recentRepeat_: bool
                        d_14_recentRepeat_ = False
                        if ((len(currentConstrainedOut)) > (0)) and ((len(generated)) > (0)):
                            d_15_lastTok_: _dafny.Seq
                            d_15_lastTok_ = (generated)[(len(generated)) - (1)]
                            d_16_lastTokCount_: int = int(0)
                            out8_: int
                            out8_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, d_15_lastTok_)
                            d_16_lastTokCount_ = out8_
                            d_14_recentRepeat_ = (d_16_lastTokCount_) >= (2)
                        if d_14_recentRepeat_:
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_12_next_ = out9_
                        elif ((d_13_validCount_) <= (8)) and ((len(validTokenGroups)) > (0)):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_12_next_ = out10_
                        elif ((d_13_validCount_) <= (12)) and ((len(d_3_flatGroups_)) > (0)):
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_12_next_ = out11_
                        elif True:
                            d_17_nextSoft_: _dafny.Seq
                            d_18_usedFallback_: bool
                            out12_: _dafny.Seq
                            out13_: bool
                            out12_, out13_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_17_nextSoft_ = out12_
                            d_18_usedFallback_ = out13_
                            d_12_next_ = d_17_nextSoft_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_appendedGenerated_: _dafny.Seq
                            d_20_appendedInside_: bool
                            d_21_appendedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                            d_19_appendedGenerated_ = out14_
                            d_20_appendedInside_ = out15_
                            d_21_appendedCurrent_ = out16_
                            generated = d_19_appendedGenerated_
                            insideConstrainedOut = d_20_appendedInside_
                            currentConstrainedOut = d_21_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

