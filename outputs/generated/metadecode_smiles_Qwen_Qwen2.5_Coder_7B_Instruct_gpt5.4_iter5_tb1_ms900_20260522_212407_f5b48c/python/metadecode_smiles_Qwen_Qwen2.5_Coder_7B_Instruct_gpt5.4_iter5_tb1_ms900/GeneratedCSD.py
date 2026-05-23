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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "When producing the constrained molecular string, prefer chemically plausible SMILES continuations for the requested class while preserving exact parser-valid output.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 64
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
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
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_5_closedGenerated_ = out1_
                        d_6_closedInside_ = out2_
                        d_7_closedCurrent_ = out3_
                        generated = d_5_closedGenerated_
                        insideConstrainedOut = d_6_closedInside_
                        currentConstrainedOut = d_7_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                        d_8_rolledGenerated_: _dafny.Seq
                        d_9_rolledCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: _dafny.Seq
                        out4_, out5_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_8_rolledGenerated_ = out4_
                        d_9_rolledCurrent_ = out5_
                        generated = d_8_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_9_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_10_stablePrefix_: _dafny.Seq
                        d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                        d_12_validCount_: int
                        out6_: int
                        out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_12_validCount_ = out6_
                        d_13_recentRepeat_: bool
                        d_13_recentRepeat_ = False
                        if (len(currentConstrainedOut)) > (0):
                            d_14_lastTok_: _dafny.Seq
                            d_14_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                            d_15_since_: int
                            out7_: int
                            out7_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(currentConstrainedOut, d_14_lastTok_)
                            d_15_since_ = out7_
                            d_13_recentRepeat_ = (d_15_since_) <= (1)
                        if ((stepTokenBudget) > (1)) and ((d_12_validCount_) > (d_2_narrowThreshold_)):
                            d_16_remaining_: int
                            d_16_remaining_ = (maxSteps) - (d_1_steps_)
                            d_17_symbolBudget_: int
                            if (stepTokenBudget) > (d_16_remaining_):
                                d_17_symbolBudget_ = d_16_remaining_
                            elif True:
                                d_17_symbolBudget_ = stepTokenBudget
                            d_18_symbolGenerated_: _dafny.Seq
                            d_19_symbolOut_: _dafny.Seq
                            d_20_hitEos_: bool
                            d_21_stepsUsed_: int
                            out8_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: int
                            out8_, out9_, out10_, out11_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_11_constrainedPrompt_, generated, currentConstrainedOut, d_17_symbolBudget_, eosToken)
                            d_18_symbolGenerated_ = out8_
                            d_19_symbolOut_ = out9_
                            d_20_hitEos_ = out10_
                            d_21_stepsUsed_ = out11_
                            generated = d_18_symbolGenerated_
                            currentConstrainedOut = d_19_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_21_stepsUsed_)
                            if d_20_hitEos_:
                                raise _dafny.Break("0")
                        elif True:
                            d_22_nextIn_: _dafny.Seq
                            d_22_nextIn_ = eosToken
                            if d_13_recentRepeat_:
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_22_nextIn_ = out12_
                            elif (d_12_validCount_) <= (d_2_narrowThreshold_):
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_22_nextIn_ = out13_
                            elif True:
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_22_nextIn_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_nextIn_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_23_appendedGenerated_: _dafny.Seq
                                d_24_appendedInside_: bool
                                d_25_appendedCurrent_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextIn_)
                                d_23_appendedGenerated_ = out15_
                                d_24_appendedInside_ = out16_
                                d_25_appendedCurrent_ = out17_
                                generated = d_23_appendedGenerated_
                                insideConstrainedOut = d_24_appendedInside_
                                currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

