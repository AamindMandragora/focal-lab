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
        d_2_clauseTokens_: _dafny.Seq
        d_2_clauseTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ASC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DESC")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NOT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXISTS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MAX")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT"))])
        d_3_structureTokens_: _dafny.Seq
        d_3_structureTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))])
        d_4_boostTokens_: _dafny.Seq = _dafny.Seq({})
        d_5_topCandidates_: _dafny.Seq = _dafny.Seq({})
        d_6_keywordBias_: _dafny.Seq = _dafny.Seq({})
        d_7_punctBias_: _dafny.Seq = _dafny.Seq({})
        d_8_completeNow_: bool = False
        d_9_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_10_constrainedPrompt_: _dafny.Seq = _dafny.Seq({})
        d_11_stablePrefix_: _dafny.Seq = _dafny.Seq({})
        d_12_chunkBudget_: int = int(0)
        d_13_chunkedGenerated_: _dafny.Seq = _dafny.Seq({})
        d_14_stoppedOnOpenSpan_: bool = False
        d_15_stoppedOnEos_: bool = False
        d_16_stepsUsed_: int = int(0)
        d_17_closedGenerated_: _dafny.Seq = _dafny.Seq({})
        d_18_closedInside_: bool = False
        d_19_closedCurrent_: _dafny.Seq = _dafny.Seq({})
        d_20_appendedGenerated_: _dafny.Seq = _dafny.Seq({})
        d_21_appendedInside_: bool = False
        d_22_appendedCurrent_: _dafny.Seq = _dafny.Seq({})
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_12_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_12_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_13_chunkedGenerated_ = out0_
                        d_14_stoppedOnOpenSpan_ = out1_
                        d_15_stoppedOnEos_ = out2_
                        d_16_stepsUsed_ = out3_
                        generated = d_13_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_16_stepsUsed_)
                        if d_15_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_14_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_8_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_completeNow_:
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_17_closedGenerated_ = out4_
                            d_18_closedInside_ = out5_
                            d_19_closedCurrent_ = out6_
                            generated = d_17_closedGenerated_
                            insideConstrainedOut = d_18_closedInside_
                            currentConstrainedOut = d_19_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_10_constrainedPrompt_ = (prompt) + (d_11_stablePrefix_)
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                            d_5_topCandidates_ = out7_
                            out8_: _dafny.Seq
                            out8_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_5_topCandidates_, d_2_clauseTokens_)
                            d_6_keywordBias_ = out8_
                            out9_: _dafny.Seq
                            out9_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_5_topCandidates_, d_3_structureTokens_)
                            d_7_punctBias_ = out9_
                            out10_: _dafny.Seq
                            out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_5_topCandidates_, d_5_topCandidates_)
                            d_4_boostTokens_ = out10_
                            (lm).GenerateLogits((d_10_constrainedPrompt_) + (currentConstrainedOut))
                            if (len(d_6_keywordBias_)) > (0):
                                (d_0_helpers_).BoostTokenLogits(lm, d_6_keywordBias_, _dafny.BigRational('8e0'))
                            if (len(d_7_punctBias_)) > (0):
                                (d_0_helpers_).BoostTokenLogits(lm, d_7_punctBias_, _dafny.BigRational('3e0'))
                            if (len(d_4_boostTokens_)) > (0):
                                (d_0_helpers_).BoostTokenLogits(lm, d_4_boostTokens_, _dafny.BigRational('15e-1'))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('12e0'))
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_9_next_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                                d_20_appendedGenerated_ = out12_
                                d_21_appendedInside_ = out13_
                                d_22_appendedCurrent_ = out14_
                                generated = d_20_appendedGenerated_
                                insideConstrainedOut = d_21_appendedInside_
                                currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

