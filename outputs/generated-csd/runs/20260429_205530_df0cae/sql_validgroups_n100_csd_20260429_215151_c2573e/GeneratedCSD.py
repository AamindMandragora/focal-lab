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
                        d_7_isComplete_: bool
                        d_7_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_isComplete_:
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
                        elif True:
                            d_11_lastBeforeComma_: _dafny.Seq
                            d_12_foundComma_: bool
                            out7_: _dafny.Seq
                            out8_: bool
                            out7_, out8_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                            d_11_lastBeforeComma_ = out7_
                            d_12_foundComma_ = out8_
                            d_13_lastBeforeOpen_: _dafny.Seq
                            d_14_foundOpen_: bool
                            out9_: _dafny.Seq
                            out10_: bool
                            out9_, out10_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")))
                            d_13_lastBeforeOpen_ = out9_
                            d_14_foundOpen_ = out10_
                            d_15_suspiciousTail_: bool
                            d_15_suspiciousTail_ = False
                            if d_12_foundComma_:
                                if ((((d_11_lastBeforeComma_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")))) or ((d_11_lastBeforeComma_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND"))))) or ((d_11_lastBeforeComma_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN"))))) or ((d_11_lastBeforeComma_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")))):
                                    d_15_suspiciousTail_ = True
                            if (not(d_15_suspiciousTail_)) and (d_14_foundOpen_):
                                if ((((d_13_lastBeforeOpen_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")))) or ((d_13_lastBeforeOpen_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND"))))) or ((d_13_lastBeforeOpen_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN"))))) or ((d_13_lastBeforeOpen_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")))):
                                    d_15_suspiciousTail_ = True
                            d_16_deadEnd_: bool
                            out11_: bool
                            out11_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_16_deadEnd_ = out11_
                            if (d_16_deadEnd_) or (d_15_suspiciousTail_):
                                d_17_repaired_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")))
                                d_17_repaired_ = out12_
                                d_18_repaired2_: _dafny.Seq
                                d_18_repaired2_ = d_17_repaired_
                                if (len(d_18_repaired2_)) == (len(currentConstrainedOut)):
                                    out13_: _dafny.Seq
                                    out13_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                                    d_18_repaired2_ = out13_
                                if (len(d_18_repaired2_)) == (len(currentConstrainedOut)):
                                    out14_: _dafny.Seq
                                    out14_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                    d_18_repaired2_ = out14_
                                d_19_dropped_: int
                                d_19_dropped_ = (len(currentConstrainedOut)) - (len(d_18_repaired2_))
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_19_dropped_):])
                                currentConstrainedOut = d_18_repaired2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_20_stablePrefix_: _dafny.Seq
                                d_20_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_21_constrainedPrompt_: _dafny.Seq
                                d_21_constrainedPrompt_ = (prompt) + (d_20_stablePrefix_)
                                (lm).GenerateLogits((d_21_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    d_22_flatPreferred_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out15_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_22_flatPreferred_ = out15_
                                    if (len(d_22_flatPreferred_)) > (0):
                                        d_23_anyValidPreferred_: bool
                                        out16_: bool
                                        out16_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_22_flatPreferred_)
                                        d_23_anyValidPreferred_ = out16_
                                        if d_23_anyValidPreferred_:
                                            d_24_candidates_: _dafny.Seq
                                            out17_: _dafny.Seq
                                            out17_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, 12, eosToken)
                                            d_24_candidates_ = out17_
                                            d_25_preferred_: _dafny.Seq
                                            out18_: _dafny.Seq
                                            out18_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_24_candidates_, d_22_flatPreferred_)
                                            d_25_preferred_ = out18_
                                            if (len(d_25_preferred_)) > (0):
                                                (d_0_helpers_).BoostTokenLogits(lm, d_25_preferred_, _dafny.BigRational('15e-1'))
                                d_26_validCount_: int
                                out19_: int
                                out19_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_26_validCount_ = out19_
                                if (len(currentConstrainedOut)) >= (8):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('8e0'))
                                elif True:
                                    if ((len(currentConstrainedOut)) >= (4)) and ((d_26_validCount_) <= (3)):
                                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('4e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_27_next_: _dafny.Seq
                                out20_: _dafny.Seq
                                out20_ = (lm).ChooseNextToken()
                                d_27_next_ = out20_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_27_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_28_appendedGenerated_: _dafny.Seq
                                    d_29_appendedInside_: bool
                                    d_30_appendedCurrent_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                    d_28_appendedGenerated_ = out21_
                                    d_29_appendedInside_ = out22_
                                    d_30_appendedCurrent_ = out23_
                                    generated = d_28_appendedGenerated_
                                    insideConstrainedOut = d_29_appendedInside_
                                    currentConstrainedOut = d_30_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

