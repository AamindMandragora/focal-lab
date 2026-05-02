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
        d_2_repairedOnce_: bool
        d_2_repairedOnce_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_chunkBudget_: int
                        d_3_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkedGenerated_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkedGenerated_ = out0_
                        d_5_stoppedOnOpenSpan_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        generated = d_4_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        cost = d_1_steps_
                        d_2_repairedOnce_ = False
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_2_repairedOnce_ = False
                    elif True:
                        d_8_completeNow_: bool
                        d_8_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_completeNow_:
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out4_
                            d_10_closedInside_ = out5_
                            d_11_closedCurrent_ = out6_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            cost = d_1_steps_
                            d_2_repairedOnce_ = False
                        elif True:
                            d_12_deadEnd_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_12_deadEnd_ = out7_
                            if (d_12_deadEnd_) and (not(d_2_repairedOnce_)):
                                d_13_stablePrefix_: _dafny.Seq
                                d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_14_repaired_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                                d_14_repaired_ = out8_
                                if (len(d_14_repaired_)) == (len(currentConstrainedOut)):
                                    out9_: _dafny.Seq
                                    out9_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))
                                    d_14_repaired_ = out9_
                                if (len(d_14_repaired_)) == (len(currentConstrainedOut)):
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")))
                                    d_14_repaired_ = out10_
                                if (len(d_14_repaired_)) == (len(currentConstrainedOut)):
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                    d_14_repaired_ = out11_
                                generated = (d_13_stablePrefix_) + (d_14_repaired_)
                                currentConstrainedOut = d_14_repaired_
                                d_2_repairedOnce_ = True
                            elif True:
                                d_2_repairedOnce_ = False
                                d_15_constrainedPrompt_: _dafny.Seq
                                d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                (lm).GenerateLogits((d_15_constrainedPrompt_) + (currentConstrainedOut))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                if (len(validTokenGroups)) > (0):
                                    d_16_flatPreferred_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_16_flatPreferred_ = out12_
                                    if (len(d_16_flatPreferred_)) > (0):
                                        d_17_flatSafe_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out13_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_flatPreferred_, (lm).Tokens)
                                        d_17_flatSafe_ = out13_
                                        if (len(d_17_flatSafe_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_17_flatSafe_, _dafny.BigRational('3e0'))
                                if (len(currentConstrainedOut)) > (0):
                                    d_18_lastTok_: _dafny.Seq
                                    d_18_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                                    if (d_18_lastTok_) in ((lm).Tokens):
                                        (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_18_lastTok_]), _dafny.BigRational('15e-1'))
                                d_19_repeatedAfterCommaTok_: _dafny.Seq
                                d_20_foundBeforeComma_: bool
                                out14_: _dafny.Seq
                                out15_: bool
                                out14_, out15_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                                d_19_repeatedAfterCommaTok_ = out14_
                                d_20_foundBeforeComma_ = out15_
                                if d_20_foundBeforeComma_:
                                    if (d_19_repeatedAfterCommaTok_) in ((lm).Tokens):
                                        (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_19_repeatedAfterCommaTok_]), _dafny.BigRational('4e0'))
                                (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('1e0'))
                                d_21_next_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (lm).ChooseNextToken()
                                d_21_next_ = out16_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                cost = d_1_steps_
                                if (d_21_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_validNext_: bool
                                    out17_: bool
                                    out17_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_21_next_)
                                    d_22_validNext_ = out17_
                                    if d_22_validNext_:
                                        d_23_appendedGenerated_: _dafny.Seq
                                        d_24_appendedInside_: bool
                                        d_25_appendedCurrent_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                        d_23_appendedGenerated_ = out18_
                                        d_24_appendedInside_ = out19_
                                        d_25_appendedCurrent_ = out20_
                                        generated = d_23_appendedGenerated_
                                        insideConstrainedOut = d_24_appendedInside_
                                        currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

