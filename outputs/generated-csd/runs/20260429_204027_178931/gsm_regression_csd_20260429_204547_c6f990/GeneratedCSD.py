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
        d_2_preferredFlat_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_preferredFlat_ = out0_
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
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkedGenerated_ = out1_
                        d_5_stoppedOnOpenSpan_ = out2_
                        d_6_stoppedOnEos_ = out3_
                        d_7_stepsUsed_ = out4_
                        generated = d_4_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_5_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
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
                            d_11_remaining_: int
                            d_11_remaining_ = (maxSteps) - (d_1_steps_)
                            d_12_narrow_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_12_narrow_ = out8_
                            if (d_12_narrow_) or ((d_11_remaining_) <= (1)):
                                d_13_repaired_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_13_repaired_ = out9_
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_13_repaired_))):])
                                currentConstrainedOut = d_13_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_14_closedGenerated2_: _dafny.Seq
                                    d_15_closedInside2_: bool
                                    d_16_closedCurrent2_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_14_closedGenerated2_ = out10_
                                    d_15_closedInside2_ = out11_
                                    d_16_closedCurrent2_ = out12_
                                    generated = d_14_closedGenerated2_
                                    insideConstrainedOut = d_15_closedInside2_
                                    currentConstrainedOut = d_16_closedCurrent2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_17_stablePrefix_: _dafny.Seq
                                d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                (lm).GenerateLogits(((prompt) + (d_17_stablePrefix_)) + (currentConstrainedOut))
                                d_18_count_: int
                                out13_: int
                                out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_18_count_ = out13_
                                if (d_18_count_) > (0):
                                    d_19_arithmeticHints_: _dafny.Seq
                                    d_19_arithmeticHints_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))])
                                    d_20_favored_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_2_preferredFlat_, d_19_arithmeticHints_)
                                    d_20_favored_ = out14_
                                    if (len(d_20_favored_)) > (0):
                                        d_21_anyFavoredValid_: bool
                                        out15_: bool
                                        out15_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_20_favored_)
                                        d_21_anyFavoredValid_ = out15_
                                        if d_21_anyFavoredValid_:
                                            (d_0_helpers_).BoostTokenLogits(lm, d_20_favored_, _dafny.BigRational('8e0'))
                                    elif True:
                                        d_22_anyHintValid_: bool
                                        out16_: bool
                                        out16_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_19_arithmeticHints_)
                                        d_22_anyHintValid_ = out16_
                                        if d_22_anyHintValid_:
                                            (d_0_helpers_).BoostTokenLogits(lm, d_19_arithmeticHints_, _dafny.BigRational('5e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_23_next_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (lm).ChooseNextToken()
                                d_23_next_ = out17_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_23_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_24_appendedGenerated_: _dafny.Seq
                                    d_25_appendedInside_: bool
                                    d_26_appendedCurrent_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                    d_24_appendedGenerated_ = out18_
                                    d_25_appendedInside_ = out19_
                                    d_26_appendedCurrent_ = out20_
                                    generated = d_24_appendedGenerated_
                                    insideConstrainedOut = d_25_appendedInside_
                                    currentConstrainedOut = d_26_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

