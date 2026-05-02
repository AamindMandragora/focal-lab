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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 8
        d_3_arithmeticHints_: _dafny.Seq
        d_3_arithmeticHints_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkedGenerated_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkedGenerated_ = out0_
                        d_6_stoppedOnOpenSpan_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
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
                        elif True:
                            d_12_stablePrefix_: _dafny.Seq
                            d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                            d_14_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_14_validCount_ = out7_
                            d_15_arithmeticLike_: bool
                            d_15_arithmeticLike_ = False
                            if (len(currentConstrainedOut)) > (0):
                                d_16_hasOpen_: bool
                                d_16_hasOpen_ = VerifiedDecoderAgent.default__.Contains((currentConstrainedOut)[(len(currentConstrainedOut)) - (1)], _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                                d_15_arithmeticLike_ = not(d_16_hasOpen_)
                            if ((d_14_validCount_) <= (d_2_narrowThreshold_)) or (d_15_arithmeticLike_):
                                d_17_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_17_next_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_17_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_18_appendedGenerated_: _dafny.Seq
                                    d_19_appendedInside_: bool
                                    d_20_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                    d_18_appendedGenerated_ = out9_
                                    d_19_appendedInside_ = out10_
                                    d_20_appendedCurrent_ = out11_
                                    generated = d_18_appendedGenerated_
                                    insideConstrainedOut = d_19_appendedInside_
                                    currentConstrainedOut = d_20_appendedCurrent_
                            elif True:
                                (lm).GenerateLogits((d_13_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    d_21_flatPreferred_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_21_flatPreferred_ = out12_
                                    if (len(d_21_flatPreferred_)) > (0):
                                        d_22_anyPreferredValid_: bool
                                        out13_: bool
                                        out13_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_21_flatPreferred_)
                                        d_22_anyPreferredValid_ = out13_
                                        if d_22_anyPreferredValid_:
                                            d_23_candidates_: _dafny.Seq
                                            out14_: _dafny.Seq
                                            out14_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                            d_23_candidates_ = out14_
                                            d_24_preferredCandidates_: _dafny.Seq
                                            out15_: _dafny.Seq
                                            out15_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_23_candidates_, d_21_flatPreferred_)
                                            d_24_preferredCandidates_ = out15_
                                            if (len(d_24_preferredCandidates_)) > (0):
                                                (d_0_helpers_).BoostTokenLogits(lm, d_24_preferredCandidates_, _dafny.BigRational('4e0'))
                                d_25_anyArithmeticValid_: bool
                                out16_: bool
                                out16_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_3_arithmeticHints_)
                                d_25_anyArithmeticValid_ = out16_
                                if d_25_anyArithmeticValid_:
                                    d_26_topCandidates_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out17_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_26_topCandidates_ = out17_
                                    d_27_arithmeticCandidates_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out18_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_26_topCandidates_, d_3_arithmeticHints_)
                                    d_27_arithmeticCandidates_ = out18_
                                    if (len(d_27_arithmeticCandidates_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_27_arithmeticCandidates_, _dafny.BigRational('6e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_28_sampled_: _dafny.Seq
                                out19_: _dafny.Seq
                                out19_ = (lm).ChooseNextToken()
                                d_28_sampled_ = out19_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_28_sampled_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_29_appendedGenerated2_: _dafny.Seq
                                    d_30_appendedInside2_: bool
                                    d_31_appendedCurrent2_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_sampled_)
                                    d_29_appendedGenerated2_ = out20_
                                    d_30_appendedInside2_ = out21_
                                    d_31_appendedCurrent2_ = out22_
                                    generated = d_29_appendedGenerated2_
                                    insideConstrainedOut = d_30_appendedInside2_
                                    currentConstrainedOut = d_31_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

