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
        d_2_flatPreferred_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatPreferred_ = out0_
        d_3_extensionTokens_: _dafny.Seq
        d_3_extensionTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "RIGHT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FULL")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NOT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXISTS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "UNION")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INTERSECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXCEPT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIKE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BETWEEN"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkedGenerated_: _dafny.Seq
                        d_6_stoppedOpen_: bool
                        d_7_stoppedEos_: bool
                        d_8_stepsUsed_: int
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkedGenerated_ = out1_
                        d_6_stoppedOpen_ = out2_
                        d_7_stoppedEos_ = out3_
                        d_8_stepsUsed_ = out4_
                        generated = d_5_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_6_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_10_isComplete_: bool
                        d_10_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_isComplete_:
                            (lm).GenerateLogits((d_9_constrainedPrompt_) + (currentConstrainedOut))
                            d_11_extendable_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, 40, eosToken)
                            d_11_extendable_ = out5_
                            d_12_preferredExtensions_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_11_extendable_, d_3_extensionTokens_)
                            d_12_preferredExtensions_ = out6_
                            if (len(d_12_preferredExtensions_)) > (0):
                                (d_0_helpers_).BoostTokenLogits(lm, d_12_preferredExtensions_, _dafny.BigRational('12e0'))
                                if (len(d_2_flatPreferred_)) > (0):
                                    d_13_preferredAlsoValid_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out7_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_11_extendable_, d_2_flatPreferred_)
                                    d_13_preferredAlsoValid_ = out7_
                                    if (len(d_13_preferredAlsoValid_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_13_preferredAlsoValid_, _dafny.BigRational('3e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_14_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (lm).ChooseNextToken()
                                d_14_next_ = out8_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_15_nextValid_: bool
                                    out9_: bool
                                    out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_14_next_)
                                    d_15_nextValid_ = out9_
                                    if d_15_nextValid_:
                                        if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                            d_16_appendedGenerated_: _dafny.Seq
                                            d_17_appendedInside_: bool
                                            d_18_appendedCurrent_: _dafny.Seq
                                            out10_: _dafny.Seq
                                            out11_: bool
                                            out12_: _dafny.Seq
                                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                            d_16_appendedGenerated_ = out10_
                                            d_17_appendedInside_ = out11_
                                            d_18_appendedCurrent_ = out12_
                                            generated = d_16_appendedGenerated_
                                            insideConstrainedOut = d_17_appendedInside_
                                            currentConstrainedOut = d_18_appendedCurrent_
                                        elif True:
                                            d_19_closedGenerated_: _dafny.Seq
                                            d_20_closedInside_: bool
                                            d_21_closedCurrent_: _dafny.Seq
                                            out13_: _dafny.Seq
                                            out14_: bool
                                            out15_: _dafny.Seq
                                            out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_19_closedGenerated_ = out13_
                                            d_20_closedInside_ = out14_
                                            d_21_closedCurrent_ = out15_
                                            generated = d_19_closedGenerated_
                                            insideConstrainedOut = d_20_closedInside_
                                            currentConstrainedOut = d_21_closedCurrent_
                                    elif True:
                                        d_22_closedGenerated2_: _dafny.Seq
                                        d_23_closedInside2_: bool
                                        d_24_closedCurrent2_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: _dafny.Seq
                                        out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_22_closedGenerated2_ = out16_
                                        d_23_closedInside2_ = out17_
                                        d_24_closedCurrent2_ = out18_
                                        generated = d_22_closedGenerated2_
                                        insideConstrainedOut = d_23_closedInside2_
                                        currentConstrainedOut = d_24_closedCurrent2_
                            elif True:
                                d_25_closedGenerated3_: _dafny.Seq
                                d_26_closedInside3_: bool
                                d_27_closedCurrent3_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_25_closedGenerated3_ = out19_
                                d_26_closedInside3_ = out20_
                                d_27_closedCurrent3_ = out21_
                                generated = d_25_closedGenerated3_
                                insideConstrainedOut = d_26_closedInside3_
                                currentConstrainedOut = d_27_closedCurrent3_
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((d_9_constrainedPrompt_) + (currentConstrainedOut))
                            d_28_candidates_: _dafny.Seq
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, 50, eosToken)
                            d_28_candidates_ = out22_
                            d_29_structural_: _dafny.Seq
                            out23_: _dafny.Seq
                            out23_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_28_candidates_, d_3_extensionTokens_)
                            d_29_structural_ = out23_
                            if (len(d_29_structural_)) > (0):
                                (d_0_helpers_).BoostTokenLogits(lm, d_29_structural_, _dafny.BigRational('8e0'))
                            if (len(d_2_flatPreferred_)) > (0):
                                d_30_preferred_: _dafny.Seq
                                out24_: _dafny.Seq
                                out24_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_28_candidates_, d_2_flatPreferred_)
                                d_30_preferred_ = out24_
                                if (len(d_30_preferred_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_30_preferred_, _dafny.BigRational('3e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_31_next_: _dafny.Seq
                            out25_: _dafny.Seq
                            out25_ = (lm).ChooseNextToken()
                            d_31_next_ = out25_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_31_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_32_appendedGenerated2_: _dafny.Seq
                                d_33_appendedInside2_: bool
                                d_34_appendedCurrent2_: _dafny.Seq
                                out26_: _dafny.Seq
                                out27_: bool
                                out28_: _dafny.Seq
                                out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_next_)
                                d_32_appendedGenerated2_ = out26_
                                d_33_appendedInside2_ = out27_
                                d_34_appendedCurrent2_ = out28_
                                generated = d_32_appendedGenerated2_
                                insideConstrainedOut = d_33_appendedInside2_
                                currentConstrainedOut = d_34_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

