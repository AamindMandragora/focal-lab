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
        d_2_openSpanToken_: _dafny.Seq
        d_2_openSpanToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_3_fromKeyword_: _dafny.Seq
        d_3_fromKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_4_flatPreferred_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_4_flatPreferred_ = out0_
        d_5_sqlBias_: _dafny.Seq
        d_5_sqlBias_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))])
        d_6_schemaFocus_: _dafny.Seq
        d_6_schemaFocus_ = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_openedGenerated_: _dafny.Seq
                        d_8_openedInside_: bool
                        d_9_openedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_7_openedGenerated_ = out1_
                        d_8_openedInside_ = out2_
                        d_9_openedCurrent_ = out3_
                        generated = d_7_openedGenerated_
                        insideConstrainedOut = d_8_openedInside_
                        currentConstrainedOut = d_9_openedCurrent_
                        d_6_schemaFocus_ = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                        cost = d_1_steps_
                    elif True:
                        d_10_completeNow_: bool
                        d_10_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_completeNow_:
                            d_11_closedGenerated_: _dafny.Seq
                            d_12_closedInside_: bool
                            d_13_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_closedGenerated_ = out4_
                            d_12_closedInside_ = out5_
                            d_13_closedCurrent_ = out6_
                            generated = d_11_closedGenerated_
                            insideConstrainedOut = d_12_closedInside_
                            currentConstrainedOut = d_13_closedCurrent_
                            d_6_schemaFocus_ = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                            cost = d_1_steps_
                            raise _dafny.Break("0")
                        elif True:
                            out7_: _dafny.Seq
                            out7_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_fromKeyword_)
                            d_6_schemaFocus_ = out7_
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                            d_16_validCount_: int
                            out8_: int
                            out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_16_validCount_ = out8_
                            if (((stepTokenBudget) == (0)) or (((d_1_steps_) + (stepTokenBudget)) > (maxSteps))) or ((d_16_validCount_) <= (8)):
                                (lm).GenerateLogits((d_15_constrainedPrompt_) + (currentConstrainedOut))
                                d_17_candidates_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 32, eosToken)
                                d_17_candidates_ = out9_
                                d_18_structural_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_17_candidates_, d_5_sqlBias_)
                                d_18_structural_ = out10_
                                if (len(d_18_structural_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_18_structural_, _dafny.BigRational('6e0'))
                                if (len(d_4_flatPreferred_)) > (0):
                                    d_19_preferred_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_17_candidates_, d_4_flatPreferred_)
                                    d_19_preferred_ = out11_
                                    if (len(d_19_preferred_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_19_preferred_, _dafny.BigRational('3e0'))
                                if (len(d_6_schemaFocus_)) > (0):
                                    d_20_focused_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_17_candidates_, d_6_schemaFocus_)
                                    d_20_focused_ = out12_
                                    if (len(d_20_focused_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_20_focused_, _dafny.BigRational('4e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_21_next_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (lm).ChooseNextToken()
                                d_21_next_ = out13_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                cost = d_1_steps_
                                if (d_21_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_appendedGenerated_: _dafny.Seq
                                    d_23_appendedInside_: bool
                                    d_24_appendedCurrent_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_22_appendedGenerated_ = out14_
                                    d_23_appendedInside_ = out15_
                                    d_24_appendedCurrent_ = out16_
                                    generated = d_22_appendedGenerated_
                                    insideConstrainedOut = d_23_appendedInside_
                                    currentConstrainedOut = d_24_appendedCurrent_
                            elif True:
                                d_25_symbolOut_: _dafny.Seq
                                d_26_hitEos_: bool
                                d_27_symbolStepsUsed_: int
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: int
                                out17_, out18_, out19_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, stepTokenBudget, eosToken)
                                d_25_symbolOut_ = out17_
                                d_26_hitEos_ = out18_
                                d_27_symbolStepsUsed_ = out19_
                                generated = (d_14_stablePrefix_) + (d_25_symbolOut_)
                                currentConstrainedOut = d_25_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_27_symbolStepsUsed_)
                                cost = d_1_steps_
                                if d_26_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

