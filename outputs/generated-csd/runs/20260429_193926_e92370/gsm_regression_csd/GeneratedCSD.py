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
        d_2_wideThreshold_: int
        d_2_wideThreshold_ = 12
        d_3_eqToken_: _dafny.Seq
        d_3_eqToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))
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
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkedGenerated_ = out0_
                        d_6_stoppedOpen_ = out1_
                        d_7_stoppedEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOpen_:
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
                            d_15_rhsContext_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_eqToken_)
                            d_15_rhsContext_ = out8_
                            if (((d_14_validCount_) > (d_2_wideThreshold_)) and ((stepTokenBudget) > (0))) and ((stepTokenBudget) <= ((maxSteps) - (d_1_steps_))):
                                d_16_symbolOut_: _dafny.Seq
                                d_17_hitEos_: bool
                                d_18_stepsUsed_: int
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: int
                                out9_, out10_, out11_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, stepTokenBudget, eosToken)
                                d_16_symbolOut_ = out9_
                                d_17_hitEos_ = out10_
                                d_18_stepsUsed_ = out11_
                                generated = (d_12_stablePrefix_) + (d_16_symbolOut_)
                                currentConstrainedOut = d_16_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_18_stepsUsed_)
                                if d_17_hitEos_:
                                    raise _dafny.Break("0")
                            elif True:
                                (lm).GenerateLogits((d_13_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(d_15_rhsContext_)) > (0):
                                    d_19_candidates_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 20, eosToken)
                                    d_19_candidates_ = out12_
                                    d_20_focused_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_19_candidates_, d_15_rhsContext_)
                                    d_20_focused_ = out13_
                                    if (len(d_20_focused_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_20_focused_, _dafny.BigRational('6e0'))
                                if (len(validTokenGroups)) > (0):
                                    d_21_flatPreferred_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_21_flatPreferred_ = out14_
                                    if (len(d_21_flatPreferred_)) > (0):
                                        d_22_preferredCandidates_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out15_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 20, eosToken)
                                        d_22_preferredCandidates_ = out15_
                                        d_23_overlap_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out16_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_22_preferredCandidates_, d_21_flatPreferred_)
                                        d_23_overlap_ = out16_
                                        if (len(d_23_overlap_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_23_overlap_, _dafny.BigRational('3e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_24_next_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (lm).ChooseNextToken()
                                d_24_next_ = out17_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_24_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_25_appendedGenerated_: _dafny.Seq
                                    d_26_appendedInside_: bool
                                    d_27_appendedCurrent_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                    d_25_appendedGenerated_ = out18_
                                    d_26_appendedInside_ = out19_
                                    d_27_appendedCurrent_ = out20_
                                    generated = d_25_appendedGenerated_
                                    insideConstrainedOut = d_26_appendedInside_
                                    currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

