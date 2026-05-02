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
        d_2_narrowThreshold_ = 12
        d_3_wideCandidateThreshold_: int
        d_3_wideCandidateThreshold_ = 40
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
                        d_9_isComplete_: bool
                        d_9_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_isComplete_:
                            d_10_closedGenerated_: _dafny.Seq
                            d_11_closedInside_: bool
                            d_12_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_closedGenerated_ = out4_
                            d_11_closedInside_ = out5_
                            d_12_closedCurrent_ = out6_
                            generated = d_10_closedGenerated_
                            insideConstrainedOut = d_11_closedInside_
                            currentConstrainedOut = d_12_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_stablePrefix_: _dafny.Seq
                            d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                            d_15_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_15_validCount_ = out7_
                            if (d_15_validCount_) <= (d_2_narrowThreshold_):
                                d_16_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_16_next_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_16_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_17_appendedGenerated_: _dafny.Seq
                                    d_18_appendedInside_: bool
                                    d_19_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                    d_17_appendedGenerated_ = out9_
                                    d_18_appendedInside_ = out10_
                                    d_19_appendedCurrent_ = out11_
                                    generated = d_17_appendedGenerated_
                                    insideConstrainedOut = d_18_appendedInside_
                                    currentConstrainedOut = d_19_appendedCurrent_
                            elif True:
                                d_20_remainingBudget_: int
                                d_20_remainingBudget_ = (maxSteps) - (d_1_steps_)
                                d_21_symbolBudget_: int
                                d_21_symbolBudget_ = stepTokenBudget
                                if (d_21_symbolBudget_) == (0):
                                    d_21_symbolBudget_ = 1
                                if (d_20_remainingBudget_) < (d_21_symbolBudget_):
                                    d_21_symbolBudget_ = d_20_remainingBudget_
                                d_22_useSymbol_: bool
                                d_22_useSymbol_ = (d_21_symbolBudget_) > (1)
                                if d_22_useSymbol_:
                                    d_23_symbolOut_: _dafny.Seq
                                    d_24_hitEos_: bool
                                    d_25_symbolStepsUsed_: int
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: int
                                    out12_, out13_, out14_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, d_21_symbolBudget_, eosToken)
                                    d_23_symbolOut_ = out12_
                                    d_24_hitEos_ = out13_
                                    d_25_symbolStepsUsed_ = out14_
                                    generated = (d_13_stablePrefix_) + (d_23_symbolOut_)
                                    currentConstrainedOut = d_23_symbolOut_
                                    d_1_steps_ = (d_1_steps_) + (d_25_symbolStepsUsed_)
                                    if d_24_hitEos_:
                                        raise _dafny.Break("0")
                                elif True:
                                    (lm).GenerateLogits((d_14_constrainedPrompt_) + (currentConstrainedOut))
                                    if (len(validTokenGroups)) > (0):
                                        d_26_flatPreferred_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out15_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                        d_26_flatPreferred_ = out15_
                                        if (len(d_26_flatPreferred_)) > (0):
                                            d_27_anyValidPreferred_: bool
                                            out16_: bool
                                            out16_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_26_flatPreferred_)
                                            d_27_anyValidPreferred_ = out16_
                                            if d_27_anyValidPreferred_:
                                                d_28_candidates_: _dafny.Seq
                                                out17_: _dafny.Seq
                                                out17_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, d_3_wideCandidateThreshold_, eosToken)
                                                d_28_candidates_ = out17_
                                                d_29_preferred_: _dafny.Seq
                                                out18_: _dafny.Seq
                                                out18_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_28_candidates_, d_26_flatPreferred_)
                                                d_29_preferred_ = out18_
                                                if (len(d_29_preferred_)) > (0):
                                                    (d_0_helpers_).BoostTokenLogits(lm, d_29_preferred_, _dafny.BigRational('5e0'))
                                    (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                    d_30_nextWide_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out19_ = (lm).ChooseNextToken()
                                    d_30_nextWide_ = out19_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_30_nextWide_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_31_appendedGeneratedWide_: _dafny.Seq
                                        d_32_appendedInsideWide_: bool
                                        d_33_appendedCurrentWide_: _dafny.Seq
                                        out20_: _dafny.Seq
                                        out21_: bool
                                        out22_: _dafny.Seq
                                        out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_nextWide_)
                                        d_31_appendedGeneratedWide_ = out20_
                                        d_32_appendedInsideWide_ = out21_
                                        d_33_appendedCurrentWide_ = out22_
                                        generated = d_31_appendedGeneratedWide_
                                        insideConstrainedOut = d_32_appendedInsideWide_
                                        currentConstrainedOut = d_33_appendedCurrentWide_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

