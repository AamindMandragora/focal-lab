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
        d_3_fromKeyword_: _dafny.Seq
        d_3_fromKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_4_fromContext_: _dafny.Seq
        d_4_fromContext_ = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_chunkBudget_: int
                        d_5_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_6_chunkedGenerated_: _dafny.Seq
                        d_7_stoppedOpen_: bool
                        d_8_stoppedEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkedGenerated_ = out0_
                        d_7_stoppedOpen_ = out1_
                        d_8_stoppedEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_4_fromContext_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_10_isComplete_: bool
                        d_10_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_isComplete_:
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
                            d_4_fromContext_ = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            out7_: _dafny.Seq
                            out7_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_fromKeyword_)
                            d_4_fromContext_ = out7_
                            d_14_validCount_: int
                            out8_: int
                            out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_14_validCount_ = out8_
                            d_15_stablePrefix_: _dafny.Seq
                            d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_16_constrainedPrompt_: _dafny.Seq
                            d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                            if (d_14_validCount_) <= (d_2_narrowThreshold_):
                                d_17_next_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_17_next_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_17_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_18_appendedGenerated_: _dafny.Seq
                                    d_19_appendedInside_: bool
                                    d_20_appendedCurrent_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                    d_18_appendedGenerated_ = out10_
                                    d_19_appendedInside_ = out11_
                                    d_20_appendedCurrent_ = out12_
                                    generated = d_18_appendedGenerated_
                                    insideConstrainedOut = d_19_appendedInside_
                                    currentConstrainedOut = d_20_appendedCurrent_
                            elif True:
                                (lm).GenerateLogits((d_16_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    d_21_flatPreferred_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_21_flatPreferred_ = out13_
                                    if (len(d_21_flatPreferred_)) > (0):
                                        d_22_preferredTop_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out14_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                        d_22_preferredTop_ = out14_
                                        d_23_preferredOverlap_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out15_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_22_preferredTop_, d_21_flatPreferred_)
                                        d_23_preferredOverlap_ = out15_
                                        if (len(d_23_preferredOverlap_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_23_preferredOverlap_, _dafny.BigRational('5e0'))
                                if (len(d_4_fromContext_)) > (0):
                                    d_24_scopedTop_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out16_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_24_scopedTop_ = out16_
                                    d_25_scopedOverlap_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out17_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_24_scopedTop_, d_4_fromContext_)
                                    d_25_scopedOverlap_ = out17_
                                    if (len(d_25_scopedOverlap_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_25_scopedOverlap_, _dafny.BigRational('6e0'))
                                d_26_symbolBudget_: int
                                d_26_symbolBudget_ = stepTokenBudget
                                if (d_26_symbolBudget_) > ((maxSteps) - (d_1_steps_)):
                                    d_26_symbolBudget_ = (maxSteps) - (d_1_steps_)
                                if (d_26_symbolBudget_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_27_symbolOut_: _dafny.Seq
                                    d_28_hitEos_: bool
                                    d_29_stepsUsed2_: int
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: int
                                    out18_, out19_, out20_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, d_26_symbolBudget_, eosToken)
                                    d_27_symbolOut_ = out18_
                                    d_28_hitEos_ = out19_
                                    d_29_stepsUsed2_ = out20_
                                    generated = (d_15_stablePrefix_) + (d_27_symbolOut_)
                                    currentConstrainedOut = d_27_symbolOut_
                                    d_1_steps_ = (d_1_steps_) + (d_29_stepsUsed2_)
                                    if d_28_hitEos_:
                                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

