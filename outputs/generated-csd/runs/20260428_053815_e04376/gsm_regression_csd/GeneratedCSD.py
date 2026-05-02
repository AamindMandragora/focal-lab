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
        d_3_openSpan_: _dafny.Seq
        d_3_openSpan_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_4_preferredFlat_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_4_preferredFlat_ = out0_
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
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, d_3_openSpan_, eosToken)
                        d_6_chunkedGenerated_ = out1_
                        d_7_stoppedOpen_ = out2_
                        d_8_stoppedEos_ = out3_
                        d_9_stepsUsed_ = out4_
                        generated = d_6_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_10_complete_: bool
                        d_10_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_complete_:
                            d_11_closedGenerated_: _dafny.Seq
                            d_12_closedInside_: bool
                            d_13_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_closedGenerated_ = out5_
                            d_12_closedInside_ = out6_
                            d_13_closedCurrent_ = out7_
                            generated = d_11_closedGenerated_
                            insideConstrainedOut = d_12_closedInside_
                            currentConstrainedOut = d_13_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                            d_16_validCount_: int
                            out8_: int
                            out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_16_validCount_ = out8_
                            d_17_remainingBudget_: int
                            d_17_remainingBudget_ = (maxSteps) - (d_1_steps_)
                            d_18_symbolBudget_: int
                            d_18_symbolBudget_ = stepTokenBudget
                            if (d_17_remainingBudget_) < (d_18_symbolBudget_):
                                d_18_symbolBudget_ = d_17_remainingBudget_
                            if ((d_16_validCount_) > (d_2_narrowThreshold_)) and ((d_18_symbolBudget_) > (0)):
                                d_19_symbolOut_: _dafny.Seq
                                d_20_hitEos_: bool
                                d_21_stepsUsed_: int
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: int
                                out9_, out10_, out11_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, d_18_symbolBudget_, eosToken)
                                d_19_symbolOut_ = out9_
                                d_20_hitEos_ = out10_
                                d_21_stepsUsed_ = out11_
                                generated = (d_14_stablePrefix_) + (d_19_symbolOut_)
                                currentConstrainedOut = d_19_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_21_stepsUsed_)
                                if d_20_hitEos_:
                                    raise _dafny.Break("0")
                            elif True:
                                (lm).GenerateLogits((d_15_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(d_4_preferredFlat_)) > (0):
                                    d_22_candidates_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 20, eosToken)
                                    d_22_candidates_ = out12_
                                    d_23_preferred_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_22_candidates_, d_4_preferredFlat_)
                                    d_23_preferred_ = out13_
                                    if (len(d_23_preferred_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_23_preferred_, _dafny.BigRational('6e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_24_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (lm).ChooseNextToken()
                                d_24_next_ = out14_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_24_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_25_appendedGenerated_: _dafny.Seq
                                    d_26_appendedInside_: bool
                                    d_27_appendedCurrent_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                    d_25_appendedGenerated_ = out15_
                                    d_26_appendedInside_ = out16_
                                    d_27_appendedCurrent_ = out17_
                                    generated = d_25_appendedGenerated_
                                    insideConstrainedOut = d_26_appendedInside_
                                    currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

