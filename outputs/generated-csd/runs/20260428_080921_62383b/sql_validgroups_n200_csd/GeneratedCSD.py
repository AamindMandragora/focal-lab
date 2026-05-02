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
        d_3_flatPreferred_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_3_flatPreferred_ = out0_
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
                        elif d_6_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out5_
                            d_10_closedInside_ = out6_
                            d_11_closedCurrent_ = out7_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_constrainedPrompt_: _dafny.Seq
                            d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_13_validCount_: int
                            out8_: int
                            out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_13_validCount_ = out8_
                            if (d_13_validCount_) <= (d_2_narrowThreshold_):
                                d_14_next_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_14_next_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_15_appendedGenerated_: _dafny.Seq
                                    d_16_appendedInside_: bool
                                    d_17_appendedCurrent_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                    d_15_appendedGenerated_ = out10_
                                    d_16_appendedInside_ = out11_
                                    d_17_appendedCurrent_ = out12_
                                    generated = d_15_appendedGenerated_
                                    insideConstrainedOut = d_16_appendedInside_
                                    currentConstrainedOut = d_17_appendedCurrent_
                            elif True:
                                if (stepTokenBudget) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    (lm).GenerateLogits((d_12_constrainedPrompt_) + (currentConstrainedOut))
                                    if (len(d_3_flatPreferred_)) > (0):
                                        d_18_candidates_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out13_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                        d_18_candidates_ = out13_
                                        d_19_preferred_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out14_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_18_candidates_, d_3_flatPreferred_)
                                        d_19_preferred_ = out14_
                                        if (len(d_19_preferred_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_19_preferred_, _dafny.BigRational('3e0'))
                                        d_20_prevTok_: _dafny.Seq
                                        d_21_foundPrev_: bool
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out15_, out16_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")))
                                        d_20_prevTok_ = out15_
                                        d_21_foundPrev_ = out16_
                                        if d_21_foundPrev_:
                                            d_22_activeIdx_: int
                                            out17_: int
                                            out17_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_20_prevTok_)
                                            d_22_activeIdx_ = out17_
                                            if (d_22_activeIdx_) >= (0):
                                                d_23_activePreferred_: _dafny.Seq
                                                out18_: _dafny.Seq
                                                out18_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_18_candidates_, (validTokenGroups)[d_22_activeIdx_])
                                                d_23_activePreferred_ = out18_
                                                if (len(d_23_activePreferred_)) > (0):
                                                    (d_0_helpers_).BoostTokenLogits(lm, d_23_activePreferred_, _dafny.BigRational('8e0'))
                                    d_24_stablePrefix_: _dafny.Seq
                                    d_24_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_25_remainingBudget_: int
                                    d_25_remainingBudget_ = (maxSteps) - (d_1_steps_)
                                    d_26_localBudget_: int
                                    d_26_localBudget_ = stepTokenBudget
                                    if (d_25_remainingBudget_) < (d_26_localBudget_):
                                        d_26_localBudget_ = d_25_remainingBudget_
                                    d_27_symbolOut_: _dafny.Seq
                                    d_28_hitEos_: bool
                                    d_29_stepsUsed2_: int
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: int
                                    out19_, out20_, out21_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, d_26_localBudget_, eosToken)
                                    d_27_symbolOut_ = out19_
                                    d_28_hitEos_ = out20_
                                    d_29_stepsUsed2_ = out21_
                                    generated = (d_24_stablePrefix_) + (d_27_symbolOut_)
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

