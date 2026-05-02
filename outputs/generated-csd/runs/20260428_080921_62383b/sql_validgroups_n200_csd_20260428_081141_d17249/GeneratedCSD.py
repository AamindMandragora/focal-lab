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
        d_3_flatPreferred_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_3_flatPreferred_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_openedGenerated_: _dafny.Seq
                        d_5_openedInside_: bool
                        d_6_openedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_4_openedGenerated_ = out1_
                        d_5_openedInside_ = out2_
                        d_6_openedCurrent_ = out3_
                        generated = d_4_openedGenerated_
                        insideConstrainedOut = d_5_openedInside_
                        currentConstrainedOut = d_6_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_7_closedGenerated_: _dafny.Seq
                            d_8_closedInside_: bool
                            d_9_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_closedGenerated_ = out4_
                            d_8_closedInside_ = out5_
                            d_9_closedCurrent_ = out6_
                            generated = d_7_closedGenerated_
                            insideConstrainedOut = d_8_closedInside_
                            currentConstrainedOut = d_9_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_10_constrainedPrompt_: _dafny.Seq
                            d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_11_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_11_validCount_ = out7_
                            if ((d_11_validCount_) <= (d_2_narrowThreshold_)) or ((stepTokenBudget) <= (1)):
                                d_12_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_12_next_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_12_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_13_appendedGenerated_: _dafny.Seq
                                    d_14_appendedInside_: bool
                                    d_15_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                    d_13_appendedGenerated_ = out9_
                                    d_14_appendedInside_ = out10_
                                    d_15_appendedCurrent_ = out11_
                                    generated = d_13_appendedGenerated_
                                    insideConstrainedOut = d_14_appendedInside_
                                    currentConstrainedOut = d_15_appendedCurrent_
                            elif True:
                                d_16_remainingBudget_: int
                                d_16_remainingBudget_ = (maxSteps) - (d_1_steps_)
                                if (d_16_remainingBudget_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_17_localBudget_: int
                                    d_17_localBudget_ = stepTokenBudget
                                    if (d_16_remainingBudget_) < (d_17_localBudget_):
                                        d_17_localBudget_ = d_16_remainingBudget_
                                    (lm).GenerateLogits((d_10_constrainedPrompt_) + (currentConstrainedOut))
                                    if (len(d_3_flatPreferred_)) > (0):
                                        d_18_candidates_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                        d_18_candidates_ = out12_
                                        d_19_preferred_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out13_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_18_candidates_, d_3_flatPreferred_)
                                        d_19_preferred_ = out13_
                                        if (len(d_19_preferred_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_19_preferred_, _dafny.BigRational('3e0'))
                                        d_20_prevTok_: _dafny.Seq
                                        d_21_foundPrev_: bool
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out14_, out15_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")))
                                        d_20_prevTok_ = out14_
                                        d_21_foundPrev_ = out15_
                                        if d_21_foundPrev_:
                                            d_22_activeIdx_: int
                                            out16_: int
                                            out16_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_20_prevTok_)
                                            d_22_activeIdx_ = out16_
                                            if (d_22_activeIdx_) >= (0):
                                                d_23_activePreferred_: _dafny.Seq
                                                out17_: _dafny.Seq
                                                out17_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_18_candidates_, (validTokenGroups)[d_22_activeIdx_])
                                                d_23_activePreferred_ = out17_
                                                if (len(d_23_activePreferred_)) > (0):
                                                    (d_0_helpers_).BoostTokenLogits(lm, d_23_activePreferred_, _dafny.BigRational('8e0'))
                                    d_24_stablePrefix_: _dafny.Seq
                                    d_24_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_25_symbolOut_: _dafny.Seq
                                    d_26_hitEos_: bool
                                    d_27_stepsUsed2_: int
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: int
                                    out18_, out19_, out20_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, d_17_localBudget_, eosToken)
                                    d_25_symbolOut_ = out18_
                                    d_26_hitEos_ = out19_
                                    d_27_stepsUsed2_ = out20_
                                    generated = (d_24_stablePrefix_) + (d_25_symbolOut_)
                                    currentConstrainedOut = d_25_symbolOut_
                                    d_1_steps_ = (d_1_steps_) + (d_27_stepsUsed2_)
                                    if d_26_hitEos_:
                                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

