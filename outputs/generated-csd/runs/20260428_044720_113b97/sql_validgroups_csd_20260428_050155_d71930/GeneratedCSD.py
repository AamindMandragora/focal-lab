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
        d_2_fuel_: int
        d_2_fuel_ = maxSteps
        d_3_flatPreferred_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_3_flatPreferred_ = out0_
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and ((d_2_fuel_) > (0)):
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
                        d_2_fuel_ = (d_2_fuel_) - (1)
                    elif True:
                        d_7_completeNow_: bool
                        d_7_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        d_8_narrow_: bool
                        out4_: bool
                        out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_8_narrow_ = out4_
                        if (d_8_narrow_) and ((0) < (len(currentConstrainedOut))):
                            d_9_stablePrefixRepair_: _dafny.Seq
                            d_9_stablePrefixRepair_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_10_repairedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                            d_10_repairedCurrent_ = out5_
                            currentConstrainedOut = d_10_repairedCurrent_
                            generated = (d_9_stablePrefixRepair_) + (currentConstrainedOut)
                            raise _dafny.Break("0")
                        elif True:
                            d_11_stablePrefix_: _dafny.Seq
                            d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_12_constrainedPrompt_: _dafny.Seq
                            d_12_constrainedPrompt_ = (prompt) + (d_11_stablePrefix_)
                            (lm).GenerateLogits((d_12_constrainedPrompt_) + (currentConstrainedOut))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_13_candidates_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, 32, eosToken)
                            d_13_candidates_ = out6_
                            d_14_preferred_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_13_candidates_, d_3_flatPreferred_)
                            d_14_preferred_ = out7_
                            if (len(d_14_preferred_)) > (0):
                                (d_0_helpers_).BoostTokenLogits(lm, d_14_preferred_, _dafny.BigRational('8e0'))
                                d_15_otherCandidates_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.SubtractTokenSets(d_13_candidates_, d_14_preferred_)
                                d_15_otherCandidates_ = out8_
                                if (len(d_15_otherCandidates_)) > (0):
                                    (d_0_helpers_).PenalizeTokenLogits(lm, d_15_otherCandidates_, _dafny.BigRational('15e-1'))
                            if (0) < (len(currentConstrainedOut)):
                                d_16_lastTok_: _dafny.Seq
                                d_16_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                                if (d_16_lastTok_) in ((lm).Tokens):
                                    (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_16_lastTok_]), _dafny.BigRational('2e0'))
                            if d_7_completeNow_:
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('6e0'))
                            elif True:
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('3e0'))
                            d_17_next_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (lm).ChooseNextToken()
                            d_17_next_ = out9_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            if (d_17_next_) == (eosToken):
                                if d_7_completeNow_:
                                    d_18_closedGenerated_: _dafny.Seq
                                    d_19_closedInside_: bool
                                    d_20_closedCurrent_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_18_closedGenerated_ = out10_
                                    d_19_closedInside_ = out11_
                                    d_20_closedCurrent_ = out12_
                                    generated = d_18_closedGenerated_
                                    insideConstrainedOut = d_19_closedInside_
                                    currentConstrainedOut = d_20_closedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_2_fuel_ = (d_2_fuel_) - (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_21_nextValid_: bool
                                out13_: bool
                                out13_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_17_next_)
                                d_21_nextValid_ = out13_
                                if (not(d_7_completeNow_)) and (d_21_nextValid_):
                                    d_22_appendedGenerated_: _dafny.Seq
                                    d_23_appendedInside_: bool
                                    d_24_appendedCurrent_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                    d_22_appendedGenerated_ = out14_
                                    d_23_appendedInside_ = out15_
                                    d_24_appendedCurrent_ = out16_
                                    generated = d_22_appendedGenerated_
                                    insideConstrainedOut = d_23_appendedInside_
                                    currentConstrainedOut = d_24_appendedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_2_fuel_ = (d_2_fuel_) - (1)
                                elif True:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

