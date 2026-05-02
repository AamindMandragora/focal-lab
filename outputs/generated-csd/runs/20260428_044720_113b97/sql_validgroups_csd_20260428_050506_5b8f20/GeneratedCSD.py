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
                    d_2_fuel_ = (d_2_fuel_) - (1)
                    if not(insideConstrainedOut):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_4_completeNow_: bool
                        d_4_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        d_5_narrow_: bool
                        out1_: bool
                        out1_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_5_narrow_ = out1_
                        if (d_5_narrow_) and ((0) < (len(currentConstrainedOut))):
                            d_6_stablePrefixRepair_: _dafny.Seq
                            d_6_stablePrefixRepair_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_7_repairedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out2_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                            d_7_repairedCurrent_ = out2_
                            generated = (d_6_stablePrefixRepair_) + (d_7_repairedCurrent_)
                            currentConstrainedOut = d_7_repairedCurrent_
                        elif True:
                            d_8_stablePrefix_: _dafny.Seq
                            d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                            (lm).GenerateLogits((d_9_constrainedPrompt_) + (currentConstrainedOut))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_10_candidates_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, 32, eosToken)
                            d_10_candidates_ = out3_
                            d_11_preferred_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_10_candidates_, d_3_flatPreferred_)
                            d_11_preferred_ = out4_
                            if (len(d_11_preferred_)) > (0):
                                (d_0_helpers_).BoostTokenLogits(lm, d_11_preferred_, _dafny.BigRational('8e0'))
                                d_12_otherCandidates_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = VerifiedDecoderAgent.CSDHelpers.SubtractTokenSets(d_10_candidates_, d_11_preferred_)
                                d_12_otherCandidates_ = out5_
                                if (len(d_12_otherCandidates_)) > (0):
                                    (d_0_helpers_).PenalizeTokenLogits(lm, d_12_otherCandidates_, _dafny.BigRational('1e0'))
                            if (0) < (len(currentConstrainedOut)):
                                d_13_lastTok_: _dafny.Seq
                                d_13_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                                if (d_13_lastTok_) in ((lm).Tokens):
                                    (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_13_lastTok_]), _dafny.BigRational('15e-1'))
                            if d_4_completeNow_:
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('6e0'))
                            elif True:
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('4e0'))
                            if (stepTokenBudget) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_14_currentSym_: _dafny.Seq
                                d_15_hitEos_: bool
                                d_16_stepsUsed_: int
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: int
                                out6_, out7_, out8_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, stepTokenBudget, eosToken)
                                d_14_currentSym_ = out6_
                                d_15_hitEos_ = out7_
                                d_16_stepsUsed_ = out8_
                                if (d_16_stepsUsed_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    if (d_16_stepsUsed_) <= ((maxSteps) - (d_1_steps_)):
                                        generated = (d_8_stablePrefix_) + (d_14_currentSym_)
                                        currentConstrainedOut = d_14_currentSym_
                                        d_1_steps_ = (d_1_steps_) + (d_16_stepsUsed_)
                                        if d_15_hitEos_:
                                            d_17_completeAfter_: bool
                                            d_17_completeAfter_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                            if d_17_completeAfter_:
                                                raise _dafny.Break("0")
                                            elif True:
                                                if (0) < (len(currentConstrainedOut)):
                                                    d_18_stablePrefixRepair2_: _dafny.Seq
                                                    d_18_stablePrefixRepair2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                                    d_19_repairedCurrent2_: _dafny.Seq
                                                    out9_: _dafny.Seq
                                                    out9_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                                    d_19_repairedCurrent2_ = out9_
                                                    generated = (d_18_stablePrefixRepair2_) + (d_19_repairedCurrent2_)
                                                    currentConstrainedOut = d_19_repairedCurrent2_
                                                elif True:
                                                    raise _dafny.Break("0")
                                    elif True:
                                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

