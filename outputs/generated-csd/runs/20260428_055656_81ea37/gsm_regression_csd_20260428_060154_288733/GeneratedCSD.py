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
        d_2_preferredFlat_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_preferredFlat_ = out0_
        d_3_preferredFlatSafe_: bool
        d_3_preferredFlatSafe_ = True
        d_4_pfIdx_: int
        d_4_pfIdx_ = 0
        while (d_4_pfIdx_) < (len(d_2_preferredFlat_)):
            if ((d_2_preferredFlat_)[d_4_pfIdx_]) in ((lm).Tokens):
                pass
            elif True:
                d_3_preferredFlatSafe_ = False
            d_4_pfIdx_ = (d_4_pfIdx_) + (1)
        d_5_openBiasDelay_: int
        d_5_openBiasDelay_ = 6
        d_6_openBiasPeriod_: int
        d_6_openBiasPeriod_ = 4
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
                        if (d_1_steps_) >= (d_5_openBiasDelay_):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('12e0'))
                            if (_dafny.euclidian_modulus((d_1_steps_) - (d_5_openBiasDelay_), d_6_openBiasPeriod_)) == (0):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('2e1'))
                        if (((len(d_2_preferredFlat_)) > (0)) and ((d_1_steps_) < (d_5_openBiasDelay_))) and (d_3_preferredFlatSafe_):
                            (d_0_helpers_).BoostTokenLogits(lm, d_2_preferredFlat_, _dafny.BigRational('3e0'))
                        d_7_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (lm).ChooseNextTokenUnconstrained()
                        d_7_next_ = out1_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                        if (d_7_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_8_deadEnd_: bool
                        out2_: bool
                        out2_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_8_deadEnd_ = out2_
                        if d_8_deadEnd_:
                            d_9_stablePrefix_: _dafny.Seq
                            d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_10_repairedGenerated_: _dafny.Seq
                            d_11_repairedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: _dafny.Seq
                            out3_, out4_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_9_stablePrefix_, generated, currentConstrainedOut)
                            d_10_repairedGenerated_ = out3_
                            d_11_repairedCurrent_ = out4_
                            generated = d_10_repairedGenerated_
                            currentConstrainedOut = d_11_repairedCurrent_
                            insideConstrainedOut = True
                            raise _dafny.Break("0")
                        elif True:
                            d_12_completeNow_: bool
                            d_12_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_12_completeNow_:
                                d_13_closedGenerated_: _dafny.Seq
                                d_14_closedInside_: bool
                                d_15_closedCurrent_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_13_closedGenerated_ = out5_
                                d_14_closedInside_ = out6_
                                d_15_closedCurrent_ = out7_
                                generated = d_13_closedGenerated_
                                insideConstrainedOut = d_14_closedInside_
                                currentConstrainedOut = d_15_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_16_stablePrefix2_: _dafny.Seq
                                d_16_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_17_constrainedPrompt_: _dafny.Seq
                                d_17_constrainedPrompt_ = (prompt) + (d_16_stablePrefix2_)
                                (lm).GenerateLogits((d_17_constrainedPrompt_) + (currentConstrainedOut))
                                if ((len(d_2_preferredFlat_)) > (0)) and (d_3_preferredFlatSafe_):
                                    d_18_candidates_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, 20, eosToken)
                                    d_18_candidates_ = out8_
                                    d_19_preferred_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out9_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_18_candidates_, d_2_preferredFlat_)
                                    d_19_preferred_ = out9_
                                    if (len(d_19_preferred_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_19_preferred_, _dafny.BigRational('8e0'))
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_20_next_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (lm).ChooseNextToken()
                                d_20_next_ = out10_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_20_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_21_appendedGenerated_: _dafny.Seq
                                    d_22_appendedInside_: bool
                                    d_23_appendedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                    d_21_appendedGenerated_ = out11_
                                    d_22_appendedInside_ = out12_
                                    d_23_appendedCurrent_ = out13_
                                    generated = d_21_appendedGenerated_
                                    insideConstrainedOut = d_22_appendedInside_
                                    currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

