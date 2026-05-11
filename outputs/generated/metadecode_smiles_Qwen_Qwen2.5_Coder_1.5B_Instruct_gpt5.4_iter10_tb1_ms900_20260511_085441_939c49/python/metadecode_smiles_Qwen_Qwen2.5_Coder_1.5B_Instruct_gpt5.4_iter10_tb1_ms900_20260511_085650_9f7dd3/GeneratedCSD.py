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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 80
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_openedGenerated_: _dafny.Seq
                        d_5_openedInside_: bool
                        d_6_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_4_openedGenerated_ = out0_
                        d_5_openedInside_ = out1_
                        d_6_openedCurrent_ = out2_
                        generated = d_4_openedGenerated_
                        insideConstrainedOut = d_5_openedInside_
                        currentConstrainedOut = d_6_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_closedGenerated_: _dafny.Seq
                        d_8_closedInside_: bool
                        d_9_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedGenerated_ = out3_
                        d_8_closedInside_ = out4_
                        d_9_closedCurrent_ = out5_
                        generated = d_7_closedGenerated_
                        insideConstrainedOut = d_8_closedInside_
                        currentConstrainedOut = d_9_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                        d_10_rolledGenerated_: _dafny.Seq
                        d_11_rolledCurrent_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: _dafny.Seq
                        out6_, out7_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_10_rolledGenerated_ = out6_
                        d_11_rolledCurrent_ = out7_
                        generated = d_10_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_11_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_12_stablePrefix_: _dafny.Seq
                        d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                        d_14_validCount_: int
                        out8_: int
                        out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_14_validCount_ = out8_
                        d_15_openParens_: int
                        out9_: int
                        out9_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")))
                        d_15_openParens_ = out9_
                        d_16_closeParens_: int
                        out10_: int
                        out10_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")))
                        d_16_closeParens_ = out10_
                        d_17_ringDigits_: int
                        out11_: int
                        out11_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")))
                        d_17_ringDigits_ = out11_
                        d_18_useRepPenalty_: bool
                        d_18_useRepPenalty_ = False
                        if ((len(currentConstrainedOut)) > (6)) and (((d_15_openParens_) > (d_16_closeParens_)) or ((d_17_ringDigits_) > (1))):
                            d_18_useRepPenalty_ = True
                        if d_18_useRepPenalty_:
                            d_19_nextRep_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_19_nextRep_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_nextRep_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_20_appendedGenerated_: _dafny.Seq
                                d_21_appendedInside_: bool
                                d_22_appendedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_nextRep_)
                                d_20_appendedGenerated_ = out13_
                                d_21_appendedInside_ = out14_
                                d_22_appendedCurrent_ = out15_
                                generated = d_20_appendedGenerated_
                                insideConstrainedOut = d_21_appendedInside_
                                currentConstrainedOut = d_22_appendedCurrent_
                        elif (d_14_validCount_) <= (2):
                            d_23_nextConf_: _dafny.Seq
                            d_24_wasConstrained_: bool
                            out16_: _dafny.Seq
                            out17_: bool
                            out16_, out17_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_23_nextConf_ = out16_
                            d_24_wasConstrained_ = out17_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_23_nextConf_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_25_appendedGenerated2_: _dafny.Seq
                                d_26_appendedInside2_: bool
                                d_27_appendedCurrent2_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nextConf_)
                                d_25_appendedGenerated2_ = out18_
                                d_26_appendedInside2_ = out19_
                                d_27_appendedCurrent2_ = out20_
                                generated = d_25_appendedGenerated2_
                                insideConstrainedOut = d_26_appendedInside2_
                                currentConstrainedOut = d_27_appendedCurrent2_
                        elif True:
                            d_28_nextAdaptive_: _dafny.Seq
                            out21_: _dafny.Seq
                            out21_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_28_nextAdaptive_ = out21_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_28_nextAdaptive_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_29_appendedGenerated3_: _dafny.Seq
                                d_30_appendedInside3_: bool
                                d_31_appendedCurrent3_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_nextAdaptive_)
                                d_29_appendedGenerated3_ = out22_
                                d_30_appendedInside3_ = out23_
                                d_31_appendedCurrent3_ = out24_
                                generated = d_29_appendedGenerated3_
                                insideConstrainedOut = d_30_appendedInside3_
                                currentConstrainedOut = d_31_appendedCurrent3_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

