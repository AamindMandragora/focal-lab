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
        d_2_narrowThreshold_ = 10
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 96
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
                        d_17_useRepPenalty_: bool
                        d_17_useRepPenalty_ = False
                        if ((len(currentConstrainedOut)) > (8)) and ((d_15_openParens_) > (d_16_closeParens_)):
                            d_17_useRepPenalty_ = True
                        if d_17_useRepPenalty_:
                            d_18_nextRep_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_18_nextRep_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_nextRep_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_19_appendedGenerated_: _dafny.Seq
                                d_20_appendedInside_: bool
                                d_21_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_nextRep_)
                                d_19_appendedGenerated_ = out12_
                                d_20_appendedInside_ = out13_
                                d_21_appendedCurrent_ = out14_
                                generated = d_19_appendedGenerated_
                                insideConstrainedOut = d_20_appendedInside_
                                currentConstrainedOut = d_21_appendedCurrent_
                        elif (d_14_validCount_) <= (2):
                            d_22_nextConf_: _dafny.Seq
                            d_23_wasConstrained_: bool
                            out15_: _dafny.Seq
                            out16_: bool
                            out15_, out16_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_22_nextConf_ = out15_
                            d_23_wasConstrained_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_nextConf_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_24_appendedGenerated2_: _dafny.Seq
                                d_25_appendedInside2_: bool
                                d_26_appendedCurrent2_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextConf_)
                                d_24_appendedGenerated2_ = out17_
                                d_25_appendedInside2_ = out18_
                                d_26_appendedCurrent2_ = out19_
                                generated = d_24_appendedGenerated2_
                                insideConstrainedOut = d_25_appendedInside2_
                                currentConstrainedOut = d_26_appendedCurrent2_
                        elif (d_14_validCount_) <= (d_2_narrowThreshold_):
                            d_27_nextGroup_: _dafny.Seq
                            out20_: _dafny.Seq
                            out20_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_27_nextGroup_ = out20_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_27_nextGroup_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_28_appendedGenerated3_: _dafny.Seq
                                d_29_appendedInside3_: bool
                                d_30_appendedCurrent3_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_nextGroup_)
                                d_28_appendedGenerated3_ = out21_
                                d_29_appendedInside3_ = out22_
                                d_30_appendedCurrent3_ = out23_
                                generated = d_28_appendedGenerated3_
                                insideConstrainedOut = d_29_appendedInside3_
                                currentConstrainedOut = d_30_appendedCurrent3_
                        elif True:
                            d_31_remaining_: int
                            d_31_remaining_ = (maxSteps) - (d_1_steps_)
                            d_32_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_31_remaining_)):
                                d_32_symbolBudget_ = d_31_remaining_
                            elif True:
                                d_32_symbolBudget_ = stepTokenBudget
                            d_33_symbolGenerated_: _dafny.Seq
                            d_34_symbolOut_: _dafny.Seq
                            d_35_hitEos_: bool
                            d_36_symbolSteps_: int
                            out24_: _dafny.Seq
                            out25_: _dafny.Seq
                            out26_: bool
                            out27_: int
                            out24_, out25_, out26_, out27_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_13_constrainedPrompt_, generated, currentConstrainedOut, d_32_symbolBudget_, eosToken)
                            d_33_symbolGenerated_ = out24_
                            d_34_symbolOut_ = out25_
                            d_35_hitEos_ = out26_
                            d_36_symbolSteps_ = out27_
                            generated = d_33_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_34_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_36_symbolSteps_)
                            if d_35_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

