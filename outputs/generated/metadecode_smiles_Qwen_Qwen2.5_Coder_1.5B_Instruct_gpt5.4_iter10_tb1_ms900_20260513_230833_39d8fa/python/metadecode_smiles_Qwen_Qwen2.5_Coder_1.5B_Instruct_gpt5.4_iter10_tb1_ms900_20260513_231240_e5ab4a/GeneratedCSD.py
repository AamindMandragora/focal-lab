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
        d_2_narrowThreshold_ = 6
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 48
        d_4_penaltyTokens_: _dafny.Seq
        d_4_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_openedGenerated_: _dafny.Seq
                        d_6_openedInside_: bool
                        d_7_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_5_openedGenerated_ = out0_
                        d_6_openedInside_ = out1_
                        d_7_openedCurrent_ = out2_
                        generated = d_5_openedGenerated_
                        insideConstrainedOut = d_6_openedInside_
                        currentConstrainedOut = d_7_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedGenerated_: _dafny.Seq
                        d_9_closedInside_: bool
                        d_10_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedGenerated_ = out3_
                        d_9_closedInside_ = out4_
                        d_10_closedCurrent_ = out5_
                        generated = d_8_closedGenerated_
                        insideConstrainedOut = d_9_closedInside_
                        currentConstrainedOut = d_10_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                        d_11_rolledGenerated_: _dafny.Seq
                        d_12_rolledCurrent_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: _dafny.Seq
                        out6_, out7_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_11_rolledGenerated_ = out6_
                        d_12_rolledCurrent_ = out7_
                        generated = d_11_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_12_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_13_stablePrefix_: _dafny.Seq
                        d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                        d_15_validCount_: int
                        out8_: int
                        out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_15_validCount_ = out8_
                        if (d_15_validCount_) <= (d_2_narrowThreshold_):
                            d_16_candidates_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                            d_16_candidates_ = out9_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_17_next_: _dafny.Seq
                            d_17_next_ = eosToken
                            if (len(d_16_candidates_)) > (0):
                                d_17_next_ = (d_16_candidates_)[0]
                                if ((d_17_next_) == (eosToken)) and ((len(d_16_candidates_)) > (1)):
                                    d_17_next_ = (d_16_candidates_)[1]
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
                            d_21_nextSoft_: _dafny.Seq
                            d_22_usedFallback_: bool
                            out13_: _dafny.Seq
                            out14_: bool
                            out13_, out14_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_21_nextSoft_ = out13_
                            d_22_usedFallback_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_21_nextSoft_) == (eosToken):
                                raise _dafny.Break("0")
                            elif d_22_usedFallback_:
                                d_23_nextPen_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, d_4_penaltyTokens_, _dafny.BigRational('4e0'), eosToken)
                                d_23_nextPen_ = out15_
                                if (d_1_steps_) < (maxSteps):
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_23_nextPen_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_24_appendedGenerated2_: _dafny.Seq
                                        d_25_appendedInside2_: bool
                                        d_26_appendedCurrent2_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: _dafny.Seq
                                        out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nextPen_)
                                        d_24_appendedGenerated2_ = out16_
                                        d_25_appendedInside2_ = out17_
                                        d_26_appendedCurrent2_ = out18_
                                        generated = d_24_appendedGenerated2_
                                        insideConstrainedOut = d_25_appendedInside2_
                                        currentConstrainedOut = d_26_appendedCurrent2_
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_27_appendedGenerated3_: _dafny.Seq
                                d_28_appendedInside3_: bool
                                d_29_appendedCurrent3_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_nextSoft_)
                                d_27_appendedGenerated3_ = out19_
                                d_28_appendedInside3_ = out20_
                                d_29_appendedCurrent3_ = out21_
                                generated = d_27_appendedGenerated3_
                                insideConstrainedOut = d_28_appendedInside3_
                                currentConstrainedOut = d_29_appendedCurrent3_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

