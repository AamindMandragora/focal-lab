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
        d_2_wantOpen_: bool
        d_2_wantOpen_ = False
        d_3_cueTokens_: _dafny.Seq
        d_3_cueTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "difference")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "product")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "quotient"))])
        d_4_penaltyTokens_: _dafny.Seq
        d_4_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_wantOpen_:
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
                            d_2_wantOpen_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_9_observedGenerated_: _dafny.Seq
                                    d_10_observedInside_: bool
                                    d_11_observedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_9_observedGenerated_ = out4_
                                    d_10_observedInside_ = out5_
                                    d_11_observedCurrent_ = out6_
                                    generated = d_9_observedGenerated_
                                    insideConstrainedOut = d_10_observedInside_
                                    currentConstrainedOut = d_11_observedCurrent_
                                    d_2_wantOpen_ = False
                                elif True:
                                    if (d_8_next_) in (d_3_cueTokens_):
                                        d_2_wantOpen_ = True
                                    elif True:
                                        d_12_prevTok_: _dafny.Seq
                                        d_13_foundPrev_: bool
                                        out7_: _dafny.Seq
                                        out8_: bool
                                        out7_, out8_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                                        d_12_prevTok_ = out7_
                                        d_13_foundPrev_ = out8_
                                        if d_13_foundPrev_:
                                            if (d_12_prevTok_) in (d_3_cueTokens_):
                                                d_2_wantOpen_ = True
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_14_closedGenerated_: _dafny.Seq
                            d_15_closedInside_: bool
                            d_16_closedCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_14_closedGenerated_ = out9_
                            d_15_closedInside_ = out10_
                            d_16_closedCurrent_ = out11_
                            generated = d_14_closedGenerated_
                            insideConstrainedOut = d_15_closedInside_
                            currentConstrainedOut = d_16_closedCurrent_
                            d_2_wantOpen_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_17_isDeadEnd_: bool
                            out12_: bool
                            out12_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_17_isDeadEnd_ = out12_
                            if d_17_isDeadEnd_:
                                d_18_rolledGenerated_: _dafny.Seq
                                d_19_rolledCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: _dafny.Seq
                                out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_18_rolledGenerated_ = out13_
                                d_19_rolledCurrent_ = out14_
                                generated = d_18_rolledGenerated_
                                insideConstrainedOut = True
                                currentConstrainedOut = d_19_rolledCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_20_stablePrefix_: _dafny.Seq
                                d_20_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_21_constrainedPrompt_: _dafny.Seq
                                d_21_constrainedPrompt_ = (prompt) + (d_20_stablePrefix_)
                                d_22_validCount_: int
                                out15_: int
                                out15_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_22_validCount_ = out15_
                                d_23_next_: _dafny.Seq
                                d_23_next_ = eosToken
                                if (len(currentConstrainedOut)) == (0):
                                    out16_: _dafny.Seq
                                    out16_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_4_penaltyTokens_, _dafny.BigRational('4e0'), 12, eosToken)
                                    d_23_next_ = out16_
                                elif (d_22_validCount_) <= (3):
                                    out17_: _dafny.Seq
                                    out17_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, d_4_penaltyTokens_, _dafny.BigRational('4e0'), eosToken)
                                    d_23_next_ = out17_
                                elif True:
                                    out18_: _dafny.Seq
                                    out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                    d_23_next_ = out18_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_23_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_24_appendedGenerated_: _dafny.Seq
                                    d_25_appendedInside_: bool
                                    d_26_appendedCurrent_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                    d_24_appendedGenerated_ = out19_
                                    d_25_appendedInside_ = out20_
                                    d_26_appendedCurrent_ = out21_
                                    generated = d_24_appendedGenerated_
                                    insideConstrainedOut = d_25_appendedInside_
                                    currentConstrainedOut = d_26_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

