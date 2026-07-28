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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write only the final symbolic expression inside << >>. Example: <<n*(n+1)/2>>. Do not use = inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_tokensInSpan_: int
        d_2_tokensInSpan_ = 0
        d_3_maxTokensInSpan_: int
        d_3_maxTokensInSpan_ = 8
        d_4_penaltyTokens_: _dafny.Seq
        d_4_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "==")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">="))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remaining_: int
                        d_5_remaining_ = (maxSteps) - (d_1_steps_)
                        d_6_chunkBudget_: int
                        d_6_chunkBudget_ = 5
                        if (d_6_chunkBudget_) > (d_5_remaining_):
                            d_6_chunkBudget_ = d_5_remaining_
                        if (d_6_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_7_chunkGenerated_: _dafny.Seq
                        d_8_stoppedOnOpen_: bool
                        d_9_stoppedOnEos_: bool
                        d_10_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_7_chunkGenerated_ = out0_
                        d_8_stoppedOnOpen_ = out1_
                        d_9_stoppedOnEos_ = out2_
                        d_10_stepsUsed_ = out3_
                        generated = d_7_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                        if d_9_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_8_stoppedOnOpen_:
                            d_11_enterGenerated_: _dafny.Seq
                            d_12_enterInside_: bool
                            d_13_enterCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_11_enterGenerated_ = out4_
                            d_12_enterInside_ = out5_
                            d_13_enterCurrent_ = out6_
                            generated = d_11_enterGenerated_
                            insideConstrainedOut = d_12_enterInside_
                            currentConstrainedOut = d_13_enterCurrent_
                            d_2_tokensInSpan_ = 0
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_14_openGenerated_: _dafny.Seq
                                d_15_openInside_: bool
                                d_16_openCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_14_openGenerated_ = out7_
                                d_15_openInside_ = out8_
                                d_16_openCurrent_ = out9_
                                generated = d_14_openGenerated_
                                insideConstrainedOut = d_15_openInside_
                                currentConstrainedOut = d_16_openCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_tokensInSpan_ = 0
                            elif True:
                                raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        if (d_1_steps_) < (maxSteps):
                            d_17_closedGenerated_: _dafny.Seq
                            d_18_closedInside_: bool
                            d_19_closedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_17_closedGenerated_ = out10_
                            d_18_closedInside_ = out11_
                            d_19_closedCurrent_ = out12_
                            generated = d_17_closedGenerated_
                            insideConstrainedOut = d_18_closedInside_
                            currentConstrainedOut = d_19_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_tokensInSpan_ = 0
                            raise _dafny.Break("0")
                        elif True:
                            raise _dafny.Break("0")
                    elif (d_2_tokensInSpan_) >= (d_3_maxTokensInSpan_):
                        d_20_rolledGenerated_: _dafny.Seq
                        d_21_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_20_rolledGenerated_ = out13_
                        d_21_rolledCurrent_ = out14_
                        generated = d_20_rolledGenerated_
                        currentConstrainedOut = d_21_rolledCurrent_
                        d_2_tokensInSpan_ = 0
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            if (d_1_steps_) < (maxSteps):
                                d_22_closedGenerated_: _dafny.Seq
                                d_23_closedInside_: bool
                                d_24_closedCurrent_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_22_closedGenerated_ = out15_
                                d_23_closedInside_ = out16_
                                d_24_closedCurrent_ = out17_
                                generated = d_22_closedGenerated_
                                insideConstrainedOut = d_23_closedInside_
                                currentConstrainedOut = d_24_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_25_constrainedPrompt_: _dafny.Seq
                                d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_26_next_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_4_penaltyTokens_, _dafny.BigRational('6e0'), 12, eosToken)
                                d_26_next_ = out18_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_26_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_27_valid_: bool
                                    out19_: bool
                                    out19_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_26_next_)
                                    d_27_valid_ = out19_
                                    if d_27_valid_:
                                        d_28_appendedGenerated_: _dafny.Seq
                                        d_29_appendedInside_: bool
                                        d_30_appendedCurrent_: _dafny.Seq
                                        out20_: _dafny.Seq
                                        out21_: bool
                                        out22_: _dafny.Seq
                                        out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                        d_28_appendedGenerated_ = out20_
                                        d_29_appendedInside_ = out21_
                                        d_30_appendedCurrent_ = out22_
                                        generated = d_28_appendedGenerated_
                                        insideConstrainedOut = d_29_appendedInside_
                                        currentConstrainedOut = d_30_appendedCurrent_
                                        d_2_tokensInSpan_ = (d_2_tokensInSpan_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        if (d_1_steps_) < (maxSteps):
                            d_31_constrainedPrompt_: _dafny.Seq
                            d_31_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_32_next_: _dafny.Seq
                            out23_: _dafny.Seq
                            out23_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_31_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_4_penaltyTokens_, _dafny.BigRational('6e0'), 12, eosToken)
                            d_32_next_ = out23_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_32_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_33_valid_: bool
                                out24_: bool
                                out24_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_32_next_)
                                d_33_valid_ = out24_
                                if d_33_valid_:
                                    d_34_appendedGenerated_: _dafny.Seq
                                    d_35_appendedInside_: bool
                                    d_36_appendedCurrent_: _dafny.Seq
                                    out25_: _dafny.Seq
                                    out26_: bool
                                    out27_: _dafny.Seq
                                    out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next_)
                                    d_34_appendedGenerated_ = out25_
                                    d_35_appendedInside_ = out26_
                                    d_36_appendedCurrent_ = out27_
                                    generated = d_34_appendedGenerated_
                                    insideConstrainedOut = d_35_appendedInside_
                                    currentConstrainedOut = d_36_appendedCurrent_
                                    d_2_tokensInSpan_ = (d_2_tokensInSpan_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_37_rolledGenerated_: _dafny.Seq
            d_38_rolledCurrent_: _dafny.Seq
            out28_: _dafny.Seq
            out29_: _dafny.Seq
            out28_, out29_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
            d_37_rolledGenerated_ = out28_
            d_38_rolledCurrent_ = out29_
            generated = d_37_rolledGenerated_
            currentConstrainedOut = d_38_rolledCurrent_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_39_closedGenerated_: _dafny.Seq
                d_40_closedInside_: bool
                d_41_closedCurrent_: _dafny.Seq
                out30_: _dafny.Seq
                out31_: bool
                out32_: _dafny.Seq
                out30_, out31_, out32_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_39_closedGenerated_ = out30_
                d_40_closedInside_ = out31_
                d_41_closedCurrent_ = out32_
                generated = d_39_closedGenerated_
                insideConstrainedOut = d_40_closedInside_
                currentConstrainedOut = d_41_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

