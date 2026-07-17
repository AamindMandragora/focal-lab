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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Give the symbolic answer inside << >>. Example: <<n*k+m>>. Be concise.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_freebudget_: int
            d_2_freebudget_ = 4
            if (d_2_freebudget_) > ((maxSteps) - (d_1_steps_)):
                d_2_freebudget_ = (maxSteps) - (d_1_steps_)
            if (d_2_freebudget_) > (0):
                d_3_chunkGenerated_: _dafny.Seq
                d_4_stoppedOnOpen_: bool
                d_5_stoppedOnEos_: bool
                d_6_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_freebudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_3_chunkGenerated_ = out0_
                d_4_stoppedOnOpen_ = out1_
                d_5_stoppedOnEos_ = out2_
                d_6_stepsUsed_ = out3_
                generated = d_3_chunkGenerated_
                d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                if d_5_stoppedOnEos_:
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                elif d_4_stoppedOnOpen_:
                    d_7_enterGenerated_: _dafny.Seq
                    d_8_enterInside_: bool
                    d_9_enterCurrent_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_7_enterGenerated_ = out4_
                    d_8_enterInside_ = out5_
                    d_9_enterCurrent_ = out6_
                    generated = d_7_enterGenerated_
                    insideConstrainedOut = d_8_enterInside_
                    currentConstrainedOut = d_9_enterCurrent_
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_10_openGenerated_: _dafny.Seq
            d_11_openInside_: bool
            d_12_openCurrent_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_10_openGenerated_ = out7_
            d_11_openInside_ = out8_
            d_12_openCurrent_ = out9_
            generated = d_10_openGenerated_
            insideConstrainedOut = d_11_openInside_
            currentConstrainedOut = d_12_openCurrent_
            d_1_steps_ = (d_1_steps_) + (1)
        d_13_tokensInSpan_: int
        d_13_tokensInSpan_ = 0
        d_14_maxTokensInSpan_: int
        d_14_maxTokensInSpan_ = 6
        while ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and ((d_13_tokensInSpan_) < (d_14_maxTokensInSpan_)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_15_closedGenerated_: _dafny.Seq
                d_16_closedInside_: bool
                d_17_closedCurrent_: _dafny.Seq
                out10_: _dafny.Seq
                out11_: bool
                out12_: _dafny.Seq
                out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_15_closedGenerated_ = out10_
                d_16_closedInside_ = out11_
                d_17_closedCurrent_ = out12_
                generated = d_15_closedGenerated_
                insideConstrainedOut = d_16_closedInside_
                currentConstrainedOut = d_17_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_18_constrainedPrompt_: _dafny.Seq
                d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_19_next_: _dafny.Seq
                out13_: _dafny.Seq
                out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                d_19_next_ = out13_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_19_next_) == (eosToken):
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                elif True:
                    d_20_valid_: bool
                    out14_: bool
                    out14_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_19_next_)
                    d_20_valid_ = out14_
                    if d_20_valid_:
                        d_21_appendedGenerated_: _dafny.Seq
                        d_22_appendedInside_: bool
                        d_23_appendedCurrent_: _dafny.Seq
                        out15_: _dafny.Seq
                        out16_: bool
                        out17_: _dafny.Seq
                        out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                        d_21_appendedGenerated_ = out15_
                        d_22_appendedInside_ = out16_
                        d_23_appendedCurrent_ = out17_
                        generated = d_21_appendedGenerated_
                        insideConstrainedOut = d_22_appendedInside_
                        currentConstrainedOut = d_23_appendedCurrent_
                        d_13_tokensInSpan_ = (d_13_tokensInSpan_) + (1)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_24_rolledGenerated_: _dafny.Seq
            d_25_rolledCurrent_: _dafny.Seq
            out18_: _dafny.Seq
            out19_: _dafny.Seq
            out18_, out19_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
            d_24_rolledGenerated_ = out18_
            d_25_rolledCurrent_ = out19_
            generated = d_24_rolledGenerated_
            currentConstrainedOut = d_25_rolledCurrent_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                d_26_constrainedPrompt2_: _dafny.Seq
                d_26_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_27_next2_: _dafny.Seq
                out20_: _dafny.Seq
                out20_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_26_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                d_27_next2_ = out20_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_27_next2_) != (eosToken):
                    d_28_valid2_: bool
                    out21_: bool
                    out21_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_27_next2_)
                    d_28_valid2_ = out21_
                    if d_28_valid2_:
                        d_29_appGen2_: _dafny.Seq
                        d_30_appInside2_: bool
                        d_31_appCurrent2_: _dafny.Seq
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: _dafny.Seq
                        out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next2_)
                        d_29_appGen2_ = out22_
                        d_30_appInside2_ = out23_
                        d_31_appCurrent2_ = out24_
                        generated = d_29_appGen2_
                        insideConstrainedOut = d_30_appInside2_
                        currentConstrainedOut = d_31_appCurrent2_
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                d_32_closedGenerated2_: _dafny.Seq
                d_33_closedInside2_: bool
                d_34_closedCurrent2_: _dafny.Seq
                out25_: _dafny.Seq
                out26_: bool
                out27_: _dafny.Seq
                out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_32_closedGenerated2_ = out25_
                d_33_closedInside2_ = out26_
                d_34_closedCurrent2_ = out27_
                generated = d_32_closedGenerated2_
                insideConstrainedOut = d_33_closedInside2_
                currentConstrainedOut = d_34_closedCurrent2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

