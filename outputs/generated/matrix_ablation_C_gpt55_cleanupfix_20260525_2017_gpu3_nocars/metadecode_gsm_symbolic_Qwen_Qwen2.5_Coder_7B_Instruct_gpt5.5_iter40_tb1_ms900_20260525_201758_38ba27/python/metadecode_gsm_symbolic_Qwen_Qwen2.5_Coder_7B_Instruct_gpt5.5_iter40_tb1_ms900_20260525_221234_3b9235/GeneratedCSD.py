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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem concisely, then give one final visible answer span: <<expression>>. Do not wrap intermediate calculations. Preserve variable names exactly, especially underscores such as n_1, n_2, k_2, and k_3; never rewrite n_2 as n2. Include every contribution from the story before giving the final expression. Use plain symbolic arithmetic, not LaTeX.")))
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_done_: bool
            d_2_done_ = False
            d_3_penaltyTokens_: _dafny.Seq
            d_3_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " n2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " n1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " k2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " k3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\[")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\]")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))])
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
                d_1_steps_ = 1
            elif (parser).IsCompletePrefix(currentConstrainedOut):
                d_7_closedGenerated0_: _dafny.Seq
                d_8_closedInside0_: bool
                d_9_closedCurrent0_: _dafny.Seq
                out3_: _dafny.Seq
                out4_: bool
                out5_: _dafny.Seq
                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_7_closedGenerated0_ = out3_
                d_8_closedInside0_ = out4_
                d_9_closedCurrent0_ = out5_
                generated = d_7_closedGenerated0_
                insideConstrainedOut = d_8_closedInside0_
                currentConstrainedOut = d_9_closedCurrent0_
                d_1_steps_ = 1
                d_2_done_ = True
            elif True:
                d_10_constrainedPrompt0_: _dafny.Seq
                d_10_constrainedPrompt0_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_11_nextConstrained0_: _dafny.Seq
                out6_: _dafny.Seq
                out6_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_10_constrainedPrompt0_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_penaltyTokens_, _dafny.BigRational('15e-1'), 40, eosToken)
                d_11_nextConstrained0_ = out6_
                d_1_steps_ = 1
                if (d_11_nextConstrained0_) == (eosToken):
                    d_2_done_ = True
                elif True:
                    d_12_appendedGenerated0_: _dafny.Seq
                    d_13_appendedInside0_: bool
                    d_14_appendedCurrent0_: _dafny.Seq
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_nextConstrained0_)
                    d_12_appendedGenerated0_ = out7_
                    d_13_appendedInside0_ = out8_
                    d_14_appendedCurrent0_ = out9_
                    generated = d_12_appendedGenerated0_
                    insideConstrainedOut = d_13_appendedInside0_
                    currentConstrainedOut = d_14_appendedCurrent0_
            while (not(d_2_done_)) and ((d_1_steps_) < (maxSteps)):
                if not(insideConstrainedOut):
                    d_15_openedGenerated2_: _dafny.Seq
                    d_16_openedInside2_: bool
                    d_17_openedCurrent2_: _dafny.Seq
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_15_openedGenerated2_ = out10_
                    d_16_openedInside2_ = out11_
                    d_17_openedCurrent2_ = out12_
                    generated = d_15_openedGenerated2_
                    insideConstrainedOut = d_16_openedInside2_
                    currentConstrainedOut = d_17_openedCurrent2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif (parser).IsCompletePrefix(currentConstrainedOut):
                    d_18_closedGenerated_: _dafny.Seq
                    d_19_closedInside_: bool
                    d_20_closedCurrent_: _dafny.Seq
                    out13_: _dafny.Seq
                    out14_: bool
                    out15_: _dafny.Seq
                    out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_18_closedGenerated_ = out13_
                    d_19_closedInside_ = out14_
                    d_20_closedCurrent_ = out15_
                    generated = d_18_closedGenerated_
                    insideConstrainedOut = d_19_closedInside_
                    currentConstrainedOut = d_20_closedCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_2_done_ = True
                elif True:
                    d_21_constrainedPrompt_: _dafny.Seq
                    d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_22_nextConstrained_: _dafny.Seq
                    out16_: _dafny.Seq
                    out16_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_penaltyTokens_, _dafny.BigRational('15e-1'), 40, eosToken)
                    d_22_nextConstrained_ = out16_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_22_nextConstrained_) == (eosToken):
                        d_2_done_ = True
                    elif True:
                        d_23_appendedGenerated_: _dafny.Seq
                        d_24_appendedInside_: bool
                        d_25_appendedCurrent_: _dafny.Seq
                        out17_: _dafny.Seq
                        out18_: bool
                        out19_: _dafny.Seq
                        out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextConstrained_)
                        d_23_appendedGenerated_ = out17_
                        d_24_appendedInside_ = out18_
                        d_25_appendedCurrent_ = out19_
                        generated = d_23_appendedGenerated_
                        insideConstrainedOut = d_24_appendedInside_
                        currentConstrainedOut = d_25_appendedCurrent_
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

