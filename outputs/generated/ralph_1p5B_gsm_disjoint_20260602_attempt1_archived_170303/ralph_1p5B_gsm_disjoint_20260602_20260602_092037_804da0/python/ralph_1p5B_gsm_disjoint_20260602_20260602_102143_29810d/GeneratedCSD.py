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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem. Write the final numeric answer inside << >> delimiters, like <<42>>. Be concise.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_tokensInSpan_: int
        d_2_tokensInSpan_ = 0
        d_3_maxTokensInSpan_: int
        d_3_maxTokensInSpan_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkBudget_: int
                        d_5_chunkBudget_ = 8
                        if (d_5_chunkBudget_) > (d_4_remaining_):
                            d_5_chunkBudget_ = d_4_remaining_
                        if (d_5_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_6_chunkGenerated_: _dafny.Seq
                        d_7_stoppedOnOpen_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkGenerated_ = out0_
                        d_7_stoppedOnOpen_ = out1_
                        d_8_stoppedOnEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOnOpen_:
                            d_10_enterGenerated_: _dafny.Seq
                            d_11_enterInside_: bool
                            d_12_enterCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_10_enterGenerated_ = out4_
                            d_11_enterInside_ = out5_
                            d_12_enterCurrent_ = out6_
                            generated = d_10_enterGenerated_
                            insideConstrainedOut = d_11_enterInside_
                            currentConstrainedOut = d_12_enterCurrent_
                            d_2_tokensInSpan_ = 0
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_13_openGenerated_: _dafny.Seq
                                d_14_openInside_: bool
                                d_15_openCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_13_openGenerated_ = out7_
                                d_14_openInside_ = out8_
                                d_15_openCurrent_ = out9_
                                generated = d_13_openGenerated_
                                insideConstrainedOut = d_14_openInside_
                                currentConstrainedOut = d_15_openCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_tokensInSpan_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        if (d_1_steps_) < (maxSteps):
                            d_16_closedGenerated_: _dafny.Seq
                            d_17_closedInside_: bool
                            d_18_closedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_closedGenerated_ = out10_
                            d_17_closedInside_ = out11_
                            d_18_closedCurrent_ = out12_
                            generated = d_16_closedGenerated_
                            insideConstrainedOut = d_17_closedInside_
                            currentConstrainedOut = d_18_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_tokensInSpan_ = 0
                            raise _dafny.Break("0")
                        elif True:
                            raise _dafny.Break("0")
                    elif (d_2_tokensInSpan_) >= (d_3_maxTokensInSpan_):
                        d_19_rolledGenerated_: _dafny.Seq
                        d_20_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_19_rolledGenerated_ = out13_
                        d_20_rolledCurrent_ = out14_
                        generated = d_19_rolledGenerated_
                        currentConstrainedOut = d_20_rolledCurrent_
                        d_2_tokensInSpan_ = 0
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            if (d_1_steps_) < (maxSteps):
                                d_21_closedGenerated_: _dafny.Seq
                                d_22_closedInside_: bool
                                d_23_closedCurrent_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_21_closedGenerated_ = out15_
                                d_22_closedInside_ = out16_
                                d_23_closedCurrent_ = out17_
                                generated = d_21_closedGenerated_
                                insideConstrainedOut = d_22_closedInside_
                                currentConstrainedOut = d_23_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_24_constrainedPrompt_: _dafny.Seq
                                d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_25_next_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_25_next_ = out18_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_25_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_26_appendedGenerated_: _dafny.Seq
                                    d_27_appendedInside_: bool
                                    d_28_appendedCurrent_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                    d_26_appendedGenerated_ = out19_
                                    d_27_appendedInside_ = out20_
                                    d_28_appendedCurrent_ = out21_
                                    generated = d_26_appendedGenerated_
                                    insideConstrainedOut = d_27_appendedInside_
                                    currentConstrainedOut = d_28_appendedCurrent_
                                    d_2_tokensInSpan_ = (d_2_tokensInSpan_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        d_29_constrainedPrompt_: _dafny.Seq
                        d_29_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_30_next_: _dafny.Seq
                        out22_: _dafny.Seq
                        out22_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_30_next_ = out22_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_30_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_31_appendedGenerated_: _dafny.Seq
                            d_32_appendedInside_: bool
                            d_33_appendedCurrent_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: bool
                            out25_: _dafny.Seq
                            out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                            d_31_appendedGenerated_ = out23_
                            d_32_appendedInside_ = out24_
                            d_33_appendedCurrent_ = out25_
                            generated = d_31_appendedGenerated_
                            insideConstrainedOut = d_32_appendedInside_
                            currentConstrainedOut = d_33_appendedCurrent_
                            d_2_tokensInSpan_ = (d_2_tokensInSpan_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

