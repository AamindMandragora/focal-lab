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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Every arithmetic computation must appear inside visible << >> delimiters, and the final computation should also be inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_outsideSinceSpan_: int
        d_2_outsideSinceSpan_ = 0
        d_3_openAfter_: int
        d_3_openAfter_ = 16
        d_4_shortPrefixLimit_: int
        d_4_shortPrefixLimit_ = 3
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_outsideSinceSpan_) >= (d_3_openAfter_):
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
                            d_2_outsideSinceSpan_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_remaining_: int
                            d_8_remaining_ = (maxSteps) - (d_1_steps_)
                            d_9_chunkBudget_: int
                            if (d_8_remaining_) == (0):
                                d_9_chunkBudget_ = 0
                            elif (d_8_remaining_) < (8):
                                d_9_chunkBudget_ = d_8_remaining_
                            elif True:
                                d_9_chunkBudget_ = 8
                            if (d_9_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_10_chunkedGenerated_: _dafny.Seq
                                d_11_stoppedOnOpenSpan_: bool
                                d_12_stoppedOnEos_: bool
                                d_13_stepsUsed_: int
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: bool
                                out6_: int
                                out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_10_chunkedGenerated_ = out3_
                                d_11_stoppedOnOpenSpan_ = out4_
                                d_12_stoppedOnEos_ = out5_
                                d_13_stepsUsed_ = out6_
                                generated = d_10_chunkedGenerated_
                                d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                                d_2_outsideSinceSpan_ = (d_2_outsideSinceSpan_) + (d_13_stepsUsed_)
                                if d_12_stoppedOnEos_:
                                    raise _dafny.Break("0")
                                elif d_11_stoppedOnOpenSpan_:
                                    d_14_enteredGenerated_: _dafny.Seq
                                    d_15_enteredInside_: bool
                                    d_16_enteredCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_14_enteredGenerated_ = out7_
                                    d_15_enteredInside_ = out8_
                                    d_16_enteredCurrent_ = out9_
                                    generated = d_14_enteredGenerated_
                                    insideConstrainedOut = d_15_enteredInside_
                                    currentConstrainedOut = d_16_enteredCurrent_
                                    d_2_outsideSinceSpan_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
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
                        d_2_outsideSinceSpan_ = 0
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_20_stablePrefix_: _dafny.Seq
                        d_20_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (d_20_stablePrefix_)
                        d_22_nextTok_: _dafny.Seq
                        d_22_nextTok_ = eosToken
                        if (len(currentConstrainedOut)) < (d_4_shortPrefixLimit_):
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('8e0'), eosToken)
                            d_22_nextTok_ = out13_
                        elif True:
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_22_nextTok_ = out14_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_22_nextTok_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_23_appendedGenerated_: _dafny.Seq
                            d_24_appendedInside_: bool
                            d_25_appendedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextTok_)
                            d_23_appendedGenerated_ = out15_
                            d_24_appendedInside_ = out16_
                            d_25_appendedCurrent_ = out17_
                            generated = d_23_appendedGenerated_
                            insideConstrainedOut = d_24_appendedInside_
                            currentConstrainedOut = d_25_appendedCurrent_
                            d_2_outsideSinceSpan_ = 0
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

