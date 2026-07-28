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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. CRITICAL: Every arithmetic calculation MUST be wrapped in << >> delimiters. Example: 'She has 3 apples and buys 2 more, so she has <<3+2=5>> apples. Then she eats 1, leaving <<5-1=4>> apples. #### 4'. Always use << >> around computations. End with '#### <number>'.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanStart_: int
        d_2_spanStart_ = 0
        d_3_spanLimit_: int
        d_3_spanLimit_ = 20
        d_4_hasOpenedSpan_: bool
        d_4_hasOpenedSpan_ = insideConstrained
        d_5_consecutiveEmptyChunks_: int
        d_5_consecutiveEmptyChunks_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remaining_: int
                        d_6_remaining_ = (maxSteps) - (d_1_steps_)
                        d_7_chunkBudget_: int
                        if (d_6_remaining_) < (50):
                            d_7_chunkBudget_ = d_6_remaining_
                        elif True:
                            d_7_chunkBudget_ = 50
                        d_8_chunkedG_: _dafny.Seq
                        d_9_stoppedOpen_: bool
                        d_10_stoppedEos_: bool
                        d_11_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_8_chunkedG_ = out0_
                        d_9_stoppedOpen_ = out1_
                        d_10_stoppedEos_ = out2_
                        d_11_stepsUsed_ = out3_
                        generated = d_8_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                        if d_9_stoppedOpen_:
                            d_12_openedGenerated_: _dafny.Seq
                            d_13_openedInside_: bool
                            d_14_openedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_12_openedGenerated_ = out4_
                            d_13_openedInside_ = out5_
                            d_14_openedCurrent_ = out6_
                            generated = d_12_openedGenerated_
                            insideConstrainedOut = d_13_openedInside_
                            currentConstrainedOut = d_14_openedCurrent_
                            d_2_spanStart_ = d_1_steps_
                            d_4_hasOpenedSpan_ = True
                            d_5_consecutiveEmptyChunks_ = 0
                        elif d_10_stoppedEos_:
                            if (not(d_4_hasOpenedSpan_)) and (((d_1_steps_) + (1)) < (maxSteps)):
                                d_15_openedGenerated_: _dafny.Seq
                                d_16_openedInside_: bool
                                d_17_openedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_15_openedGenerated_ = out7_
                                d_16_openedInside_ = out8_
                                d_17_openedCurrent_ = out9_
                                generated = d_15_openedGenerated_
                                insideConstrainedOut = d_16_openedInside_
                                currentConstrainedOut = d_17_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanStart_ = d_1_steps_
                                d_4_hasOpenedSpan_ = True
                            elif True:
                                raise _dafny.Break("0")
                        elif (d_11_stepsUsed_) == (0):
                            d_5_consecutiveEmptyChunks_ = (d_5_consecutiveEmptyChunks_) + (1)
                            if (d_5_consecutiveEmptyChunks_) >= (2):
                                raise _dafny.Break("0")
                            if (not(d_4_hasOpenedSpan_)) and (((d_1_steps_) + (1)) < (maxSteps)):
                                d_18_openedGenerated_: _dafny.Seq
                                d_19_openedInside_: bool
                                d_20_openedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_18_openedGenerated_ = out10_
                                d_19_openedInside_ = out11_
                                d_20_openedCurrent_ = out12_
                                generated = d_18_openedGenerated_
                                insideConstrainedOut = d_19_openedInside_
                                currentConstrainedOut = d_20_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanStart_ = d_1_steps_
                                d_4_hasOpenedSpan_ = True
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_5_consecutiveEmptyChunks_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_21_closedGenerated_: _dafny.Seq
                        d_22_closedInside_: bool
                        d_23_closedCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_21_closedGenerated_ = out13_
                        d_22_closedInside_ = out14_
                        d_23_closedCurrent_ = out15_
                        generated = d_21_closedGenerated_
                        insideConstrainedOut = d_22_closedInside_
                        currentConstrainedOut = d_23_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif ((d_1_steps_) - (d_2_spanStart_)) >= (d_3_spanLimit_):
                        d_24_rolledGenerated_: _dafny.Seq
                        d_25_rolledCurrent_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: _dafny.Seq
                        out16_, out17_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_24_rolledGenerated_ = out16_
                        d_25_rolledCurrent_ = out17_
                        generated = d_24_rolledGenerated_
                        currentConstrainedOut = d_25_rolledCurrent_
                        insideConstrainedOut = True
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            if (d_1_steps_) < (maxSteps):
                                d_26_closedGenerated_: _dafny.Seq
                                d_27_closedInside_: bool
                                d_28_closedCurrent_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_26_closedGenerated_ = out18_
                                d_27_closedInside_ = out19_
                                d_28_closedCurrent_ = out20_
                                generated = d_26_closedGenerated_
                                insideConstrainedOut = d_27_closedInside_
                                currentConstrainedOut = d_28_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_29_stablePrefix_: _dafny.Seq
                        d_29_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_30_constrainedPrompt_: _dafny.Seq
                        d_30_constrainedPrompt_ = (prompt) + (d_29_stablePrefix_)
                        d_31_remaining_: int
                        d_31_remaining_ = (maxSteps) - (d_1_steps_)
                        d_32_spanRemaining_: int
                        d_32_spanRemaining_ = (d_3_spanLimit_) - ((d_1_steps_) - (d_2_spanStart_))
                        d_33_symBudget_: int
                        if (d_31_remaining_) < (d_32_spanRemaining_):
                            d_33_symBudget_ = d_31_remaining_
                        elif True:
                            d_33_symBudget_ = d_32_spanRemaining_
                        if (d_33_symBudget_) == (0):
                            d_33_symBudget_ = 1
                        d_34_symbolGenerated_: _dafny.Seq
                        d_35_symbolOut_: _dafny.Seq
                        d_36_hitEos_: bool
                        d_37_stepsUsed_: int
                        out21_: _dafny.Seq
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: int
                        out21_, out22_, out23_, out24_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_30_constrainedPrompt_, generated, currentConstrainedOut, d_33_symBudget_, eosToken)
                        d_34_symbolGenerated_ = out21_
                        d_35_symbolOut_ = out22_
                        d_36_hitEos_ = out23_
                        d_37_stepsUsed_ = out24_
                        generated = d_34_symbolGenerated_
                        currentConstrainedOut = d_35_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_37_stepsUsed_)
                        if d_36_hitEos_:
                            raise _dafny.Break("0")
                        if (d_37_stepsUsed_) == (0):
                            d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

