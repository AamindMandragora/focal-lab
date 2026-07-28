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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step. For every arithmetic computation and the final answer, wrap the expression in << >> delimiters like <<2+3=5>>. Use many short << >> spans rather than one long span. After the last computation write '#### <answer>'.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanStart_: int
        d_2_spanStart_ = 0
        d_3_spanLimit_: int
        d_3_spanLimit_ = 20
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkBudget_: int
                        if (d_4_remaining_) < (40):
                            d_5_chunkBudget_ = d_4_remaining_
                        elif True:
                            d_5_chunkBudget_ = 40
                        d_6_chunkedG_: _dafny.Seq
                        d_7_stoppedOpen_: bool
                        d_8_stoppedEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkedG_ = out0_
                        d_7_stoppedOpen_ = out1_
                        d_8_stoppedEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOpen_:
                            d_10_openedGenerated_: _dafny.Seq
                            d_11_openedInside_: bool
                            d_12_openedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_10_openedGenerated_ = out4_
                            d_11_openedInside_ = out5_
                            d_12_openedCurrent_ = out6_
                            generated = d_10_openedGenerated_
                            insideConstrainedOut = d_11_openedInside_
                            currentConstrainedOut = d_12_openedCurrent_
                            d_2_spanStart_ = d_1_steps_
                        elif (d_9_stepsUsed_) == (0):
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_closedGenerated_: _dafny.Seq
                        d_14_closedInside_: bool
                        d_15_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_closedGenerated_ = out7_
                        d_14_closedInside_ = out8_
                        d_15_closedCurrent_ = out9_
                        generated = d_13_closedGenerated_
                        insideConstrainedOut = d_14_closedInside_
                        currentConstrainedOut = d_15_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif ((d_1_steps_) - (d_2_spanStart_)) >= (d_3_spanLimit_):
                        d_16_rolledGenerated_: _dafny.Seq
                        d_17_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_16_rolledGenerated_ = out10_
                        d_17_rolledCurrent_ = out11_
                        generated = d_16_rolledGenerated_
                        currentConstrainedOut = d_17_rolledCurrent_
                        insideConstrainedOut = True
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            if (d_1_steps_) < (maxSteps):
                                d_18_closedGenerated_: _dafny.Seq
                                d_19_closedInside_: bool
                                d_20_closedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_18_closedGenerated_ = out12_
                                d_19_closedInside_ = out13_
                                d_20_closedCurrent_ = out14_
                                generated = d_18_closedGenerated_
                                insideConstrainedOut = d_19_closedInside_
                                currentConstrainedOut = d_20_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_21_stablePrefix_: _dafny.Seq
                        d_21_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (d_21_stablePrefix_)
                        d_23_remaining_: int
                        d_23_remaining_ = (maxSteps) - (d_1_steps_)
                        d_24_spanRemaining_: int
                        d_24_spanRemaining_ = (d_3_spanLimit_) - ((d_1_steps_) - (d_2_spanStart_))
                        d_25_symBudget_: int
                        if (d_23_remaining_) < (d_24_spanRemaining_):
                            d_25_symBudget_ = d_23_remaining_
                        elif True:
                            d_25_symBudget_ = d_24_spanRemaining_
                        if (d_25_symBudget_) == (0):
                            d_25_symBudget_ = 1
                        d_26_symbolGenerated_: _dafny.Seq
                        d_27_symbolOut_: _dafny.Seq
                        d_28_hitEos_: bool
                        d_29_stepsUsed_: int
                        out15_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: bool
                        out18_: int
                        out15_, out16_, out17_, out18_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_22_constrainedPrompt_, generated, currentConstrainedOut, d_25_symBudget_, eosToken)
                        d_26_symbolGenerated_ = out15_
                        d_27_symbolOut_ = out16_
                        d_28_hitEos_ = out17_
                        d_29_stepsUsed_ = out18_
                        generated = d_26_symbolGenerated_
                        currentConstrainedOut = d_27_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_29_stepsUsed_)
                        if d_28_hitEos_:
                            raise _dafny.Break("0")
                        if (d_29_stepsUsed_) == (0):
                            d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

