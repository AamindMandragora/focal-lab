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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem. Show all arithmetic steps in plain text (no delimiters). At the very end, put ONLY the final answer expression inside << >> delimiters. Example: Step 1: 3 * 5 = 15. Step 2: 15 + 2 = 17. The answer is <<17>>. Use only numbers, variables like {x}, and operators (+, -, *, /, //) inside << >>. Only ONE final << >> at the end.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_maxConstrainedLen_: int
        d_2_maxConstrainedLen_ = 50
        d_3_hasOpenedSpan_: bool
        d_3_hasOpenedSpan_ = insideConstrained
        d_4_maxChunkSize_: int
        d_4_maxChunkSize_ = 300
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remaining_: int
                        d_5_remaining_ = (maxSteps) - (d_1_steps_)
                        if ((not(d_3_hasOpenedSpan_)) and ((d_5_remaining_) <= (20))) and ((d_5_remaining_) >= (2)):
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out0_
                            d_7_openedInside_ = out1_
                            d_8_openedCurrent_ = out2_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_3_hasOpenedSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_budget_: int
                            if (d_5_remaining_) < (d_4_maxChunkSize_):
                                d_9_budget_ = d_5_remaining_
                            elif True:
                                d_9_budget_ = d_4_maxChunkSize_
                            if (d_9_budget_) == (0):
                                raise _dafny.Break("0")
                            d_10_chunkedG_: _dafny.Seq
                            d_11_stoppedOpen_: bool
                            d_12_stoppedEos_: bool
                            d_13_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_budget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkedG_ = out3_
                            d_11_stoppedOpen_ = out4_
                            d_12_stoppedEos_ = out5_
                            d_13_stepsUsed_ = out6_
                            generated = d_10_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            if d_12_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_hasOpenedSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out7_
                        d_15_closedInside_ = out8_
                        d_16_closedCurrent_ = out9_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_2_maxConstrainedLen_):
                        d_17_rolledGenerated_: _dafny.Seq
                        d_18_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_17_rolledGenerated_ = out10_
                        d_18_rolledCurrent_ = out11_
                        generated = d_17_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_18_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_next_: _dafny.Seq
                        out12_: _dafny.Seq
                        out12_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_20_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_21_appendedGenerated_: _dafny.Seq
                            d_22_appendedInside_: bool
                            d_23_appendedCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_21_appendedGenerated_ = out13_
                            d_22_appendedInside_ = out14_
                            d_23_appendedCurrent_ = out15_
                            generated = d_21_appendedGenerated_
                            insideConstrainedOut = d_22_appendedInside_
                            currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_24_closedGenerated_: _dafny.Seq
                d_25_closedInside_: bool
                d_26_closedCurrent_: _dafny.Seq
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_24_closedGenerated_ = out16_
                d_25_closedInside_ = out17_
                d_26_closedCurrent_ = out18_
                generated = d_24_closedGenerated_
                insideConstrainedOut = d_25_closedInside_
                currentConstrainedOut = d_26_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_27_rolledGenerated_: _dafny.Seq
                d_28_rolledCurrent_: _dafny.Seq
                out19_: _dafny.Seq
                out20_: _dafny.Seq
                out19_, out20_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_27_rolledGenerated_ = out19_
                d_28_rolledCurrent_ = out20_
                generated = d_27_rolledGenerated_
                currentConstrainedOut = d_28_rolledCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                    d_29_closedGenerated2_: _dafny.Seq
                    d_30_closedInside2_: bool
                    d_31_closedCurrent2_: _dafny.Seq
                    out21_: _dafny.Seq
                    out22_: bool
                    out23_: _dafny.Seq
                    out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_29_closedGenerated2_ = out21_
                    d_30_closedInside2_ = out22_
                    d_31_closedCurrent2_ = out23_
                    generated = d_29_closedGenerated2_
                    insideConstrainedOut = d_30_closedInside2_
                    currentConstrainedOut = d_31_closedCurrent2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

