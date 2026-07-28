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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem. First, reason through the problem step by step using plain arithmetic (compute actual numbers, not variables). Then write the final numeric answer inside << >> delimiters. Example: <<42>> or <<6+7>>. Use only digits and +, -, *, /, (, ) inside << >>. Do not write variables like x, n, or placeholders inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_maxConstrainedLen_: int
        d_2_maxConstrainedLen_ = 32
        d_3_reasoningBudget_: int
        if (maxSteps) > (700):
            d_3_reasoningBudget_ = 650
        elif (maxSteps) > (100):
            d_3_reasoningBudget_ = (maxSteps) - (50)
        elif True:
            d_3_reasoningBudget_ = _dafny.euclidian_division(maxSteps, 2)
        d_4_spanOpened_: bool
        d_4_spanOpened_ = insideConstrained
        d_5_chunkSize_: int
        d_5_chunkSize_ = 100
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remaining_: int
                        d_6_remaining_ = (maxSteps) - (d_1_steps_)
                        if (((d_1_steps_) >= (d_3_reasoningBudget_)) and (not(d_4_spanOpened_))) and ((d_6_remaining_) >= (5)):
                            d_7_openedGenerated_: _dafny.Seq
                            d_8_openedInside_: bool
                            d_9_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_openedGenerated_ = out0_
                            d_8_openedInside_ = out1_
                            d_9_openedCurrent_ = out2_
                            generated = d_7_openedGenerated_
                            insideConstrainedOut = d_8_openedInside_
                            currentConstrainedOut = d_9_openedCurrent_
                            d_4_spanOpened_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_thisBudget_: int
                            if (d_6_remaining_) < (d_5_chunkSize_):
                                d_10_thisBudget_ = d_6_remaining_
                            elif True:
                                d_10_thisBudget_ = d_5_chunkSize_
                            if (d_10_thisBudget_) == (0):
                                raise _dafny.Break("0")
                            d_11_chunkedG_: _dafny.Seq
                            d_12_stoppedOpen_: bool
                            d_13_stoppedEos_: bool
                            d_14_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_thisBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_11_chunkedG_ = out3_
                            d_12_stoppedOpen_ = out4_
                            d_13_stoppedEos_ = out5_
                            d_14_stepsUsed_ = out6_
                            generated = d_11_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                            if d_13_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_12_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_4_spanOpened_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out7_
                        d_16_closedInside_ = out8_
                        d_17_closedCurrent_ = out9_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif (len(currentConstrainedOut)) >= (d_2_maxConstrainedLen_):
                        d_18_rolledGenerated_: _dafny.Seq
                        d_19_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_18_rolledGenerated_ = out10_
                        d_19_rolledCurrent_ = out11_
                        generated = d_18_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_19_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_next_: _dafny.Seq
                        out12_: _dafny.Seq
                        out12_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_21_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_21_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_22_appendedGenerated_: _dafny.Seq
                            d_23_appendedInside_: bool
                            d_24_appendedCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_22_appendedGenerated_ = out13_
                            d_23_appendedInside_ = out14_
                            d_24_appendedCurrent_ = out15_
                            generated = d_22_appendedGenerated_
                            insideConstrainedOut = d_23_appendedInside_
                            currentConstrainedOut = d_24_appendedCurrent_
                    pass
            pass
        if insideConstrainedOut:
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_25_closedGenerated_: _dafny.Seq
                d_26_closedInside_: bool
                d_27_closedCurrent_: _dafny.Seq
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_25_closedGenerated_ = out16_
                d_26_closedInside_ = out17_
                d_27_closedCurrent_ = out18_
                generated = d_25_closedGenerated_
                insideConstrainedOut = d_26_closedInside_
                currentConstrainedOut = d_27_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
            elif (d_1_steps_) < (maxSteps):
                d_28_rolledGenerated_: _dafny.Seq
                d_29_rolledCurrent_: _dafny.Seq
                out19_: _dafny.Seq
                out20_: _dafny.Seq
                out19_, out20_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_28_rolledGenerated_ = out19_
                d_29_rolledCurrent_ = out20_
                generated = d_28_rolledGenerated_
                currentConstrainedOut = d_29_rolledCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                    d_30_closedGenerated2_: _dafny.Seq
                    d_31_closedInside2_: bool
                    d_32_closedCurrent2_: _dafny.Seq
                    out21_: _dafny.Seq
                    out22_: bool
                    out23_: _dafny.Seq
                    out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_30_closedGenerated2_ = out21_
                    d_31_closedInside2_ = out22_
                    d_32_closedCurrent2_ = out23_
                    generated = d_30_closedGenerated2_
                    insideConstrainedOut = d_31_closedInside2_
                    currentConstrainedOut = d_32_closedCurrent2_
                    d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

