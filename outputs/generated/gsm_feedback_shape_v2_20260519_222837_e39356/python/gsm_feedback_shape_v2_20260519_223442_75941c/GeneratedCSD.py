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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write every arithmetic computation inside visible delimiters << and >>, and end with a final <<computed answer>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spansClosed_: int
        d_2_spansClosed_ = 0
        d_3_bridgeLimit_: int
        d_3_bridgeLimit_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_spansClosed_) == (0):
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
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_remaining_: int
                            d_7_remaining_ = (maxSteps) - (d_1_steps_)
                            d_8_chunkBudget_: int
                            if (d_3_bridgeLimit_) < (d_7_remaining_):
                                d_8_chunkBudget_ = d_3_bridgeLimit_
                            elif True:
                                d_8_chunkBudget_ = d_7_remaining_
                            d_9_chunkedGenerated_: _dafny.Seq
                            d_10_stoppedOpen_: bool
                            d_11_stoppedEos_: bool
                            d_12_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_chunkedGenerated_ = out3_
                            d_10_stoppedOpen_ = out4_
                            d_11_stoppedEos_ = out5_
                            d_12_stepsUsed_ = out6_
                            generated = d_9_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                            if d_11_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_10_stoppedOpen_:
                                d_13_observedGenerated_: _dafny.Seq
                                d_14_observedInside_: bool
                                d_15_observedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_13_observedGenerated_ = out7_
                                d_14_observedInside_ = out8_
                                d_15_observedCurrent_ = out9_
                                generated = d_13_observedGenerated_
                                insideConstrainedOut = d_14_observedInside_
                                currentConstrainedOut = d_15_observedCurrent_
                            elif (d_1_steps_) < (maxSteps):
                                d_16_reopenedGenerated_: _dafny.Seq
                                d_17_reopenedInside_: bool
                                d_18_reopenedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_16_reopenedGenerated_ = out10_
                                d_17_reopenedInside_ = out11_
                                d_18_reopenedCurrent_ = out12_
                                generated = d_16_reopenedGenerated_
                                insideConstrainedOut = d_17_reopenedInside_
                                currentConstrainedOut = d_18_reopenedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_19_closedGenerated_: _dafny.Seq
                        d_20_closedInside_: bool
                        d_21_closedCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_19_closedGenerated_ = out13_
                        d_20_closedInside_ = out14_
                        d_21_closedCurrent_ = out15_
                        generated = d_19_closedGenerated_
                        insideConstrainedOut = d_20_closedInside_
                        currentConstrainedOut = d_21_closedCurrent_
                        d_2_spansClosed_ = (d_2_spansClosed_) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_23_validCount_: int
                        out16_: int
                        out16_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_23_validCount_ = out16_
                        d_24_next_: _dafny.Seq
                        d_24_next_ = eosToken
                        if (d_23_validCount_) <= (12):
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_24_next_ = out17_
                        elif True:
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_24_next_ = out18_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_24_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_25_appendedGenerated_: _dafny.Seq
                            d_26_appendedInside_: bool
                            d_27_appendedCurrent_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                            d_25_appendedGenerated_ = out19_
                            d_26_appendedInside_ = out20_
                            d_27_appendedCurrent_ = out21_
                            generated = d_25_appendedGenerated_
                            insideConstrainedOut = d_26_appendedInside_
                            currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

