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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_rollbackLimit_: int
        d_2_rollbackLimit_ = 40
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_chunkBudget_: int
                        if ((maxSteps) - (d_1_steps_)) >= (3):
                            d_3_chunkBudget_ = 3
                        elif True:
                            d_3_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_3_chunkBudget_) > (0):
                            d_4_chunkedG_: _dafny.Seq
                            d_5_stoppedOpen_: bool
                            d_6_stoppedEos_: bool
                            d_7_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_4_chunkedG_ = out0_
                            d_5_stoppedOpen_ = out1_
                            d_6_stoppedEos_ = out2_
                            d_7_stepsUsed_ = out3_
                            generated = d_4_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                            if d_6_stoppedEos_:
                                raise _dafny.Break("0")
                            if d_5_stoppedOpen_:
                                d_8___v0_: _dafny.Seq
                                d_9_observedInside_: bool
                                d_10_observedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_8___v0_ = out4_
                                d_9_observedInside_ = out5_
                                d_10_observedCurrent_ = out6_
                                insideConstrainedOut = d_9_observedInside_
                                currentConstrainedOut = d_10_observedCurrent_
                            elif (d_1_steps_) < (maxSteps):
                                d_11_openedGenerated_: _dafny.Seq
                                d_12_openedInside_: bool
                                d_13_openedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_11_openedGenerated_ = out7_
                                d_12_openedInside_ = out8_
                                d_13_openedCurrent_ = out9_
                                generated = d_11_openedGenerated_
                                insideConstrainedOut = d_12_openedInside_
                                currentConstrainedOut = d_13_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out10_
                        d_15_closedInside_ = out11_
                        d_16_closedCurrent_ = out12_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
                        d_17_rolledGenerated_: _dafny.Seq
                        d_18_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_17_rolledGenerated_ = out13_
                        d_18_rolledCurrent_ = out14_
                        generated = d_17_rolledGenerated_
                        currentConstrainedOut = d_18_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_penaltyTokens_: _dafny.Seq
                        d_20_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken])
                        d_21_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_20_penaltyTokens_, _dafny.BigRational('1e1'), 1000, eosToken)
                        d_21_next_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_21_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_22_appendedGenerated_: _dafny.Seq
                            d_23_appendedInside_: bool
                            d_24_appendedCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_22_appendedGenerated_ = out16_
                            d_23_appendedInside_ = out17_
                            d_24_appendedCurrent_ = out18_
                            generated = d_22_appendedGenerated_
                            insideConstrainedOut = d_23_appendedInside_
                            currentConstrainedOut = d_24_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

