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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Every arithmetic computation must appear inside visible << >> delimiters, and the final answer should be the last visible << >> span.")))
        (d_0_helpers_).SetNonDeterministic(lm, False)
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openAfterMarker_: bool
        d_2_openAfterMarker_ = False
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 24
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_openAfterMarker_:
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
                            d_2_openAfterMarker_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_chunkBudget_: int
                            d_7_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            d_8_chunkedGenerated_: _dafny.Seq
                            d_9_stoppedOnOpenSpan_: bool
                            d_10_stoppedOnEos_: bool
                            d_11_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_8_chunkedGenerated_ = out3_
                            d_9_stoppedOnOpenSpan_ = out4_
                            d_10_stoppedOnEos_ = out5_
                            d_11_stepsUsed_ = out6_
                            generated = d_8_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                            if d_10_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_9_stoppedOnOpenSpan_:
                                d_12_enteredGenerated_: _dafny.Seq
                                d_13_enteredInside_: bool
                                d_14_enteredCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_12_enteredGenerated_ = out7_
                                d_13_enteredInside_ = out8_
                                d_14_enteredCurrent_ = out9_
                                generated = d_12_enteredGenerated_
                                insideConstrainedOut = d_13_enteredInside_
                                currentConstrainedOut = d_14_enteredCurrent_
                                d_2_openAfterMarker_ = False
                            elif True:
                                if (len(generated)) > (0):
                                    d_15_lastTok_: _dafny.Seq
                                    d_15_lastTok_ = (generated)[(len(generated)) - (1)]
                                    if ((d_15_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or ((d_15_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))):
                                        d_2_openAfterMarker_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
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
                    elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                        d_19_rolledGenerated_: _dafny.Seq
                        d_20_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_19_rolledGenerated_ = out13_
                        d_20_rolledCurrent_ = out14_
                        generated = d_19_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_20_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_22_next_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_22_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_23_appendedGenerated_: _dafny.Seq
                            d_24_appendedInside_: bool
                            d_25_appendedCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                            d_23_appendedGenerated_ = out16_
                            d_24_appendedInside_ = out17_
                            d_25_appendedCurrent_ = out18_
                            generated = d_23_appendedGenerated_
                            insideConstrainedOut = d_24_appendedInside_
                            currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

