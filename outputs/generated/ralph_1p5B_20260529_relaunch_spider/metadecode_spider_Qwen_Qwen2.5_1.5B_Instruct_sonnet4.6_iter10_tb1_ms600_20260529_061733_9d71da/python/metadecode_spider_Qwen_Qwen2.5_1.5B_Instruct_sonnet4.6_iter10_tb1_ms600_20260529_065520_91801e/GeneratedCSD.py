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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write a single SQL query using lowercase keywords. Use spaces around parentheses like: count ( * ), max ( col ), min ( col ). Use table aliases and join conditions exactly as in the schema. Output only: SQL: <<query>>")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and (((d_1_steps_) + (6)) <= (maxSteps)):
            d_2_chunkGenerated_: _dafny.Seq
            d_3_stoppedOnOpenSpan_: bool
            d_4_stoppedOnEos_: bool
            d_5_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, 6, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_2_chunkGenerated_ = out0_
            d_3_stoppedOnOpenSpan_ = out1_
            d_4_stoppedOnEos_ = out2_
            d_5_stepsUsed_ = out3_
            generated = d_2_chunkGenerated_
            d_1_steps_ = (d_1_steps_) + (d_5_stepsUsed_)
            if d_3_stoppedOnOpenSpan_:
                d_6_enteredGenerated_: _dafny.Seq
                d_7_enteredInside_: bool
                d_8_enteredCurrent_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_6_enteredGenerated_ = out4_
                d_7_enteredInside_ = out5_
                d_8_enteredCurrent_ = out6_
                generated = d_6_enteredGenerated_
                insideConstrainedOut = d_7_enteredInside_
                currentConstrainedOut = d_8_enteredCurrent_
            elif (not(d_4_stoppedOnEos_)) and ((d_1_steps_) < (maxSteps)):
                d_9_openedGenerated_: _dafny.Seq
                d_10_openedInside_: bool
                d_11_openedCurrent_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_9_openedGenerated_ = out7_
                d_10_openedInside_ = out8_
                d_11_openedCurrent_ = out9_
                generated = d_9_openedGenerated_
                insideConstrainedOut = d_10_openedInside_
                currentConstrainedOut = d_11_openedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_12_openedGenerated_: _dafny.Seq
            d_13_openedInside_: bool
            d_14_openedCurrent_: _dafny.Seq
            out10_: _dafny.Seq
            out11_: bool
            out12_: _dafny.Seq
            out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_12_openedGenerated_ = out10_
            d_13_openedInside_ = out11_
            d_14_openedCurrent_ = out12_
            generated = d_12_openedGenerated_
            insideConstrainedOut = d_13_openedInside_
            currentConstrainedOut = d_14_openedCurrent_
            d_1_steps_ = (d_1_steps_) + (1)
        d_15_narrowThreshold_: int
        d_15_narrowThreshold_ = 16
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out13_
                        d_17_closedInside_ = out14_
                        d_18_closedCurrent_ = out15_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_next_: _dafny.Seq
                        out16_: _dafny.Seq
                        out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), d_15_narrowThreshold_, eosToken)
                        d_20_next_ = out16_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_21_appendedGenerated_: _dafny.Seq
                            d_22_appendedInside_: bool
                            d_23_appendedCurrent_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_21_appendedGenerated_ = out17_
                            d_22_appendedInside_ = out18_
                            d_23_appendedCurrent_ = out19_
                            generated = d_21_appendedGenerated_
                            insideConstrainedOut = d_22_appendedInside_
                            currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

