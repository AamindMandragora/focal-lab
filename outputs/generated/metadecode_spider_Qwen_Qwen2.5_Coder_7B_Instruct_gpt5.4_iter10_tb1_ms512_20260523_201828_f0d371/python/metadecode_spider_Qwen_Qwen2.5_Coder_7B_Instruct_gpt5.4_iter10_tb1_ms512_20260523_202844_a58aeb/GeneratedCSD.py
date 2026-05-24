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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query in the required constrained span format. Do not add explanation, prose, or multiple alternatives.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_chunkBudget_: int
                        d_3_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkedGenerated_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkedGenerated_ = out1_
                        d_5_stoppedOnOpenSpan_ = out2_
                        d_6_stoppedOnEos_ = out3_
                        d_7_stepsUsed_ = out4_
                        generated = d_4_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOnOpenSpan_:
                            d_8_enteredGenerated_: _dafny.Seq
                            d_9_enteredInside_: bool
                            d_10_enteredCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_8_enteredGenerated_ = out5_
                            d_9_enteredInside_ = out6_
                            d_10_enteredCurrent_ = out7_
                            generated = d_8_enteredGenerated_
                            insideConstrainedOut = d_9_enteredInside_
                            currentConstrainedOut = d_10_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_closedGenerated_: _dafny.Seq
                        d_12_closedInside_: bool
                        d_13_closedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_closedGenerated_ = out8_
                        d_12_closedInside_ = out9_
                        d_13_closedCurrent_ = out10_
                        generated = d_11_closedGenerated_
                        insideConstrainedOut = d_12_closedInside_
                        currentConstrainedOut = d_13_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_14_stablePrefix_: _dafny.Seq
                        d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                        d_16_next_: _dafny.Seq
                        d_16_next_ = eosToken
                        if (len(currentConstrainedOut)) < (2):
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_16_next_ = out11_
                        elif True:
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_flatGroups_, _dafny.BigRational('2e0'), 12, eosToken)
                            d_16_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_16_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_17_appendedGenerated_: _dafny.Seq
                            d_18_appendedInside_: bool
                            d_19_appendedCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                            d_17_appendedGenerated_ = out13_
                            d_18_appendedInside_ = out14_
                            d_19_appendedCurrent_ = out15_
                            generated = d_17_appendedGenerated_
                            insideConstrainedOut = d_18_appendedInside_
                            currentConstrainedOut = d_19_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

