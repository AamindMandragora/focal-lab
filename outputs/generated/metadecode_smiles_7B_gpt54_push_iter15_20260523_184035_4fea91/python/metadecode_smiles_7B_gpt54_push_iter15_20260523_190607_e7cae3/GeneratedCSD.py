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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Produce a valid SMILES string for the requested molecular class. Use the constrained span only for the final SMILES answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_repetitionThreshold_: int
        d_3_repetitionThreshold_ = 6
        d_4_outsideChunkCap_: int
        d_4_outsideChunkCap_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remainingOutside_: int
                        d_5_remainingOutside_ = (maxSteps) - (d_1_steps_)
                        d_6_chunkBudget_: int
                        if (d_5_remainingOutside_) <= (d_4_outsideChunkCap_):
                            d_6_chunkBudget_ = d_5_remainingOutside_
                        elif True:
                            d_6_chunkBudget_ = d_4_outsideChunkCap_
                        d_7_chunkedGenerated_: _dafny.Seq
                        d_8_stoppedOnOpenSpan_: bool
                        d_9_stoppedOnEos_: bool
                        d_10_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_7_chunkedGenerated_ = out0_
                        d_8_stoppedOnOpenSpan_ = out1_
                        d_9_stoppedOnEos_ = out2_
                        d_10_stepsUsed_ = out3_
                        generated = d_7_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                        if d_9_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_8_stoppedOnOpenSpan_:
                            d_11_enteredGenerated_: _dafny.Seq
                            d_12_enteredInside_: bool
                            d_13_enteredCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_11_enteredGenerated_ = out4_
                            d_12_enteredInside_ = out5_
                            d_13_enteredCurrent_ = out6_
                            generated = d_11_enteredGenerated_
                            insideConstrainedOut = d_12_enteredInside_
                            currentConstrainedOut = d_13_enteredCurrent_
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
                    elif True:
                        d_17_stablePrefix_: _dafny.Seq
                        d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (d_17_stablePrefix_)
                        d_19_nextInside_: _dafny.Seq
                        d_19_nextInside_ = eosToken
                        if (len(currentConstrainedOut)) >= (d_3_repetitionThreshold_):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), currentConstrainedOut, _dafny.BigRational('3e0'), d_2_narrowThreshold_, eosToken)
                            d_19_nextInside_ = out10_
                        elif True:
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_19_nextInside_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_19_nextInside_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_20_appendedGenerated_: _dafny.Seq
                            d_21_appendedInside_: bool
                            d_22_appendedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_nextInside_)
                            d_20_appendedGenerated_ = out12_
                            d_21_appendedInside_ = out13_
                            d_22_appendedCurrent_ = out14_
                            generated = d_20_appendedGenerated_
                            insideConstrainedOut = d_21_appendedInside_
                            currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

