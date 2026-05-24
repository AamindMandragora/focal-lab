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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one valid SMILES string for the requested molecular class. Avoid prose. If a constrained span is used, make it contain the whole molecule.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_observedAnySpan_: bool
        d_2_observedAnySpan_ = insideConstrained
        d_3_openAfter_: int
        d_3_openAfter_ = 12
        d_4_rollbackLimit_: int
        d_4_rollbackLimit_ = 48
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_observedAnySpan_)) and ((d_1_steps_) >= (d_3_openAfter_)):
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out0_
                            d_6_openedInside_ = out1_
                            d_7_openedCurrent_ = out2_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_2_observedAnySpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_remainingChunk_: int
                            d_8_remainingChunk_ = (maxSteps) - (d_1_steps_)
                            d_9_chunkBudget_: int
                            if (d_8_remainingChunk_) > (8):
                                d_9_chunkBudget_ = 8
                            elif True:
                                d_9_chunkBudget_ = d_8_remainingChunk_
                            d_10_chunkedGenerated_: _dafny.Seq
                            d_11_stoppedOnOpen_: bool
                            d_12_stoppedOnEos_: bool
                            d_13_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkedGenerated_ = out3_
                            d_11_stoppedOnOpen_ = out4_
                            d_12_stoppedOnEos_ = out5_
                            d_13_stepsUsed_ = out6_
                            generated = d_10_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            if d_12_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOnOpen_:
                                d_14_enteredGenerated_: _dafny.Seq
                                d_15_enteredInside_: bool
                                d_16_enteredCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_enteredGenerated_ = out7_
                                d_15_enteredInside_ = out8_
                                d_16_enteredCurrent_ = out9_
                                generated = d_14_enteredGenerated_
                                insideConstrainedOut = d_15_enteredInside_
                                currentConstrainedOut = d_16_enteredCurrent_
                                d_2_observedAnySpan_ = True
                    elif True:
                        d_17_isComplete_: bool
                        d_17_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_17_isComplete_:
                            d_18_closedGenerated_: _dafny.Seq
                            d_19_closedInside_: bool
                            d_20_closedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_18_closedGenerated_ = out10_
                            d_19_closedInside_ = out11_
                            d_20_closedCurrent_ = out12_
                            generated = d_18_closedGenerated_
                            insideConstrainedOut = d_19_closedInside_
                            currentConstrainedOut = d_20_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (len(currentConstrainedOut)) >= (d_4_rollbackLimit_):
                            d_21_rolledGenerated_: _dafny.Seq
                            d_22_rolledCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: _dafny.Seq
                            out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_21_rolledGenerated_ = out13_
                            d_22_rolledCurrent_ = out14_
                            generated = d_21_rolledGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_22_rolledCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_23_stablePrefix_: _dafny.Seq
                            d_23_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_24_constrainedPrompt_: _dafny.Seq
                            d_24_constrainedPrompt_ = (prompt) + (d_23_stablePrefix_)
                            d_25_next_: _dafny.Seq
                            d_26_wasConstrained_: bool
                            out15_: _dafny.Seq
                            out16_: bool
                            out15_, out16_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_25_next_ = out15_
                            d_26_wasConstrained_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_25_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_27_appendedGenerated_: _dafny.Seq
                                d_28_appendedInside_: bool
                                d_29_appendedCurrent_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                d_27_appendedGenerated_ = out17_
                                d_28_appendedInside_ = out18_
                                d_29_appendedCurrent_ = out19_
                                generated = d_27_appendedGenerated_
                                insideConstrainedOut = d_28_appendedInside_
                                currentConstrainedOut = d_29_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

