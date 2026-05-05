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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_rollbackLimit_: int
        d_2_rollbackLimit_ = 64
        d_3_outsideChunkCap_: int
        d_3_outsideChunkCap_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remainingOutside_: int
                        d_4_remainingOutside_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkBudget_: int
                        if (d_4_remainingOutside_) <= (d_3_outsideChunkCap_):
                            d_5_chunkBudget_ = d_4_remainingOutside_
                        elif True:
                            d_5_chunkBudget_ = d_3_outsideChunkCap_
                        d_6_chunkedGenerated_: _dafny.Seq
                        d_7_stoppedOnOpenSpan_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkedGenerated_ = out0_
                        d_7_stoppedOnOpenSpan_ = out1_
                        d_8_stoppedOnEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOnOpenSpan_:
                            d_10_enteredGenerated_: _dafny.Seq
                            d_11_enteredInside_: bool
                            d_12_enteredCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_10_enteredGenerated_ = out4_
                            d_11_enteredInside_ = out5_
                            d_12_enteredCurrent_ = out6_
                            generated = d_10_enteredGenerated_
                            insideConstrainedOut = d_11_enteredInside_
                            currentConstrainedOut = d_12_enteredCurrent_
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
                    elif (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
                        d_16_rolledGenerated_: _dafny.Seq
                        d_17_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_16_rolledGenerated_ = out10_
                        d_17_rolledCurrent_ = out11_
                        generated = d_16_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_17_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_stablePrefix_: _dafny.Seq
                        d_18_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (d_18_stablePrefix_)
                        d_20_next_: _dafny.Seq
                        d_20_next_ = eosToken
                        if (len(currentConstrainedOut)) == (0):
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_20_next_ = out12_
                        elif True:
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_20_next_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_21_appendedGenerated_: _dafny.Seq
                            d_22_appendedInside_: bool
                            d_23_appendedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_21_appendedGenerated_ = out14_
                            d_22_appendedInside_ = out15_
                            d_23_appendedCurrent_ = out16_
                            generated = d_21_appendedGenerated_
                            insideConstrainedOut = d_22_appendedInside_
                            currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

