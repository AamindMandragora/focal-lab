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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_chunkBudget_: int
                        d_2_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_3_chunkedGenerated_: _dafny.Seq
                        d_4_stoppedOnOpenSpan_: bool
                        d_5_stoppedOnEos_: bool
                        d_6_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_3_chunkedGenerated_ = out0_
                        d_4_stoppedOnOpenSpan_ = out1_
                        d_5_stoppedOnEos_ = out2_
                        d_6_stepsUsed_ = out3_
                        generated = d_3_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                        if d_5_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_4_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_7_isComplete_: bool
                        d_7_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_isComplete_:
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out4_
                            d_9_closedInside_ = out5_
                            d_10_closedCurrent_ = out6_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_deadEnd_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_11_deadEnd_ = out7_
                            if d_11_deadEnd_:
                                d_12_repaired_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_12_repaired_ = out8_
                                d_13_dropped_: int
                                d_13_dropped_ = (len(currentConstrainedOut)) - (len(d_12_repaired_))
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_13_dropped_):])
                                currentConstrainedOut = d_12_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_14_stablePrefix_: _dafny.Seq
                                d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                (lm).GenerateLogits(((prompt) + (d_14_stablePrefix_)) + (currentConstrainedOut))
                                d_15_validCount_: int
                                out9_: int
                                out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_15_validCount_ = out9_
                                if (len(currentConstrainedOut)) >= (4):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('12e0'))
                                elif True:
                                    if (d_15_validCount_) <= (2):
                                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('8e0'))
                                if (len(validTokenGroups)) > (0):
                                    d_16_flatPreferred_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_16_flatPreferred_ = out10_
                                    if (len(d_16_flatPreferred_)) > (0):
                                        d_17_candidates_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out11_ = (d_0_helpers_).TopValidCandidates(lm, parser, (prompt) + (d_14_stablePrefix_), currentConstrainedOut, 8, eosToken)
                                        d_17_candidates_ = out11_
                                        d_18_preferred_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out12_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_17_candidates_, d_16_flatPreferred_)
                                        d_18_preferred_ = out12_
                                        if (len(d_18_preferred_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_18_preferred_, _dafny.BigRational('1e0'))
                                d_19_next_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_14_stablePrefix_), currentConstrainedOut, eosToken)
                                d_19_next_ = out13_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_19_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_20_appendedGenerated_: _dafny.Seq
                                    d_21_appendedInside_: bool
                                    d_22_appendedCurrent_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                    d_20_appendedGenerated_ = out14_
                                    d_21_appendedInside_ = out15_
                                    d_22_appendedCurrent_ = out16_
                                    generated = d_20_appendedGenerated_
                                    insideConstrainedOut = d_21_appendedInside_
                                    currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

