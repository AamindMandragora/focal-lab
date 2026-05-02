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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
                        d_2_remaining_: int
                        d_2_remaining_ = (maxSteps) - (d_1_steps_)
                        d_3_chunkBudget_: int
                        d_3_chunkBudget_ = d_2_remaining_
                        d_4_chunkedGenerated_: _dafny.Seq
                        d_5_stoppedOpen_: bool
                        d_6_stoppedEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkedGenerated_ = out0_
                        d_5_stoppedOpen_ = out1_
                        d_6_stoppedEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        generated = d_4_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_8_complete_: bool
                        d_8_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_complete_:
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out4_
                            d_10_closedInside_ = out5_
                            d_11_closedCurrent_ = out6_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_12_validCount_ = out7_
                            d_13_narrow_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 3)
                            d_13_narrow_ = out8_
                            d_14_candidateCap_: int
                            d_14_candidateCap_ = 3
                            if (d_12_validCount_) >= (5):
                                d_14_candidateCap_ = 5
                            d_15_candidates_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, d_14_candidateCap_, eosToken)
                            d_15_candidates_ = out9_
                            if (len(d_15_candidates_)) > (0):
                                if d_13_narrow_:
                                    (d_0_helpers_).BoostTokenLogits(lm, d_15_candidates_, _dafny.BigRational('8e0'))
                                elif True:
                                    (d_0_helpers_).BoostTokenLogits(lm, d_15_candidates_, _dafny.BigRational('3e0'))
                            d_16_next_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_16_next_ = out10_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_17_stablePrefix_: _dafny.Seq
                                d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_18_appendedGenerated_: _dafny.Seq
                                d_19_appendedInside_: bool
                                d_20_appendedCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, (d_17_stablePrefix_) + (currentConstrainedOut), currentConstrainedOut, d_16_next_)
                                d_18_appendedGenerated_ = out11_
                                d_19_appendedInside_ = out12_
                                d_20_appendedCurrent_ = out13_
                                generated = d_18_appendedGenerated_
                                insideConstrainedOut = d_19_appendedInside_
                                currentConstrainedOut = d_20_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

