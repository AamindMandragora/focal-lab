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
        d_2_spanOpenedByStrategy_: bool
        d_2_spanOpenedByStrategy_ = insideConstrained
        d_3_prefixDelay_: int
        if (maxSteps) >= (3):
            d_3_prefixDelay_ = 2
        elif True:
            d_3_prefixDelay_ = 1
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_4_complete_: bool
                        d_4_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_complete_:
                            d_5_closedGenerated_: _dafny.Seq
                            d_6_closedInside_: bool
                            d_7_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_closedGenerated_ = out0_
                            d_6_closedInside_ = out1_
                            d_7_closedCurrent_ = out2_
                            generated = d_5_closedGenerated_
                            insideConstrainedOut = d_6_closedInside_
                            currentConstrainedOut = d_7_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            if ((d_1_steps_) + (1)) >= (maxSteps):
                                raise _dafny.Break("0")
                            elif True:
                                d_8_stablePrefix_: _dafny.Seq
                                d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_9_constrainedPrompt_: _dafny.Seq
                                d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                                d_10_next_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_10_next_ = out3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_10_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_11_appendedGenerated_: _dafny.Seq
                                    d_12_appendedInside_: bool
                                    d_13_appendedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                    d_11_appendedGenerated_ = out4_
                                    d_12_appendedInside_ = out5_
                                    d_13_appendedCurrent_ = out6_
                                    generated = d_11_appendedGenerated_
                                    insideConstrainedOut = d_12_appendedInside_
                                    currentConstrainedOut = d_13_appendedCurrent_
                    elif True:
                        if d_2_spanOpenedByStrategy_:
                            raise _dafny.Break("0")
                        elif True:
                            if (d_1_steps_) < (d_3_prefixDelay_):
                                d_14_remaining_: int
                                d_14_remaining_ = (maxSteps) - (d_1_steps_)
                                d_15_want_: int
                                d_15_want_ = (d_3_prefixDelay_) - (d_1_steps_)
                                d_16_chunk_: int
                                if (d_15_want_) <= (d_14_remaining_):
                                    d_16_chunk_ = d_15_want_
                                elif True:
                                    d_16_chunk_ = d_14_remaining_
                                d_17_chunkGenerated_: _dafny.Seq
                                d_18_stoppedOnOpenSpan_: bool
                                d_19_stoppedOnEos_: bool
                                d_20_stepsUsed_: int
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: bool
                                out10_: int
                                out7_, out8_, out9_, out10_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_16_chunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_17_chunkGenerated_ = out7_
                                d_18_stoppedOnOpenSpan_ = out8_
                                d_19_stoppedOnEos_ = out9_
                                d_20_stepsUsed_ = out10_
                                generated = d_17_chunkGenerated_
                                d_1_steps_ = (d_1_steps_) + (d_20_stepsUsed_)
                                if (d_18_stoppedOnOpenSpan_) or (d_19_stoppedOnEos_):
                                    raise _dafny.Break("0")
                            elif True:
                                if ((d_1_steps_) + (2)) <= (maxSteps):
                                    d_21_openedGenerated_: _dafny.Seq
                                    d_22_openedInside_: bool
                                    d_23_openedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_21_openedGenerated_ = out11_
                                    d_22_openedInside_ = out12_
                                    d_23_openedCurrent_ = out13_
                                    generated = d_21_openedGenerated_
                                    insideConstrainedOut = d_22_openedInside_
                                    currentConstrainedOut = d_23_openedCurrent_
                                    d_2_spanOpenedByStrategy_ = True
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

