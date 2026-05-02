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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        insideConstrainedOut: bool = False
        currentConstrainedOut: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_spanOpened_: bool
            d_2_spanOpened_ = insideConstrained
            with _dafny.label("0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("0"):
                        if not(insideConstrainedOut):
                            if not(d_2_spanOpened_):
                                d_3_remaining_: int
                                d_3_remaining_ = (maxSteps) - (d_1_steps_)
                                d_4_chunkBudget_: int
                                d_4_chunkBudget_ = d_3_remaining_
                                if (3) < (d_4_chunkBudget_):
                                    d_4_chunkBudget_ = 3
                                d_5_chunkGenerated_: _dafny.Seq
                                d_6_stoppedOnOpen_: bool
                                d_7_stoppedOnEos_: bool
                                d_8_used_: int
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: bool
                                out3_: int
                                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_5_chunkGenerated_ = out0_
                                d_6_stoppedOnOpen_ = out1_
                                d_7_stoppedOnEos_ = out2_
                                d_8_used_ = out3_
                                generated = d_5_chunkGenerated_
                                d_1_steps_ = (d_1_steps_) + (d_8_used_)
                                if d_7_stoppedOnEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    if d_6_stoppedOnOpen_:
                                        insideConstrainedOut = True
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        d_2_spanOpened_ = True
                                    elif True:
                                        if (d_1_steps_) < (maxSteps):
                                            d_9_openedGenerated_: _dafny.Seq
                                            d_10_openedInside_: bool
                                            d_11_openedCurrent_: _dafny.Seq
                                            out4_: _dafny.Seq
                                            out5_: bool
                                            out6_: _dafny.Seq
                                            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                            d_9_openedGenerated_ = out4_
                                            d_10_openedInside_ = out5_
                                            d_11_openedCurrent_ = out6_
                                            generated = d_9_openedGenerated_
                                            insideConstrainedOut = d_10_openedInside_
                                            currentConstrainedOut = d_11_openedCurrent_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            d_2_spanOpened_ = True
                                        elif True:
                                            raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_12_stablePrefix_: _dafny.Seq
                            d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                            d_14_completeNow_: bool
                            d_14_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            d_15_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_15_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_15_next_) == (eosToken):
                                if d_14_completeNow_:
                                    if (d_1_steps_) < (maxSteps):
                                        d_16_closedGenerated_: _dafny.Seq
                                        d_17_closedInside_: bool
                                        d_18_closedCurrent_: _dafny.Seq
                                        out8_: _dafny.Seq
                                        out9_: bool
                                        out10_: _dafny.Seq
                                        out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_16_closedGenerated_ = out8_
                                        d_17_closedInside_ = out9_
                                        d_18_closedCurrent_ = out10_
                                        generated = d_16_closedGenerated_
                                        insideConstrainedOut = d_17_closedInside_
                                        currentConstrainedOut = d_18_closedCurrent_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                if not(d_14_completeNow_):
                                    d_19_appendedGenerated_: _dafny.Seq
                                    d_20_appendedInside_: bool
                                    d_21_appendedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                    d_19_appendedGenerated_ = out11_
                                    d_20_appendedInside_ = out12_
                                    d_21_appendedCurrent_ = out13_
                                    generated = d_19_appendedGenerated_
                                    insideConstrainedOut = d_20_appendedInside_
                                    currentConstrainedOut = d_21_appendedCurrent_
                                elif True:
                                    raise _dafny.Break("0")
                        pass
                pass
            cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

