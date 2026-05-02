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
        d_2_finalizeThreshold_: int
        d_2_finalizeThreshold_ = 2
        d_3_narrowFinalizeCount_: int
        d_3_narrowFinalizeCount_ = 1
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_1_steps_) + (1)) > (maxSteps):
                            raise _dafny.Break("0")
                        elif True:
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
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_7_complete_: bool
                        d_7_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_complete_:
                            d_8_remaining_: int
                            d_8_remaining_ = (maxSteps) - (d_1_steps_)
                            d_9_narrow_: bool
                            out3_: bool
                            out3_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, d_3_narrowFinalizeCount_)
                            d_9_narrow_ = out3_
                            if ((d_8_remaining_) <= (d_2_finalizeThreshold_)) or (d_9_narrow_):
                                if ((d_1_steps_) + (1)) > (maxSteps):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_10_closedGenerated_: _dafny.Seq
                                    d_11_closedInside_: bool
                                    d_12_closedCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_10_closedGenerated_ = out4_
                                    d_11_closedInside_ = out5_
                                    d_12_closedCurrent_ = out6_
                                    generated = d_10_closedGenerated_
                                    insideConstrainedOut = d_11_closedInside_
                                    currentConstrainedOut = d_12_closedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_13_stablePrefix_: _dafny.Seq
                                d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_14_constrainedPrompt_: _dafny.Seq
                                d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                                d_15_next_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_15_next_ = out7_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_stillIncomplete_: bool
                                    d_16_stillIncomplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if d_16_stillIncomplete_:
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_17_appendedGenerated_: _dafny.Seq
                                        d_18_appendedInside_: bool
                                        d_19_appendedCurrent_: _dafny.Seq
                                        out8_: _dafny.Seq
                                        out9_: bool
                                        out10_: _dafny.Seq
                                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                        d_17_appendedGenerated_ = out8_
                                        d_18_appendedInside_ = out9_
                                        d_19_appendedCurrent_ = out10_
                                        generated = d_17_appendedGenerated_
                                        insideConstrainedOut = d_18_appendedInside_
                                        currentConstrainedOut = d_19_appendedCurrent_
                        elif True:
                            d_20_stablePrefix2_: _dafny.Seq
                            d_20_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_21_constrainedPrompt2_: _dafny.Seq
                            d_21_constrainedPrompt2_ = (prompt) + (d_20_stablePrefix2_)
                            d_22_next2_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_21_constrainedPrompt2_, currentConstrainedOut, eosToken)
                            d_22_next2_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_23_appendedGenerated2_: _dafny.Seq
                                d_24_appendedInside2_: bool
                                d_25_appendedCurrent2_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next2_)
                                d_23_appendedGenerated2_ = out12_
                                d_24_appendedInside2_ = out13_
                                d_25_appendedCurrent2_ = out14_
                                generated = d_23_appendedGenerated2_
                                insideConstrainedOut = d_24_appendedInside2_
                                currentConstrainedOut = d_25_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

