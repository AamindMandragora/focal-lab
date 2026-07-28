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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem. After a brief setup sentence, immediately write your first expression as <<expression=value>>. Show ALL arithmetic steps as <<expr=value>>. End with <<answer>> then #### number.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_unconstrainedSinceLastSpan_: int
        d_2_unconstrainedSinceLastSpan_ = 0
        d_3_forceSpanAfter_: int
        d_3_forceSpanAfter_ = 60
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_unconstrainedSinceLastSpan_) >= (d_3_forceSpanAfter_):
                            if (d_1_steps_) < (maxSteps):
                                d_4_openGenerated_: _dafny.Seq
                                d_5_openInside_: bool
                                d_6_openCurrent_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_4_openGenerated_ = out0_
                                d_5_openInside_ = out1_
                                d_6_openCurrent_ = out2_
                                generated = d_4_openGenerated_
                                insideConstrainedOut = d_5_openInside_
                                currentConstrainedOut = d_6_openCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_unconstrainedSinceLastSpan_ = 0
                        elif True:
                            d_7_remaining_: int
                            d_7_remaining_ = (maxSteps) - (d_1_steps_)
                            d_8_chunkMax_: int
                            d_8_chunkMax_ = 12
                            if (d_7_remaining_) < (d_8_chunkMax_):
                                d_8_chunkMax_ = d_7_remaining_
                            if (d_8_chunkMax_) == (0):
                                raise _dafny.Break("0")
                            d_9_generatedOut_: _dafny.Seq
                            d_10_stoppedOnOpenSpan_: bool
                            d_11_stoppedOnEos_: bool
                            d_12_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkMax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_generatedOut_ = out3_
                            d_10_stoppedOnOpenSpan_ = out4_
                            d_11_stoppedOnEos_ = out5_
                            d_12_stepsUsed_ = out6_
                            d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                            generated = d_9_generatedOut_
                            d_2_unconstrainedSinceLastSpan_ = (d_2_unconstrainedSinceLastSpan_) + (d_12_stepsUsed_)
                            if d_11_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_10_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_unconstrainedSinceLastSpan_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        if (d_1_steps_) >= (maxSteps):
                            raise _dafny.Break("0")
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
                        d_2_unconstrainedSinceLastSpan_ = 0
                    elif True:
                        if (d_1_steps_) >= (maxSteps):
                            raise _dafny.Break("0")
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_17_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_17_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_17_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_18_appendedGenerated_: _dafny.Seq
                            d_19_appendedInside_: bool
                            d_20_appendedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                            d_18_appendedGenerated_ = out11_
                            d_19_appendedInside_ = out12_
                            d_20_appendedCurrent_ = out13_
                            generated = d_18_appendedGenerated_
                            insideConstrainedOut = d_19_appendedInside_
                            currentConstrainedOut = d_20_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

