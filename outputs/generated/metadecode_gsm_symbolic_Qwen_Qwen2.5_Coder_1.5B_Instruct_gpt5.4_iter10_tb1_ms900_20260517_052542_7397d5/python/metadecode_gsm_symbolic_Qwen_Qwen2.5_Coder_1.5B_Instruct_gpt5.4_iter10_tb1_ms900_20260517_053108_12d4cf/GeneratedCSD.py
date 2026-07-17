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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. Write each arithmetic computation inside visible << >> delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_remainingOutside_: int
                        d_2_remainingOutside_ = (maxSteps) - (d_1_steps_)
                        d_3_chunkBudget_: int
                        if (d_2_remainingOutside_) <= (8):
                            d_3_chunkBudget_ = d_2_remainingOutside_
                        elif True:
                            d_3_chunkBudget_ = 8
                        d_4_chunkedGenerated_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkedGenerated_ = out0_
                        d_5_stoppedOnOpenSpan_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        generated = d_4_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOnOpenSpan_:
                            d_8_observedGenerated_: _dafny.Seq
                            d_9_observedInside_: bool
                            d_10_observedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_8_observedGenerated_ = out4_
                            d_9_observedInside_ = out5_
                            d_10_observedCurrent_ = out6_
                            generated = d_8_observedGenerated_
                            insideConstrainedOut = d_9_observedInside_
                            currentConstrainedOut = d_10_observedCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_closedGenerated_: _dafny.Seq
                        d_12_closedInside_: bool
                        d_13_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_closedGenerated_ = out7_
                        d_12_closedInside_ = out8_
                        d_13_closedCurrent_ = out9_
                        generated = d_11_closedGenerated_
                        insideConstrainedOut = d_12_closedInside_
                        currentConstrainedOut = d_13_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_14_stablePrefix_: _dafny.Seq
                        d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                        d_16_remainingInside_: int
                        d_16_remainingInside_ = (maxSteps) - (d_1_steps_)
                        d_17_symbolBudget_: int
                        if (stepTokenBudget) == (0):
                            d_17_symbolBudget_ = 1
                        elif (stepTokenBudget) > (d_16_remainingInside_):
                            d_17_symbolBudget_ = d_16_remainingInside_
                        elif True:
                            d_17_symbolBudget_ = stepTokenBudget
                        d_18_symbolGenerated_: _dafny.Seq
                        d_19_symbolCurrent_: _dafny.Seq
                        d_20_hitEos_: bool
                        d_21_stepsUsed_: int
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: int
                        out10_, out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_15_constrainedPrompt_, generated, currentConstrainedOut, d_17_symbolBudget_, eosToken)
                        d_18_symbolGenerated_ = out10_
                        d_19_symbolCurrent_ = out11_
                        d_20_hitEos_ = out12_
                        d_21_stepsUsed_ = out13_
                        generated = d_18_symbolGenerated_
                        currentConstrainedOut = d_19_symbolCurrent_
                        d_1_steps_ = (d_1_steps_) + (d_21_stepsUsed_)
                        if d_20_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

