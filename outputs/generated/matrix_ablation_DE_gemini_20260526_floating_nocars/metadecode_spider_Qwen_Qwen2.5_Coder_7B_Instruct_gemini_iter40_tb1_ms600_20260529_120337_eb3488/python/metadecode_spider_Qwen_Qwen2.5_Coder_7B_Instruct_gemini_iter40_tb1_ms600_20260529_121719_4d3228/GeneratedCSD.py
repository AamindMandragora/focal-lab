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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_remaining__budget_: int
                        d_2_remaining__budget_ = (maxSteps) - (d_1_steps_)
                        d_3_chunk__budget_: int
                        if (10) < (d_2_remaining__budget_):
                            d_3_chunk__budget_ = 10
                        elif True:
                            d_3_chunk__budget_ = d_2_remaining__budget_
                        d_4_generatedOut_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunk__budget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_generatedOut_ = out0_
                        d_5_stoppedOnOpenSpan_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        generated = d_4_generatedOut_
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        if d_5_stoppedOnOpenSpan_:
                            d_8_enteredGenerated_: _dafny.Seq
                            d_9_enteredInside_: bool
                            d_10_enteredCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_8_enteredGenerated_ = out4_
                            d_9_enteredInside_ = out5_
                            d_10_enteredCurrent_ = out6_
                            generated = d_8_enteredGenerated_
                            insideConstrainedOut = d_9_enteredInside_
                            currentConstrainedOut = d_10_enteredCurrent_
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_11_openedGenerated_: _dafny.Seq
                                d_12_openedInside_: bool
                                d_13_openedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_11_openedGenerated_ = out7_
                                d_12_openedInside_ = out8_
                                d_13_openedCurrent_ = out9_
                                generated = d_11_openedGenerated_
                                insideConstrainedOut = d_12_openedInside_
                                currentConstrainedOut = d_13_openedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out10_
                        d_15_closedInside_ = out11_
                        d_16_closedCurrent_ = out12_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_17_remaining__budget_: int
                        d_17_remaining__budget_ = (maxSteps) - (d_1_steps_)
                        d_18_symbol__budget_: int
                        if (30) < (d_17_remaining__budget_):
                            d_18_symbol__budget_ = 30
                        elif True:
                            d_18_symbol__budget_ = d_17_remaining__budget_
                        if (d_18_symbol__budget_) > (0):
                            d_19_constrainedPrompt_: _dafny.Seq
                            d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_20_genOut_: _dafny.Seq
                            d_21_currentOut_: _dafny.Seq
                            d_22_hitEos_: bool
                            d_23_stepsUsed_: int
                            out13_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: int
                            out13_, out14_, out15_, out16_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_19_constrainedPrompt_, generated, currentConstrainedOut, d_18_symbol__budget_, eosToken)
                            d_20_genOut_ = out13_
                            d_21_currentOut_ = out14_
                            d_22_hitEos_ = out15_
                            d_23_stepsUsed_ = out16_
                            d_1_steps_ = (d_1_steps_) + (d_23_stepsUsed_)
                            generated = d_20_genOut_
                            currentConstrainedOut = d_21_currentOut_
                            if d_22_hitEos_:
                                raise _dafny.Break("0")
                        elif True:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

