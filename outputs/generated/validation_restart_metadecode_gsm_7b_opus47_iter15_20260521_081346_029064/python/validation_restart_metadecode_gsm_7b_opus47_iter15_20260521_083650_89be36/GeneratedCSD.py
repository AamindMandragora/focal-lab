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
        d_2_freeTokensSinceLastSpan_: int
        d_2_freeTokensSinceLastSpan_ = 0
        d_3_forceOpenThreshold_: int
        d_3_forceOpenThreshold_ = 40
        d_4_spansOpened_: int
        d_4_spansOpened_ = 0
        d_5_maxSpansToForce_: int
        d_5_maxSpansToForce_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (((d_2_freeTokensSinceLastSpan_) >= (d_3_forceOpenThreshold_)) and ((d_4_spansOpened_) < (d_5_maxSpansToForce_))) and (((maxSteps) - (d_1_steps_)) >= (4)):
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out0_
                            d_7_openedInside_ = out1_
                            d_8_openedCurrent_ = out2_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_freeTokensSinceLastSpan_ = 0
                            d_4_spansOpened_ = (d_4_spansOpened_) + (1)
                        elif True:
                            d_9_chunkBudget_: int
                            d_9_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            if (d_9_chunkBudget_) > (16):
                                d_9_chunkBudget_ = 16
                            d_10_chunkedG_: _dafny.Seq
                            d_11_stoppedOpen_: bool
                            d_12_stoppedEos_: bool
                            d_13_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkedG_ = out3_
                            d_11_stoppedOpen_ = out4_
                            d_12_stoppedEos_ = out5_
                            d_13_stepsUsed_ = out6_
                            generated = d_10_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            if d_12_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_freeTokensSinceLastSpan_ = 0
                                d_4_spansOpened_ = (d_4_spansOpened_) + (1)
                            elif (d_13_stepsUsed_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_2_freeTokensSinceLastSpan_ = (d_2_freeTokensSinceLastSpan_) + (d_13_stepsUsed_)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out7_
                        d_15_closedInside_ = out8_
                        d_16_closedCurrent_ = out9_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_freeTokensSinceLastSpan_ = 0
                    elif True:
                        d_17_stablePrefix_: _dafny.Seq
                        d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (d_17_stablePrefix_)
                        d_19_remaining_: int
                        d_19_remaining_ = (maxSteps) - (d_1_steps_)
                        d_20_symbolBudget_: int
                        if (d_19_remaining_) > (24):
                            d_20_symbolBudget_ = 24
                        elif True:
                            d_20_symbolBudget_ = d_19_remaining_
                        d_21_symbolGenerated_: _dafny.Seq
                        d_22_symbolOut_: _dafny.Seq
                        d_23_hitEos_: bool
                        d_24_stepsUsed_: int
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: int
                        out10_, out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_18_constrainedPrompt_, generated, currentConstrainedOut, d_20_symbolBudget_, eosToken)
                        d_21_symbolGenerated_ = out10_
                        d_22_symbolOut_ = out11_
                        d_23_hitEos_ = out12_
                        d_24_stepsUsed_ = out13_
                        generated = d_21_symbolGenerated_
                        currentConstrainedOut = d_22_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_24_stepsUsed_)
                        if d_23_hitEos_:
                            raise _dafny.Break("0")
                        if (d_24_stepsUsed_) == (0):
                            if (d_1_steps_) < (maxSteps):
                                d_25_constrainedPrompt2_: _dafny.Seq
                                d_25_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_26_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_25_constrainedPrompt2_, currentConstrainedOut, eosToken)
                                d_26_next_ = out14_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_26_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_27_appendedGenerated_: _dafny.Seq
                                    d_28_appendedInside_: bool
                                    d_29_appendedCurrent_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                    d_27_appendedGenerated_ = out15_
                                    d_28_appendedInside_ = out16_
                                    d_29_appendedCurrent_ = out17_
                                    generated = d_27_appendedGenerated_
                                    insideConstrainedOut = d_28_appendedInside_
                                    currentConstrainedOut = d_29_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

