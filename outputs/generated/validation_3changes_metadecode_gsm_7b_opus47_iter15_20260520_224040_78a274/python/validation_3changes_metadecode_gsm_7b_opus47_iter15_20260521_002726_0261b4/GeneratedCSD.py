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
                        d_2_chunkBudget_: int
                        d_2_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_3_chunkedG_: _dafny.Seq
                        d_4_stoppedOpen_: bool
                        d_5_stoppedEos_: bool
                        d_6_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_3_chunkedG_ = out0_
                        d_4_stoppedOpen_ = out1_
                        d_5_stoppedEos_ = out2_
                        d_6_stepsUsed_ = out3_
                        generated = d_3_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                        if d_5_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_4_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif (d_6_stepsUsed_) == (0):
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_closedGenerated_: _dafny.Seq
                        d_8_closedInside_: bool
                        d_9_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedGenerated_ = out4_
                        d_8_closedInside_ = out5_
                        d_9_closedCurrent_ = out6_
                        generated = d_7_closedGenerated_
                        insideConstrainedOut = d_8_closedInside_
                        currentConstrainedOut = d_9_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_10_equalsCount_: int
                        out7_: int
                        out7_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                        d_10_equalsCount_ = out7_
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        if (d_10_equalsCount_) > (0):
                            d_12_remaining_: int
                            d_12_remaining_ = (maxSteps) - (d_1_steps_)
                            d_13_symbolBudget_: int
                            if (d_12_remaining_) < (8):
                                d_13_symbolBudget_ = d_12_remaining_
                            elif True:
                                d_13_symbolBudget_ = 8
                            if (d_13_symbolBudget_) == (0):
                                raise _dafny.Break("0")
                            d_14_symbolGenerated_: _dafny.Seq
                            d_15_symbolOut_: _dafny.Seq
                            d_16_hitEos_: bool
                            d_17_stepsUsed_: int
                            out8_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: int
                            out8_, out9_, out10_, out11_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_11_constrainedPrompt_, generated, currentConstrainedOut, d_13_symbolBudget_, eosToken)
                            d_14_symbolGenerated_ = out8_
                            d_15_symbolOut_ = out9_
                            d_16_hitEos_ = out10_
                            d_17_stepsUsed_ = out11_
                            generated = d_14_symbolGenerated_
                            currentConstrainedOut = d_15_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_17_stepsUsed_)
                            if d_16_hitEos_:
                                raise _dafny.Break("0")
                            if (d_17_stepsUsed_) == (0):
                                raise _dafny.Break("0")
                        elif True:
                            d_18_nGated_: _dafny.Seq
                            d_19_wasConstrained_: bool
                            out12_: _dafny.Seq
                            out13_: bool
                            out12_, out13_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_18_nGated_ = out12_
                            d_19_wasConstrained_ = out13_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_nGated_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_20_appendedGenerated_: _dafny.Seq
                                d_21_appendedInside_: bool
                                d_22_appendedCurrent_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_nGated_)
                                d_20_appendedGenerated_ = out14_
                                d_21_appendedInside_ = out15_
                                d_22_appendedCurrent_ = out16_
                                generated = d_20_appendedGenerated_
                                insideConstrainedOut = d_21_appendedInside_
                                currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

