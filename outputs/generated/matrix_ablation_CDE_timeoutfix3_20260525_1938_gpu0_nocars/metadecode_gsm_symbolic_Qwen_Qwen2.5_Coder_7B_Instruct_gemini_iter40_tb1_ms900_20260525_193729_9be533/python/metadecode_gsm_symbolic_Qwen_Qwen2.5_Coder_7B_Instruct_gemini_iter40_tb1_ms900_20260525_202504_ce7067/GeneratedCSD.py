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
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out4_
                            insideConstrainedOut = out5_
                            currentConstrainedOut = out6_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_closedGenerated_: _dafny.Seq
                        d_8_closedInside_: bool
                        d_9_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedGenerated_ = out7_
                        d_8_closedInside_ = out8_
                        d_9_closedCurrent_ = out9_
                        generated = d_7_closedGenerated_
                        insideConstrainedOut = d_8_closedInside_
                        currentConstrainedOut = d_9_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_10_stablePrefix_: _dafny.Seq
                        d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                        d_12_validCount_: int
                        out10_: int
                        out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_12_validCount_ = out10_
                        d_13_narrowThreshold_: int
                        d_13_narrowThreshold_ = 20
                        if (d_12_validCount_) <= (d_13_narrowThreshold_):
                            d_14_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_14_next_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_15_appendedGenerated_: _dafny.Seq
                                d_16_appendedInside_: bool
                                d_17_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                d_15_appendedGenerated_ = out12_
                                d_16_appendedInside_ = out13_
                                d_17_appendedCurrent_ = out14_
                                generated = d_15_appendedGenerated_
                                insideConstrainedOut = d_16_appendedInside_
                                currentConstrainedOut = d_17_appendedCurrent_
                        elif True:
                            d_18_remainingBudget_: int
                            d_18_remainingBudget_ = (maxSteps) - (d_1_steps_)
                            d_19_symbolBudget_: int
                            d_19_symbolBudget_ = 16
                            if (d_19_symbolBudget_) > (d_18_remainingBudget_):
                                d_19_symbolBudget_ = d_18_remainingBudget_
                            d_20_symbolGenerated_: _dafny.Seq
                            d_21_symbolOut_: _dafny.Seq
                            d_22_hitEos_: bool
                            d_23_stepsUsed_: int
                            out15_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: int
                            out15_, out16_, out17_, out18_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_11_constrainedPrompt_, generated, currentConstrainedOut, d_19_symbolBudget_, eosToken)
                            d_20_symbolGenerated_ = out15_
                            d_21_symbolOut_ = out16_
                            d_22_hitEos_ = out17_
                            d_23_stepsUsed_ = out18_
                            generated = d_20_symbolGenerated_
                            currentConstrainedOut = d_21_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_23_stepsUsed_)
                            if d_22_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

