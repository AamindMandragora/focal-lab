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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. For EVERY intermediate calculation, write it inside << >> like this: <<expr = value>>. The final answer MUST be inside << >> delimiters. Always use << and >> around any arithmetic expression or numeric result. Example: There are <<3 * 4 = 12>> apples, so the answer is <<12>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_chunkSize_: int
        d_2_chunkSize_ = 8
        d_3_freeTokensSinceLastSpan_: int
        d_3_freeTokensSinceLastSpan_ = 0
        d_4_forceOpenThreshold_: int
        d_4_forceOpenThreshold_ = 60
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_3_freeTokensSinceLastSpan_) >= (d_4_forceOpenThreshold_)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                            d_5_newGenerated_: _dafny.Seq
                            d_6_newInside_: bool
                            d_7_newCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_newGenerated_ = out0_
                            d_6_newInside_ = out1_
                            d_7_newCurrent_ = out2_
                            generated = d_5_newGenerated_
                            insideConstrainedOut = d_6_newInside_
                            currentConstrainedOut = d_7_newCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_freeTokensSinceLastSpan_ = 0
                        elif ((d_1_steps_) + (d_2_chunkSize_)) <= (maxSteps):
                            d_8_chunkGenerated_: _dafny.Seq
                            d_9_stoppedOnOpenSpan_: bool
                            d_10_stoppedOnEos_: bool
                            d_11_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_8_chunkGenerated_ = out3_
                            d_9_stoppedOnOpenSpan_ = out4_
                            d_10_stoppedOnEos_ = out5_
                            d_11_stepsUsed_ = out6_
                            d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                            generated = d_8_chunkGenerated_
                            d_3_freeTokensSinceLastSpan_ = (d_3_freeTokensSinceLastSpan_) + (d_11_stepsUsed_)
                            if d_10_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_9_stoppedOnOpenSpan_:
                                d_12_newGenerated_: _dafny.Seq
                                d_13_newInside_: bool
                                d_14_newCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_12_newGenerated_ = out7_
                                d_13_newInside_ = out8_
                                d_14_newCurrent_ = out9_
                                generated = d_12_newGenerated_
                                insideConstrainedOut = d_13_newInside_
                                currentConstrainedOut = d_14_newCurrent_
                                d_3_freeTokensSinceLastSpan_ = 0
                        elif True:
                            d_15_next_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_15_next_ = out10_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_freeTokensSinceLastSpan_ = (d_3_freeTokensSinceLastSpan_) + (1)
                            if (d_15_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_15_next_]))
                                if (d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_16_newGenerated_: _dafny.Seq
                                    d_17_newInside_: bool
                                    d_18_newCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_16_newGenerated_ = out11_
                                    d_17_newInside_ = out12_
                                    d_18_newCurrent_ = out13_
                                    generated = d_16_newGenerated_
                                    insideConstrainedOut = d_17_newInside_
                                    currentConstrainedOut = d_18_newCurrent_
                                    d_3_freeTokensSinceLastSpan_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_19_closedGenerated_: _dafny.Seq
                        d_20_closedInside_: bool
                        d_21_closedCurrent_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_19_closedGenerated_ = out14_
                        d_20_closedInside_ = out15_
                        d_21_closedCurrent_ = out16_
                        generated = d_19_closedGenerated_
                        insideConstrainedOut = d_20_closedInside_
                        currentConstrainedOut = d_21_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_23_next_: _dafny.Seq
                        out17_: _dafny.Seq
                        out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_23_next_ = out17_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_23_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_24_appendedGenerated_: _dafny.Seq
                            d_25_appendedInside_: bool
                            d_26_appendedCurrent_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: bool
                            out20_: _dafny.Seq
                            out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                            d_24_appendedGenerated_ = out18_
                            d_25_appendedInside_ = out19_
                            d_26_appendedCurrent_ = out20_
                            generated = d_24_appendedGenerated_
                            insideConstrainedOut = d_25_appendedInside_
                            currentConstrainedOut = d_26_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

