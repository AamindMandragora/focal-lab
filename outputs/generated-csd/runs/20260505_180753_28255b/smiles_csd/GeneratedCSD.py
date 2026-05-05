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
        d_2_openedAnySpan_: bool
        d_2_openedAnySpan_ = (insideConstrained) or ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in (generatedPrefix))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_2_openedAnySpan_):
                            d_3_openedGenerated_: _dafny.Seq
                            d_4_openedInside_: bool
                            d_5_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_3_openedGenerated_ = out0_
                            d_4_openedInside_ = out1_
                            d_5_openedCurrent_ = out2_
                            generated = d_3_openedGenerated_
                            insideConstrainedOut = d_4_openedInside_
                            currentConstrainedOut = d_5_openedCurrent_
                            d_2_openedAnySpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_chunkBudget_: int
                            d_6_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            d_7_chunkedGenerated_: _dafny.Seq
                            d_8_stoppedOnOpenSpan_: bool
                            d_9_stoppedOnEos_: bool
                            d_10_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_7_chunkedGenerated_ = out3_
                            d_8_stoppedOnOpenSpan_ = out4_
                            d_9_stoppedOnEos_ = out5_
                            d_10_stepsUsed_ = out6_
                            generated = d_7_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                            if d_9_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_8_stoppedOnOpenSpan_:
                                d_11_enteredGenerated_: _dafny.Seq
                                d_12_enteredInside_: bool
                                d_13_enteredCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_11_enteredGenerated_ = out7_
                                d_12_enteredInside_ = out8_
                                d_13_enteredCurrent_ = out9_
                                generated = d_11_enteredGenerated_
                                insideConstrainedOut = d_12_enteredInside_
                                currentConstrainedOut = d_13_enteredCurrent_
                                d_2_openedAnySpan_ = True
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
                    elif True:
                        d_17_stablePrefix_: _dafny.Seq
                        d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (d_17_stablePrefix_)
                        d_19_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_19_next_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_20_appendedGenerated_: _dafny.Seq
                            d_21_appendedInside_: bool
                            d_22_appendedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
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

