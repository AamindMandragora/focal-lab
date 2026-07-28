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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve math word problems step by step. For EVERY arithmetic expression and the final answer, you MUST use << >> delimiters. Example: The total is <<3 + 4>>. Final answer: <<7>>. Always wrap symbolic expressions in << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_chunkSize_: int
        d_2_chunkSize_ = 8
        d_3_unconstrainedTokensSinceLastSpan_: int
        d_3_unconstrainedTokensSinceLastSpan_ = 0
        d_4_forceSpanThreshold_: int
        d_4_forceSpanThreshold_ = 60
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_3_unconstrainedTokensSinceLastSpan_) >= (d_4_forceSpanThreshold_)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                            d_5_openGenerated_: _dafny.Seq
                            d_6_openInside_: bool
                            d_7_openCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openGenerated_ = out0_
                            d_6_openInside_ = out1_
                            d_7_openCurrent_ = out2_
                            generated = d_5_openGenerated_
                            insideConstrainedOut = d_6_openInside_
                            currentConstrainedOut = d_7_openCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_unconstrainedTokensSinceLastSpan_ = 0
                        elif True:
                            d_8_remaining_: int
                            d_8_remaining_ = (maxSteps) - (d_1_steps_)
                            d_9_chunkMax_: int
                            if (d_8_remaining_) < (d_2_chunkSize_):
                                d_9_chunkMax_ = d_8_remaining_
                            elif True:
                                d_9_chunkMax_ = d_2_chunkSize_
                            if (d_9_chunkMax_) == (0):
                                raise _dafny.Break("0")
                            d_10_newGenerated_: _dafny.Seq
                            d_11_stoppedOnOpenSpan_: bool
                            d_12_stoppedOnEos_: bool
                            d_13_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkMax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_newGenerated_ = out3_
                            d_11_stoppedOnOpenSpan_ = out4_
                            d_12_stoppedOnEos_ = out5_
                            d_13_stepsUsed_ = out6_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            generated = d_10_newGenerated_
                            d_3_unconstrainedTokensSinceLastSpan_ = (d_3_unconstrainedTokensSinceLastSpan_) + (d_13_stepsUsed_)
                            if d_12_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOnOpenSpan_:
                                d_14_enteredGenerated_: _dafny.Seq
                                d_15_enteredInside_: bool
                                d_16_enteredCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_enteredGenerated_ = out7_
                                d_15_enteredInside_ = out8_
                                d_16_enteredCurrent_ = out9_
                                generated = d_14_enteredGenerated_
                                insideConstrainedOut = d_15_enteredInside_
                                currentConstrainedOut = d_16_enteredCurrent_
                                d_3_unconstrainedTokensSinceLastSpan_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        if ((len(currentConstrainedOut)) > (0)) and (((currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                            d_17_rolledGenerated_: _dafny.Seq
                            d_18_rolledCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: _dafny.Seq
                            out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_17_rolledGenerated_ = out10_
                            d_18_rolledCurrent_ = out11_
                            generated = d_17_rolledGenerated_
                            currentConstrainedOut = d_18_rolledCurrent_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_19_closedGenerated_: _dafny.Seq
                                d_20_closedInside_: bool
                                d_21_closedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_19_closedGenerated_ = out12_
                                d_20_closedInside_ = out13_
                                d_21_closedCurrent_ = out14_
                                generated = d_19_closedGenerated_
                                insideConstrainedOut = d_20_closedInside_
                                currentConstrainedOut = d_21_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_unconstrainedTokensSinceLastSpan_ = 0
                            elif ((d_1_steps_) + (1)) <= (maxSteps):
                                d_22_constrainedPrompt_: _dafny.Seq
                                d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_23_next_: _dafny.Seq
                                d_24_wasConstrained_: bool
                                out15_: _dafny.Seq
                                out16_: bool
                                out15_, out16_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_23_next_ = out15_
                                d_24_wasConstrained_ = out16_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_23_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_25_appendedGenerated_: _dafny.Seq
                                    d_26_appendedInside_: bool
                                    d_27_appendedCurrent_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                    d_25_appendedGenerated_ = out17_
                                    d_26_appendedInside_ = out18_
                                    d_27_appendedCurrent_ = out19_
                                    generated = d_25_appendedGenerated_
                                    insideConstrainedOut = d_26_appendedInside_
                                    currentConstrainedOut = d_27_appendedCurrent_
                        elif ((d_1_steps_) + (1)) <= (maxSteps):
                            d_28_closedGenerated_: _dafny.Seq
                            d_29_closedInside_: bool
                            d_30_closedCurrent_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_28_closedGenerated_ = out20_
                            d_29_closedInside_ = out21_
                            d_30_closedCurrent_ = out22_
                            generated = d_28_closedGenerated_
                            insideConstrainedOut = d_29_closedInside_
                            currentConstrainedOut = d_30_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_unconstrainedTokensSinceLastSpan_ = 0
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_31_constrainedPrompt_: _dafny.Seq
                        d_31_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_32_next_: _dafny.Seq
                        d_33_wasConstrained_: bool
                        out23_: _dafny.Seq
                        out24_: bool
                        out23_, out24_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_31_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_32_next_ = out23_
                        d_33_wasConstrained_ = out24_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_32_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_34_appendedGenerated_: _dafny.Seq
                            d_35_appendedInside_: bool
                            d_36_appendedCurrent_: _dafny.Seq
                            out25_: _dafny.Seq
                            out26_: bool
                            out27_: _dafny.Seq
                            out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next_)
                            d_34_appendedGenerated_ = out25_
                            d_35_appendedInside_ = out26_
                            d_36_appendedCurrent_ = out27_
                            generated = d_34_appendedGenerated_
                            insideConstrainedOut = d_35_appendedInside_
                            currentConstrainedOut = d_36_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

