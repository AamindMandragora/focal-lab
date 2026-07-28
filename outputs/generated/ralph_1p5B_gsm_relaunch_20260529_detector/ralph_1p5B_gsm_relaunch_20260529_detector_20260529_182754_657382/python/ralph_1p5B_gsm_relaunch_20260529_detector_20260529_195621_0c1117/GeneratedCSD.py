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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve math word problems step by step. For each calculation, wrap the symbolic expression in << >>. Example: <<n1 + n2>>. Always use << >> for the final answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_chunkSize_: int
        d_2_chunkSize_ = 8
        d_3_unconstrainedTokensSinceLastSpan_: int
        d_3_unconstrainedTokensSinceLastSpan_ = 0
        d_4_forceSpanThreshold_: int
        d_4_forceSpanThreshold_ = 50
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
                        if ((d_1_steps_) + (1)) <= (maxSteps):
                            d_17_closedGenerated_: _dafny.Seq
                            d_18_closedInside_: bool
                            d_19_closedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_17_closedGenerated_ = out10_
                            d_18_closedInside_ = out11_
                            d_19_closedCurrent_ = out12_
                            generated = d_17_closedGenerated_
                            insideConstrainedOut = d_18_closedInside_
                            currentConstrainedOut = d_19_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_next_: _dafny.Seq
                        d_22_wasConstrained_: bool
                        out13_: _dafny.Seq
                        out14_: bool
                        out13_, out14_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_21_next_ = out13_
                        d_22_wasConstrained_ = out14_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_21_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (d_21_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                            if (d_1_steps_) < (maxSteps):
                                d_23_forced_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_23_forced_ = out15_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_23_forced_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif (d_23_forced_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                    d_24_appendedGenerated_: _dafny.Seq
                                    d_25_appendedInside_: bool
                                    d_26_appendedCurrent_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_forced_)
                                    d_24_appendedGenerated_ = out16_
                                    d_25_appendedInside_ = out17_
                                    d_26_appendedCurrent_ = out18_
                                    generated = d_24_appendedGenerated_
                                    insideConstrainedOut = d_25_appendedInside_
                                    currentConstrainedOut = d_26_appendedCurrent_
                                elif True:
                                    d_27_appendedGenerated_: _dafny.Seq
                                    d_28_appendedInside_: bool
                                    d_29_appendedCurrent_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_forced_)
                                    d_27_appendedGenerated_ = out19_
                                    d_28_appendedInside_ = out20_
                                    d_29_appendedCurrent_ = out21_
                                    generated = d_27_appendedGenerated_
                                    insideConstrainedOut = d_28_appendedInside_
                                    currentConstrainedOut = d_29_appendedCurrent_
                        elif True:
                            d_30_appendedGenerated_: _dafny.Seq
                            d_31_appendedInside_: bool
                            d_32_appendedCurrent_: _dafny.Seq
                            out22_: _dafny.Seq
                            out23_: bool
                            out24_: _dafny.Seq
                            out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_30_appendedGenerated_ = out22_
                            d_31_appendedInside_ = out23_
                            d_32_appendedCurrent_ = out24_
                            generated = d_30_appendedGenerated_
                            insideConstrainedOut = d_31_appendedInside_
                            currentConstrainedOut = d_32_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

