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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write each intermediate calculation and the final answer inside << >> delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_remaining_: int
                        d_2_remaining_ = (maxSteps) - (d_1_steps_)
                        d_3_chunkBudget_: int
                        d_3_chunkBudget_ = 8
                        if (d_2_remaining_) < (d_3_chunkBudget_):
                            d_3_chunkBudget_ = d_2_remaining_
                        if (d_3_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_4_chunkGenerated_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkGenerated_ = out0_
                        d_5_stoppedOnOpenSpan_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        generated = d_4_chunkGenerated_
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOnOpenSpan_:
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
                        if (d_1_steps_) >= (maxSteps):
                            raise _dafny.Break("0")
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
                        if (d_1_steps_) >= (maxSteps):
                            d_17_rolledGenerated_: _dafny.Seq
                            d_18_rolledCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: _dafny.Seq
                            out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_17_rolledGenerated_ = out13_
                            d_18_rolledCurrent_ = out14_
                            generated = d_17_rolledGenerated_
                            currentConstrainedOut = d_18_rolledCurrent_
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_20_next_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            d_21_rolledGenerated_: _dafny.Seq
                            d_22_rolledCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: _dafny.Seq
                            out16_, out17_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_21_rolledGenerated_ = out16_
                            d_22_rolledCurrent_ = out17_
                            generated = d_21_rolledGenerated_
                            currentConstrainedOut = d_22_rolledCurrent_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_23_closedGenerated_: _dafny.Seq
                                d_24_closedInside_: bool
                                d_25_closedCurrent_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_23_closedGenerated_ = out18_
                                d_24_closedInside_ = out19_
                                d_25_closedCurrent_ = out20_
                                generated = d_23_closedGenerated_
                                insideConstrainedOut = d_24_closedInside_
                                currentConstrainedOut = d_25_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                        elif True:
                            d_26_valid_: bool
                            out21_: bool
                            out21_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_20_next_)
                            d_26_valid_ = out21_
                            if d_26_valid_:
                                d_27_appendedGenerated_: _dafny.Seq
                                d_28_appendedInside_: bool
                                d_29_appendedCurrent_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                d_27_appendedGenerated_ = out22_
                                d_28_appendedInside_ = out23_
                                d_29_appendedCurrent_ = out24_
                                generated = d_27_appendedGenerated_
                                insideConstrainedOut = d_28_appendedInside_
                                currentConstrainedOut = d_29_appendedCurrent_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_30_closedGenerated_: _dafny.Seq
                                    d_31_closedInside_: bool
                                    d_32_closedCurrent_: _dafny.Seq
                                    out25_: _dafny.Seq
                                    out26_: bool
                                    out27_: _dafny.Seq
                                    out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_30_closedGenerated_ = out25_
                                    d_31_closedInside_ = out26_
                                    d_32_closedCurrent_ = out27_
                                    generated = d_30_closedGenerated_
                                    insideConstrainedOut = d_31_closedInside_
                                    currentConstrainedOut = d_32_closedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

