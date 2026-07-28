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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve each math problem step by step. For each calculation, wrap the expression inside << >>. Use exact variable names from the problem statement (no curly braces). Example: if the problem uses 'price1' and 'n1', write <<n1 * price1>>. Final answer: <<expression>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_chunkSize_: int
        d_2_chunkSize_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkBudget_: int
                        if (d_3_remaining_) < (d_2_chunkSize_):
                            d_4_chunkBudget_ = d_3_remaining_
                        elif True:
                            d_4_chunkBudget_ = d_2_chunkSize_
                        if (d_4_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_5_chunkedGenerated_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkedGenerated_ = out0_
                        d_6_stoppedOnOpenSpan_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        generated = d_5_chunkedGenerated_
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOnOpenSpan_:
                            d_9_enteredGenerated_: _dafny.Seq
                            d_10_enteredInside_: bool
                            d_11_enteredCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_9_enteredGenerated_ = out4_
                            d_10_enteredInside_ = out5_
                            d_11_enteredCurrent_ = out6_
                            generated = d_9_enteredGenerated_
                            insideConstrainedOut = d_10_enteredInside_
                            currentConstrainedOut = d_11_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out7_
                        d_13_closedInside_ = out8_
                        d_14_closedCurrent_ = out9_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_isDeadEnd_: bool
                        out10_: bool
                        out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_15_isDeadEnd_ = out10_
                        if d_15_isDeadEnd_:
                            d_16_rolledGenerated_: _dafny.Seq
                            d_17_rolledCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_16_rolledGenerated_ = out11_
                            d_17_rolledCurrent_ = out12_
                            generated = d_16_rolledGenerated_
                            currentConstrainedOut = d_17_rolledCurrent_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_18_closedGenerated_: _dafny.Seq
                                d_19_closedInside_: bool
                                d_20_closedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_18_closedGenerated_ = out13_
                                d_19_closedInside_ = out14_
                                d_20_closedCurrent_ = out15_
                                generated = d_18_closedGenerated_
                                insideConstrainedOut = d_19_closedInside_
                                currentConstrainedOut = d_20_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_21_stillDeadEnd_: bool
                                out16_: bool
                                out16_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                                d_21_stillDeadEnd_ = out16_
                                if d_21_stillDeadEnd_:
                                    d_22_constrainedPrompt2_: _dafny.Seq
                                    d_22_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_23_next2_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out17_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), 50, eosToken)
                                    d_23_next2_ = out17_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_24_constrainedPrompt3_: _dafny.Seq
                                    d_24_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_25_next3_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_24_constrainedPrompt3_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), 50, eosToken)
                                    d_25_next3_ = out18_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_25_next3_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                                        pass
                                    elif True:
                                        d_26_valid3_: bool
                                        out19_: bool
                                        out19_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_25_next3_)
                                        d_26_valid3_ = out19_
                                        if d_26_valid3_:
                                            d_27_appendedGenerated3_: _dafny.Seq
                                            d_28_appendedInside3_: bool
                                            d_29_appendedCurrent3_: _dafny.Seq
                                            out20_: _dafny.Seq
                                            out21_: bool
                                            out22_: _dafny.Seq
                                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next3_)
                                            d_27_appendedGenerated3_ = out20_
                                            d_28_appendedInside3_ = out21_
                                            d_29_appendedCurrent3_ = out22_
                                            generated = d_27_appendedGenerated3_
                                            insideConstrainedOut = d_28_appendedInside3_
                                            currentConstrainedOut = d_29_appendedCurrent3_
                        elif True:
                            d_30_constrainedPrompt_: _dafny.Seq
                            d_30_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_31_next_: _dafny.Seq
                            out23_: _dafny.Seq
                            out23_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_30_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), 50, eosToken)
                            d_31_next_ = out23_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_31_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_32_appendedGenerated_: _dafny.Seq
                                d_33_appendedInside_: bool
                                d_34_appendedCurrent_: _dafny.Seq
                                out24_: _dafny.Seq
                                out25_: bool
                                out26_: _dafny.Seq
                                out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_next_)
                                d_32_appendedGenerated_ = out24_
                                d_33_appendedInside_ = out25_
                                d_34_appendedCurrent_ = out26_
                                generated = d_32_appendedGenerated_
                                insideConstrainedOut = d_33_appendedInside_
                                currentConstrainedOut = d_34_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

