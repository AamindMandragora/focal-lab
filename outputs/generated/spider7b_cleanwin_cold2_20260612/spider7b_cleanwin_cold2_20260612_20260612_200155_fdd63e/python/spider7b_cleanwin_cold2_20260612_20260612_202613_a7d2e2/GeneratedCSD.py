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
        d_2_guidance_: _dafny.Seq
        d_2_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write the simplest, most direct SQL query that answers the question. Use only tables and columns from the schema. Output format: SQL: <<your SQL here>>. No markdown, no explanation."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_2_guidance_)
        if not(insideConstrainedOut):
            d_3_chunkBudget_: int
            if (maxSteps) > (20):
                d_3_chunkBudget_ = 20
            elif True:
                d_3_chunkBudget_ = maxSteps
            d_4_generatedOut_: _dafny.Seq
            d_5_stoppedOnOpenSpan_: bool
            d_6_stoppedOnEos_: bool
            d_7_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_4_generatedOut_ = out0_
            d_5_stoppedOnOpenSpan_ = out1_
            d_6_stoppedOnEos_ = out2_
            d_7_stepsUsed_ = out3_
            d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
            generated = d_4_generatedOut_
            if d_6_stoppedOnEos_:
                cost = d_1_steps_
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            if d_5_stoppedOnOpenSpan_:
                d_8_g2_: _dafny.Seq
                d_9_i2_: bool
                d_10_c2_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_8_g2_ = out4_
                d_9_i2_ = out5_
                d_10_c2_ = out6_
                generated = d_8_g2_
                insideConstrainedOut = d_9_i2_
                currentConstrainedOut = d_10_c2_
            elif True:
                if (d_1_steps_) < (maxSteps):
                    d_11_g2_: _dafny.Seq
                    d_12_i2_: bool
                    d_13_c2_: _dafny.Seq
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_11_g2_ = out7_
                    d_12_i2_ = out8_
                    d_13_c2_ = out9_
                    d_1_steps_ = (d_1_steps_) + (1)
                    generated = d_11_g2_
                    insideConstrainedOut = d_12_i2_
                    currentConstrainedOut = d_13_c2_
                elif True:
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_14_closeReserve_: int
        d_14_closeReserve_ = 2
        with _dafny.label("0"):
            while (((d_1_steps_) + (d_14_closeReserve_)) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) > (0)):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out10_
                        d_16_closedInside_ = out11_
                        d_17_closedCurrent_ = out12_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                        d_19_next_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) > (0)):
                                if (d_1_steps_) < (maxSteps):
                                    d_20_closedGenerated_: _dafny.Seq
                                    d_21_closedInside_: bool
                                    d_22_closedCurrent_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_20_closedGenerated_ = out14_
                                    d_21_closedInside_ = out15_
                                    d_22_closedCurrent_ = out16_
                                    generated = d_20_closedGenerated_
                                    insideConstrainedOut = d_21_closedInside_
                                    currentConstrainedOut = d_22_closedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif (len(currentConstrainedOut)) > (0):
                                d_23_rolledGenerated_: _dafny.Seq
                                d_24_rolledCurrent_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: _dafny.Seq
                                out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_23_rolledGenerated_ = out17_
                                d_24_rolledCurrent_ = out18_
                                generated = d_23_rolledGenerated_
                                currentConstrainedOut = d_24_rolledCurrent_
                                insideConstrainedOut = True
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) > (0)):
                                    if (d_1_steps_) < (maxSteps):
                                        d_25_closedGenerated_: _dafny.Seq
                                        d_26_closedInside_: bool
                                        d_27_closedCurrent_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_25_closedGenerated_ = out19_
                                        d_26_closedInside_ = out20_
                                        d_27_closedCurrent_ = out21_
                                        generated = d_25_closedGenerated_
                                        insideConstrainedOut = d_26_closedInside_
                                        currentConstrainedOut = d_27_closedCurrent_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_28_appendedGenerated_: _dafny.Seq
                                d_29_appendedInside_: bool
                                d_30_appendedCurrent_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_28_appendedGenerated_ = out22_
                                d_29_appendedInside_ = out23_
                                d_30_appendedCurrent_ = out24_
                                generated = d_28_appendedGenerated_
                                insideConstrainedOut = d_29_appendedInside_
                                currentConstrainedOut = d_30_appendedCurrent_
                            elif (len(currentConstrainedOut)) == (0):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_19_next_]))
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) > (0)):
                d_31_closedGenerated_: _dafny.Seq
                d_32_closedInside_: bool
                d_33_closedCurrent_: _dafny.Seq
                out25_: _dafny.Seq
                out26_: bool
                out27_: _dafny.Seq
                out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_31_closedGenerated_ = out25_
                d_32_closedInside_ = out26_
                d_33_closedCurrent_ = out27_
                generated = d_31_closedGenerated_
                insideConstrainedOut = d_32_closedInside_
                currentConstrainedOut = d_33_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
            elif (len(currentConstrainedOut)) > (0):
                d_34_rolledGenerated_: _dafny.Seq
                d_35_rolledCurrent_: _dafny.Seq
                out28_: _dafny.Seq
                out29_: _dafny.Seq
                out28_, out29_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_34_rolledGenerated_ = out28_
                d_35_rolledCurrent_ = out29_
                generated = d_34_rolledGenerated_
                currentConstrainedOut = d_35_rolledCurrent_
                insideConstrainedOut = True
                if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) > (0))) and ((d_1_steps_) < (maxSteps)):
                    d_36_closedGenerated_: _dafny.Seq
                    d_37_closedInside_: bool
                    d_38_closedCurrent_: _dafny.Seq
                    out30_: _dafny.Seq
                    out31_: bool
                    out32_: _dafny.Seq
                    out30_, out31_, out32_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_36_closedGenerated_ = out30_
                    d_37_closedInside_ = out31_
                    d_38_closedCurrent_ = out32_
                    generated = d_36_closedGenerated_
                    insideConstrainedOut = d_37_closedInside_
                    currentConstrainedOut = d_38_closedCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

