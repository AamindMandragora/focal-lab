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
        d_2_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output a single valid SQL query. Format exactly as: SQL: <<your SQL query here>>. Use only the tables and columns from the schema provided. No explanation, no markdown, no extra text."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_2_guidance_)
        d_3_closeReserve_: int
        d_3_closeReserve_ = 2
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_4_maxChunk_: int
            d_4_maxChunk_ = 8
            if (d_4_maxChunk_) > ((maxSteps) - (d_1_steps_)):
                d_4_maxChunk_ = (maxSteps) - (d_1_steps_)
            if (d_4_maxChunk_) > (0):
                d_5_generatedOut_: _dafny.Seq
                d_6_stoppedOnOpenSpan_: bool
                d_7_stoppedOnEos_: bool
                d_8_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_maxChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_5_generatedOut_ = out0_
                d_6_stoppedOnOpenSpan_ = out1_
                d_7_stoppedOnEos_ = out2_
                d_8_stepsUsed_ = out3_
                d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                generated = d_5_generatedOut_
                if d_7_stoppedOnEos_:
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                elif d_6_stoppedOnOpenSpan_:
                    d_9_g2_: _dafny.Seq
                    d_10_i2_: bool
                    d_11_c2_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_9_g2_ = out4_
                    d_10_i2_ = out5_
                    d_11_c2_ = out6_
                    generated = d_9_g2_
                    insideConstrainedOut = d_10_i2_
                    currentConstrainedOut = d_11_c2_
                elif True:
                    if (d_1_steps_) < (maxSteps):
                        d_12_g2_: _dafny.Seq
                        d_13_i2_: bool
                        d_14_c2_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_12_g2_ = out7_
                        d_13_i2_ = out8_
                        d_14_c2_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        generated = d_12_g2_
                        insideConstrainedOut = d_13_i2_
                        currentConstrainedOut = d_14_c2_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_15_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_15_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_15_next_]))
                            if (d_15_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out11_
                        d_17_closedInside_ = out12_
                        d_18_closedCurrent_ = out13_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif ((d_1_steps_) + (d_3_closeReserve_)) >= (maxSteps):
                        d_19_rolledGenerated_: _dafny.Seq
                        d_20_rolledCurrent_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: _dafny.Seq
                        out14_, out15_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_19_rolledGenerated_ = out14_
                        d_20_rolledCurrent_ = out15_
                        generated = d_19_rolledGenerated_
                        currentConstrainedOut = d_20_rolledCurrent_
                        insideConstrainedOut = True
                        d_21_cg_: _dafny.Seq
                        d_22_ci_: bool
                        d_23_cc_: _dafny.Seq
                        d_24_closed_: bool
                        out16_: _dafny.Seq
                        out17_: bool
                        out18_: _dafny.Seq
                        out19_: bool
                        out16_, out17_, out18_, out19_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_21_cg_ = out16_
                        d_22_ci_ = out17_
                        d_23_cc_ = out18_
                        d_24_closed_ = out19_
                        d_1_steps_ = (d_1_steps_) + (1)
                        generated = d_21_cg_
                        insideConstrainedOut = d_22_ci_
                        currentConstrainedOut = d_23_cc_
                        raise _dafny.Break("0")
                    elif True:
                        d_25_constrainedPrompt_: _dafny.Seq
                        d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_26_next_: _dafny.Seq
                        out20_: _dafny.Seq
                        out20_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                        d_26_next_ = out20_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_26_next_) == (eosToken):
                            d_27_rolledGenerated_: _dafny.Seq
                            d_28_rolledCurrent_: _dafny.Seq
                            out21_: _dafny.Seq
                            out22_: _dafny.Seq
                            out21_, out22_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_27_rolledGenerated_ = out21_
                            d_28_rolledCurrent_ = out22_
                            generated = d_27_rolledGenerated_
                            currentConstrainedOut = d_28_rolledCurrent_
                            insideConstrainedOut = True
                            if (d_1_steps_) < (maxSteps):
                                d_29_cg_: _dafny.Seq
                                d_30_ci_: bool
                                d_31_cc_: _dafny.Seq
                                d_32_closed_: bool
                                out23_: _dafny.Seq
                                out24_: bool
                                out25_: _dafny.Seq
                                out26_: bool
                                out23_, out24_, out25_, out26_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_29_cg_ = out23_
                                d_30_ci_ = out24_
                                d_31_cc_ = out25_
                                d_32_closed_ = out26_
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_29_cg_
                                insideConstrainedOut = d_30_ci_
                                currentConstrainedOut = d_31_cc_
                            raise _dafny.Break("0")
                        elif True:
                            d_33_appendedGenerated_: _dafny.Seq
                            d_34_appendedInside_: bool
                            d_35_appendedCurrent_: _dafny.Seq
                            out27_: _dafny.Seq
                            out28_: bool
                            out29_: _dafny.Seq
                            out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                            d_33_appendedGenerated_ = out27_
                            d_34_appendedInside_ = out28_
                            d_35_appendedCurrent_ = out29_
                            generated = d_33_appendedGenerated_
                            insideConstrainedOut = d_34_appendedInside_
                            currentConstrainedOut = d_35_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

