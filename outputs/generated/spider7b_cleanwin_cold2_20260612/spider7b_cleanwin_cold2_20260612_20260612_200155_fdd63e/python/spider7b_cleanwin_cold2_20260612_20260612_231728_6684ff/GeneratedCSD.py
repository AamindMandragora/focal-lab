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
        d_2_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<your SQL query>>. Use only schema-provided table and column names. No explanation, no markdown."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_2_guidance_)
        d_3_closeReserve_: int
        d_3_closeReserve_ = 2
        if (not(insideConstrainedOut)) and (((d_1_steps_) + (5)) < (maxSteps)):
            d_4_genOut_: _dafny.Seq
            d_5_stoppedOnOpen_: bool
            d_6_stoppedOnEos_: bool
            d_7_chunkSteps_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, 5, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_4_genOut_ = out0_
            d_5_stoppedOnOpen_ = out1_
            d_6_stoppedOnEos_ = out2_
            d_7_chunkSteps_ = out3_
            d_1_steps_ = (d_1_steps_) + (d_7_chunkSteps_)
            if d_6_stoppedOnEos_:
                generated = d_4_genOut_
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            generated = d_4_genOut_
            if d_5_stoppedOnOpen_:
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
            elif (d_1_steps_) < (maxSteps):
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
        elif (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_14_g2_: _dafny.Seq
            d_15_i2_: bool
            d_16_c2_: _dafny.Seq
            out10_: _dafny.Seq
            out11_: bool
            out12_: _dafny.Seq
            out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_14_g2_ = out10_
            d_15_i2_ = out11_
            d_16_c2_ = out12_
            d_1_steps_ = (d_1_steps_) + (1)
            generated = d_14_g2_
            insideConstrainedOut = d_15_i2_
            currentConstrainedOut = d_16_c2_
        d_17_useRepPenalty_: bool
        d_17_useRepPenalty_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
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
                        raise _dafny.Break("0")
                    elif ((d_1_steps_) + (d_3_closeReserve_)) >= (maxSteps):
                        d_21_rolledGenerated_: _dafny.Seq
                        d_22_rolledCurrent_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: _dafny.Seq
                        out16_, out17_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_21_rolledGenerated_ = out16_
                        d_22_rolledCurrent_ = out17_
                        generated = d_21_rolledGenerated_
                        currentConstrainedOut = d_22_rolledCurrent_
                        insideConstrainedOut = True
                        d_23_cg_: _dafny.Seq
                        d_24_ci_: bool
                        d_25_cc_: _dafny.Seq
                        d_26_closed_: bool
                        out18_: _dafny.Seq
                        out19_: bool
                        out20_: _dafny.Seq
                        out21_: bool
                        out18_, out19_, out20_, out21_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_23_cg_ = out18_
                        d_24_ci_ = out19_
                        d_25_cc_ = out20_
                        d_26_closed_ = out21_
                        d_1_steps_ = (d_1_steps_) + (1)
                        generated = d_23_cg_
                        insideConstrainedOut = d_24_ci_
                        currentConstrainedOut = d_25_cc_
                        raise _dafny.Break("0")
                    elif True:
                        d_27_constrainedPrompt_: _dafny.Seq
                        d_27_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_28_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if d_17_useRepPenalty_:
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_28_next_ = out22_
                        elif True:
                            out23_: _dafny.Seq
                            out23_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_28_next_ = out23_
                        d_17_useRepPenalty_ = not(d_17_useRepPenalty_)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_28_next_) == (eosToken):
                            d_29_rolledGenerated_: _dafny.Seq
                            d_30_rolledCurrent_: _dafny.Seq
                            out24_: _dafny.Seq
                            out25_: _dafny.Seq
                            out24_, out25_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_29_rolledGenerated_ = out24_
                            d_30_rolledCurrent_ = out25_
                            generated = d_29_rolledGenerated_
                            currentConstrainedOut = d_30_rolledCurrent_
                            insideConstrainedOut = True
                            if (d_1_steps_) < (maxSteps):
                                d_31_cg_: _dafny.Seq
                                d_32_ci_: bool
                                d_33_cc_: _dafny.Seq
                                d_34_closed_: bool
                                out26_: _dafny.Seq
                                out27_: bool
                                out28_: _dafny.Seq
                                out29_: bool
                                out26_, out27_, out28_, out29_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_31_cg_ = out26_
                                d_32_ci_ = out27_
                                d_33_cc_ = out28_
                                d_34_closed_ = out29_
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_31_cg_
                                insideConstrainedOut = d_32_ci_
                                currentConstrainedOut = d_33_cc_
                            raise _dafny.Break("0")
                        elif True:
                            d_35_appendedGenerated_: _dafny.Seq
                            d_36_appendedInside_: bool
                            d_37_appendedCurrent_: _dafny.Seq
                            out30_: _dafny.Seq
                            out31_: bool
                            out32_: _dafny.Seq
                            out30_, out31_, out32_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                            d_35_appendedGenerated_ = out30_
                            d_36_appendedInside_ = out31_
                            d_37_appendedCurrent_ = out32_
                            generated = d_35_appendedGenerated_
                            insideConstrainedOut = d_36_appendedInside_
                            currentConstrainedOut = d_37_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

