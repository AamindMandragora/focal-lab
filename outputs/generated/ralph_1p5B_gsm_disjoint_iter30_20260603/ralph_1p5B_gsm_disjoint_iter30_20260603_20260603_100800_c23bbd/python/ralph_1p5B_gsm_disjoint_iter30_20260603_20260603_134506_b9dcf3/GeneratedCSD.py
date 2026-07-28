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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final numeric answer inside << >> delimiters, e.g. <<42>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_remaining_: int
                        d_2_remaining_ = (maxSteps) - (d_1_steps_)
                        d_3_chunkBudget_: int
                        if (d_2_remaining_) < (25):
                            d_3_chunkBudget_ = d_2_remaining_
                        elif True:
                            d_3_chunkBudget_ = 25
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
                        generated = d_4_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOnOpenSpan_:
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
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        if (d_1_steps_) < (maxSteps):
                            d_11_g2_: _dafny.Seq
                            d_12_i2_: bool
                            d_13_c2_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_g2_ = out7_
                            d_12_i2_ = out8_
                            d_13_c2_ = out9_
                            generated = d_11_g2_
                            insideConstrainedOut = d_12_i2_
                            currentConstrainedOut = d_13_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_14_isDeadEnd_: bool
                        out10_: bool
                        out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_14_isDeadEnd_ = out10_
                        if d_14_isDeadEnd_:
                            d_15_gRolled_: _dafny.Seq
                            d_16_cRolled_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_15_gRolled_ = out11_
                            d_16_cRolled_ = out12_
                            generated = d_15_gRolled_
                            currentConstrainedOut = d_16_cRolled_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_17_g2_: _dafny.Seq
                                d_18_i2_: bool
                                d_19_c2_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_17_g2_ = out13_
                                d_18_i2_ = out14_
                                d_19_c2_ = out15_
                                generated = d_17_g2_
                                insideConstrainedOut = d_18_i2_
                                currentConstrainedOut = d_19_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif (d_1_steps_) < (maxSteps):
                                d_20_next_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_20_next_ = out16_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_20_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_21_g2_: _dafny.Seq
                                    d_22_i2_: bool
                                    d_23_c2_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                    d_21_g2_ = out17_
                                    d_22_i2_ = out18_
                                    d_23_c2_ = out19_
                                    generated = d_21_g2_
                                    insideConstrainedOut = d_22_i2_
                                    currentConstrainedOut = d_23_c2_
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_24_next_: _dafny.Seq
                            d_25_wasConstrained_: bool
                            out20_: _dafny.Seq
                            out21_: bool
                            out20_, out21_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_24_next_ = out20_
                            d_25_wasConstrained_ = out21_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_24_next_) == (eosToken):
                                d_26_gRolled_: _dafny.Seq
                                d_27_cRolled_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: _dafny.Seq
                                out22_, out23_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_26_gRolled_ = out22_
                                d_27_cRolled_ = out23_
                                generated = d_26_gRolled_
                                currentConstrainedOut = d_27_cRolled_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_28_g2_: _dafny.Seq
                                    d_29_i2_: bool
                                    d_30_c2_: _dafny.Seq
                                    out24_: _dafny.Seq
                                    out25_: bool
                                    out26_: _dafny.Seq
                                    out24_, out25_, out26_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_28_g2_ = out24_
                                    d_29_i2_ = out25_
                                    d_30_c2_ = out26_
                                    generated = d_28_g2_
                                    insideConstrainedOut = d_29_i2_
                                    currentConstrainedOut = d_30_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_31_g2_: _dafny.Seq
                                d_32_i2_: bool
                                d_33_c2_: _dafny.Seq
                                out27_: _dafny.Seq
                                out28_: bool
                                out29_: _dafny.Seq
                                out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_31_g2_ = out27_
                                d_32_i2_ = out28_
                                d_33_c2_ = out29_
                                generated = d_31_g2_
                                insideConstrainedOut = d_32_i2_
                                currentConstrainedOut = d_33_c2_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_34_gRolled_: _dafny.Seq
            d_35_cRolled_: _dafny.Seq
            out30_: _dafny.Seq
            out31_: _dafny.Seq
            out30_, out31_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
            d_34_gRolled_ = out30_
            d_35_cRolled_ = out31_
            generated = d_34_gRolled_
            currentConstrainedOut = d_35_cRolled_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                d_36_next_: _dafny.Seq
                out32_: _dafny.Seq
                out32_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                d_36_next_ = out32_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_36_next_) != (eosToken):
                    d_37_g2_: _dafny.Seq
                    d_38_i2_: bool
                    d_39_c2_: _dafny.Seq
                    out33_: _dafny.Seq
                    out34_: bool
                    out35_: _dafny.Seq
                    out33_, out34_, out35_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_36_next_)
                    d_37_g2_ = out33_
                    d_38_i2_ = out34_
                    d_39_c2_ = out35_
                    generated = d_37_g2_
                    insideConstrainedOut = d_38_i2_
                    currentConstrainedOut = d_39_c2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_40_g2_: _dafny.Seq
                d_41_i2_: bool
                d_42_c2_: _dafny.Seq
                out36_: _dafny.Seq
                out37_: bool
                out38_: _dafny.Seq
                out36_, out37_, out38_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_40_g2_ = out36_
                d_41_i2_ = out37_
                d_42_c2_ = out38_
                generated = d_40_g2_
                insideConstrainedOut = d_41_i2_
                currentConstrainedOut = d_42_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

