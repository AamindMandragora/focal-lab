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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final numeric answer inside << >> delimiters. Example: <<42>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 30
        d_3_spanTokensUsed_: int
        d_3_spanTokensUsed_ = 0
        d_4_spanMaxTokens_: int
        d_4_spanMaxTokens_ = 8
        d_5_hasSeenOpenSpan_: bool
        d_5_hasSeenOpenSpan_ = insideConstrained
        d_6_stuckCount_: int
        d_6_stuckCount_ = 0
        d_7_stuckLimit_: int
        d_7_stuckLimit_ = 3
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_remaining_: int
                        d_8_remaining_ = (maxSteps) - (d_1_steps_)
                        if ((d_8_remaining_) <= (3)) and (not(d_5_hasSeenOpenSpan_)):
                            d_9_g2_: _dafny.Seq
                            d_10_i2_: bool
                            d_11_c2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_9_g2_ = out0_
                            d_10_i2_ = out1_
                            d_11_c2_ = out2_
                            generated = d_9_g2_
                            insideConstrainedOut = d_10_i2_
                            currentConstrainedOut = d_11_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                            d_6_stuckCount_ = 0
                            d_5_hasSeenOpenSpan_ = True
                        elif (d_8_remaining_) <= (3):
                            d_12_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                            raise _dafny.Break("0")
                        elif True:
                            d_13_chunkBudget_: int
                            if (d_8_remaining_) < (d_2_freeChunkSize_):
                                d_13_chunkBudget_ = d_8_remaining_
                            elif True:
                                d_13_chunkBudget_ = d_2_freeChunkSize_
                            d_14_chunkGenerated_: _dafny.Seq
                            d_15_stoppedOnOpenSpan_: bool
                            d_16_stoppedOnEos_: bool
                            d_17_stepsUsed_: int
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: bool
                            out7_: int
                            out4_, out5_, out6_, out7_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_13_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_14_chunkGenerated_ = out4_
                            d_15_stoppedOnOpenSpan_ = out5_
                            d_16_stoppedOnEos_ = out6_
                            d_17_stepsUsed_ = out7_
                            generated = d_14_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_17_stepsUsed_)
                            if d_16_stoppedOnEos_:
                                if (not(d_5_hasSeenOpenSpan_)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                                    d_18_g2_: _dafny.Seq
                                    d_19_i2_: bool
                                    d_20_c2_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_18_g2_ = out8_
                                    d_19_i2_ = out9_
                                    d_20_c2_ = out10_
                                    generated = d_18_g2_
                                    insideConstrainedOut = d_19_i2_
                                    currentConstrainedOut = d_20_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                                    d_6_stuckCount_ = 0
                                    d_5_hasSeenOpenSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif d_15_stoppedOnOpenSpan_:
                                d_21_g2_: _dafny.Seq
                                d_22_i2_: bool
                                d_23_c2_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_21_g2_ = out11_
                                d_22_i2_ = out12_
                                d_23_c2_ = out13_
                                generated = d_21_g2_
                                insideConstrainedOut = d_22_i2_
                                currentConstrainedOut = d_23_c2_
                                d_3_spanTokensUsed_ = 0
                                d_6_stuckCount_ = 0
                                d_5_hasSeenOpenSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_24_g2_: _dafny.Seq
                        d_25_i2_: bool
                        d_26_c2_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_24_g2_ = out14_
                        d_25_i2_ = out15_
                        d_26_c2_ = out16_
                        generated = d_24_g2_
                        insideConstrainedOut = d_25_i2_
                        currentConstrainedOut = d_26_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokensUsed_ = 0
                        d_6_stuckCount_ = 0
                    elif True:
                        d_27_isDeadEnd_: bool
                        out17_: bool
                        out17_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_27_isDeadEnd_ = out17_
                        if ((d_27_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_))) or ((d_6_stuckCount_) >= (d_7_stuckLimit_)):
                            d_28_gRolled_: _dafny.Seq
                            d_29_cRolled_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: _dafny.Seq
                            out18_, out19_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_28_gRolled_ = out18_
                            d_29_cRolled_ = out19_
                            generated = d_28_gRolled_
                            currentConstrainedOut = d_29_cRolled_
                            d_6_stuckCount_ = (d_6_stuckCount_) + (1)
                            d_3_spanTokensUsed_ = 0
                            d_30_isComp_: bool
                            d_30_isComp_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if (d_30_isComp_) and ((d_1_steps_) < (maxSteps)):
                                d_31_g2_: _dafny.Seq
                                d_32_i2_: bool
                                d_33_c2_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_31_g2_ = out20_
                                d_32_i2_ = out21_
                                d_33_c2_ = out22_
                                generated = d_31_g2_
                                insideConstrainedOut = d_32_i2_
                                currentConstrainedOut = d_33_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_spanTokensUsed_ = 0
                                d_6_stuckCount_ = 0
                            elif (d_1_steps_) < (maxSteps):
                                d_34_constrainedPrompt_: _dafny.Seq
                                d_34_constrainedPrompt_ = (prompt) + (generated)
                                d_35_next_: _dafny.Seq
                                out23_: _dafny.Seq
                                out23_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_34_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_35_next_ = out23_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_35_next_) == (eosToken):
                                    if (d_6_stuckCount_) >= (d_7_stuckLimit_):
                                        raise _dafny.Break("0")
                                elif True:
                                    d_36_g2_: _dafny.Seq
                                    d_37_i2_: bool
                                    d_38_c2_: _dafny.Seq
                                    out24_: _dafny.Seq
                                    out25_: bool
                                    out26_: _dafny.Seq
                                    out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_35_next_)
                                    d_36_g2_ = out24_
                                    d_37_i2_ = out25_
                                    d_38_c2_ = out26_
                                    generated = d_36_g2_
                                    insideConstrainedOut = d_37_i2_
                                    currentConstrainedOut = d_38_c2_
                                    d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                    d_39_isComp2_: bool
                                    d_39_isComp2_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if (d_39_isComp2_) and ((d_1_steps_) < (maxSteps)):
                                        d_40_g3_: _dafny.Seq
                                        d_41_i3_: bool
                                        d_42_c3_: _dafny.Seq
                                        out27_: _dafny.Seq
                                        out28_: bool
                                        out29_: _dafny.Seq
                                        out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_40_g3_ = out27_
                                        d_41_i3_ = out28_
                                        d_42_c3_ = out29_
                                        generated = d_40_g3_
                                        insideConstrainedOut = d_41_i3_
                                        currentConstrainedOut = d_42_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                                        d_6_stuckCount_ = 0
                        elif True:
                            d_43_constrainedPrompt_: _dafny.Seq
                            d_43_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_44_next_: _dafny.Seq
                            out30_: _dafny.Seq
                            out30_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_43_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_44_next_ = out30_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_44_next_) == (eosToken):
                                d_45_gRolled_: _dafny.Seq
                                d_46_cRolled_: _dafny.Seq
                                out31_: _dafny.Seq
                                out32_: _dafny.Seq
                                out31_, out32_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_45_gRolled_ = out31_
                                d_46_cRolled_ = out32_
                                generated = d_45_gRolled_
                                currentConstrainedOut = d_46_cRolled_
                                d_6_stuckCount_ = (d_6_stuckCount_) + (1)
                                d_3_spanTokensUsed_ = 0
                                d_47_isComp_: bool
                                d_47_isComp_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if (d_47_isComp_) and ((d_1_steps_) < (maxSteps)):
                                    d_48_g2_: _dafny.Seq
                                    d_49_i2_: bool
                                    d_50_c2_: _dafny.Seq
                                    out33_: _dafny.Seq
                                    out34_: bool
                                    out35_: _dafny.Seq
                                    out33_, out34_, out35_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_48_g2_ = out33_
                                    d_49_i2_ = out34_
                                    d_50_c2_ = out35_
                                    generated = d_48_g2_
                                    insideConstrainedOut = d_49_i2_
                                    currentConstrainedOut = d_50_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                                    d_6_stuckCount_ = 0
                                elif (d_6_stuckCount_) >= (d_7_stuckLimit_):
                                    raise _dafny.Break("0")
                            elif True:
                                d_51_g2_: _dafny.Seq
                                d_52_i2_: bool
                                d_53_c2_: _dafny.Seq
                                out36_: _dafny.Seq
                                out37_: bool
                                out38_: _dafny.Seq
                                out36_, out37_, out38_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_44_next_)
                                d_51_g2_ = out36_
                                d_52_i2_ = out37_
                                d_53_c2_ = out38_
                                generated = d_51_g2_
                                insideConstrainedOut = d_52_i2_
                                currentConstrainedOut = d_53_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                d_6_stuckCount_ = 0
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_54_gRolled_: _dafny.Seq
            d_55_cRolled_: _dafny.Seq
            out39_: _dafny.Seq
            out40_: _dafny.Seq
            out39_, out40_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_54_gRolled_ = out39_
            d_55_cRolled_ = out40_
            generated = d_54_gRolled_
            currentConstrainedOut = d_55_cRolled_
            d_56_isComp_: bool
            d_56_isComp_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if (d_56_isComp_) and ((d_1_steps_) < (maxSteps)):
                d_57_g2_: _dafny.Seq
                d_58_i2_: bool
                d_59_c2_: _dafny.Seq
                out41_: _dafny.Seq
                out42_: bool
                out43_: _dafny.Seq
                out41_, out42_, out43_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_57_g2_ = out41_
                d_58_i2_ = out42_
                d_59_c2_ = out43_
                generated = d_57_g2_
                insideConstrainedOut = d_58_i2_
                currentConstrainedOut = d_59_c2_
                d_1_steps_ = (d_1_steps_) + (1)
            elif ((d_1_steps_) + (2)) <= (maxSteps):
                d_60_constrainedPrompt_: _dafny.Seq
                d_60_constrainedPrompt_ = (prompt) + (generated)
                d_61_next_: _dafny.Seq
                out44_: _dafny.Seq
                out44_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_60_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_61_next_ = out44_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_61_next_) != (eosToken):
                    d_62_g2_: _dafny.Seq
                    d_63_i2_: bool
                    d_64_c2_: _dafny.Seq
                    out45_: _dafny.Seq
                    out46_: bool
                    out47_: _dafny.Seq
                    out45_, out46_, out47_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_61_next_)
                    d_62_g2_ = out45_
                    d_63_i2_ = out46_
                    d_64_c2_ = out47_
                    generated = d_62_g2_
                    insideConstrainedOut = d_63_i2_
                    currentConstrainedOut = d_64_c2_
                    d_65_isComp2_: bool
                    d_65_isComp2_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if (d_65_isComp2_) and ((d_1_steps_) < (maxSteps)):
                        d_66_g3_: _dafny.Seq
                        d_67_i3_: bool
                        d_68_c3_: _dafny.Seq
                        out48_: _dafny.Seq
                        out49_: bool
                        out50_: _dafny.Seq
                        out48_, out49_, out50_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_66_g3_ = out48_
                        d_67_i3_ = out49_
                        d_68_c3_ = out50_
                        generated = d_66_g3_
                        insideConstrainedOut = d_67_i3_
                        currentConstrainedOut = d_68_c3_
                        d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

