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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Place ONLY the final arithmetic expression inside << >> at the end. Keep the span short. Example: <<n1 * p - discount>>. Do not repeat yourself.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 20
        d_3_spanTokensUsed_: int
        d_3_spanTokensUsed_ = 0
        d_4_spanMaxTokens_: int
        d_4_spanMaxTokens_ = 15
        d_5_hasSeenOpenSpan_: bool
        d_5_hasSeenOpenSpan_ = insideConstrained
        d_6_totalUnconstrainedTokens_: int
        d_6_totalUnconstrainedTokens_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_remaining_: int
                        d_7_remaining_ = (maxSteps) - (d_1_steps_)
                        if ((((d_7_remaining_) <= (80)) or ((d_6_totalUnconstrainedTokens_) >= (100))) and (not(d_5_hasSeenOpenSpan_))) and ((d_7_remaining_) > (3)):
                            d_8_g2_: _dafny.Seq
                            d_9_i2_: bool
                            d_10_c2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_g2_ = out0_
                            d_9_i2_ = out1_
                            d_10_c2_ = out2_
                            generated = d_8_g2_
                            insideConstrainedOut = d_9_i2_
                            currentConstrainedOut = d_10_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                            d_5_hasSeenOpenSpan_ = True
                        elif True:
                            d_11_chunkBudget_: int
                            if (d_7_remaining_) < (d_2_freeChunkSize_):
                                d_11_chunkBudget_ = d_7_remaining_
                            elif True:
                                d_11_chunkBudget_ = d_2_freeChunkSize_
                            if (d_11_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            d_12_chunkGenerated_: _dafny.Seq
                            d_13_stoppedOnOpenSpan_: bool
                            d_14_stoppedOnEos_: bool
                            d_15_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_11_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_12_chunkGenerated_ = out3_
                            d_13_stoppedOnOpenSpan_ = out4_
                            d_14_stoppedOnEos_ = out5_
                            d_15_stepsUsed_ = out6_
                            generated = d_12_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_15_stepsUsed_)
                            d_6_totalUnconstrainedTokens_ = (d_6_totalUnconstrainedTokens_) + (d_15_stepsUsed_)
                            if d_14_stoppedOnEos_:
                                if (not(d_5_hasSeenOpenSpan_)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                                    d_16_g2_: _dafny.Seq
                                    d_17_i2_: bool
                                    d_18_c2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_16_g2_ = out7_
                                    d_17_i2_ = out8_
                                    d_18_c2_ = out9_
                                    generated = d_16_g2_
                                    insideConstrainedOut = d_17_i2_
                                    currentConstrainedOut = d_18_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                                    d_5_hasSeenOpenSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif d_13_stoppedOnOpenSpan_:
                                d_19_g2_: _dafny.Seq
                                d_20_i2_: bool
                                d_21_c2_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_19_g2_ = out10_
                                d_20_i2_ = out11_
                                d_21_c2_ = out12_
                                generated = d_19_g2_
                                insideConstrainedOut = d_20_i2_
                                currentConstrainedOut = d_21_c2_
                                d_3_spanTokensUsed_ = 0
                                d_5_hasSeenOpenSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_22_g2_: _dafny.Seq
                        d_23_i2_: bool
                        d_24_c2_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_22_g2_ = out13_
                        d_23_i2_ = out14_
                        d_24_c2_ = out15_
                        generated = d_22_g2_
                        insideConstrainedOut = d_23_i2_
                        currentConstrainedOut = d_24_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokensUsed_ = 0
                    elif True:
                        d_25_isDeadEnd_: bool
                        out16_: bool
                        out16_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_25_isDeadEnd_ = out16_
                        if (d_25_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_)):
                            d_26_gRolled_: _dafny.Seq
                            d_27_cRolled_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: _dafny.Seq
                            out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_26_gRolled_ = out17_
                            d_27_cRolled_ = out18_
                            generated = d_26_gRolled_
                            currentConstrainedOut = d_27_cRolled_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_28_g2_: _dafny.Seq
                                d_29_i2_: bool
                                d_30_c2_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_28_g2_ = out19_
                                d_29_i2_ = out20_
                                d_30_c2_ = out21_
                                generated = d_28_g2_
                                insideConstrainedOut = d_29_i2_
                                currentConstrainedOut = d_30_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                if (d_1_steps_) < (maxSteps):
                                    d_31_constrainedPrompt_: _dafny.Seq
                                    d_31_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_32_next_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out22_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_31_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_32_next_ = out22_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_32_next_) == (eosToken):
                                        d_33_gR2_: _dafny.Seq
                                        d_34_cR2_: _dafny.Seq
                                        out23_: _dafny.Seq
                                        out24_: _dafny.Seq
                                        out23_, out24_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                        d_33_gR2_ = out23_
                                        d_34_cR2_ = out24_
                                        generated = d_33_gR2_
                                        currentConstrainedOut = d_34_cR2_
                                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                            d_35_g2_: _dafny.Seq
                                            d_36_i2_: bool
                                            d_37_c2_: _dafny.Seq
                                            out25_: _dafny.Seq
                                            out26_: bool
                                            out27_: _dafny.Seq
                                            out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_35_g2_ = out25_
                                            d_36_i2_ = out26_
                                            d_37_c2_ = out27_
                                            generated = d_35_g2_
                                            insideConstrainedOut = d_36_i2_
                                            currentConstrainedOut = d_37_c2_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        d_38_g2_: _dafny.Seq
                                        d_39_i2_: bool
                                        d_40_c2_: _dafny.Seq
                                        out28_: _dafny.Seq
                                        out29_: bool
                                        out30_: _dafny.Seq
                                        out28_, out29_, out30_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next_)
                                        d_38_g2_ = out28_
                                        d_39_i2_ = out29_
                                        d_40_c2_ = out30_
                                        generated = d_38_g2_
                                        insideConstrainedOut = d_39_i2_
                                        currentConstrainedOut = d_40_c2_
                                        d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                            d_41_g3_: _dafny.Seq
                                            d_42_i3_: bool
                                            d_43_c3_: _dafny.Seq
                                            out31_: _dafny.Seq
                                            out32_: bool
                                            out33_: _dafny.Seq
                                            out31_, out32_, out33_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_41_g3_ = out31_
                                            d_42_i3_ = out32_
                                            d_43_c3_ = out33_
                                            generated = d_41_g3_
                                            insideConstrainedOut = d_42_i3_
                                            currentConstrainedOut = d_43_c3_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            d_3_spanTokensUsed_ = 0
                        elif True:
                            d_44_constrainedPrompt_: _dafny.Seq
                            d_44_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_45_next_: _dafny.Seq
                            out34_: _dafny.Seq
                            out34_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_44_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_45_next_ = out34_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_45_next_) == (eosToken):
                                d_46_gRolled_: _dafny.Seq
                                d_47_cRolled_: _dafny.Seq
                                out35_: _dafny.Seq
                                out36_: _dafny.Seq
                                out35_, out36_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_46_gRolled_ = out35_
                                d_47_cRolled_ = out36_
                                generated = d_46_gRolled_
                                currentConstrainedOut = d_47_cRolled_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_48_g2_: _dafny.Seq
                                    d_49_i2_: bool
                                    d_50_c2_: _dafny.Seq
                                    out37_: _dafny.Seq
                                    out38_: bool
                                    out39_: _dafny.Seq
                                    out37_, out38_, out39_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_48_g2_ = out37_
                                    d_49_i2_ = out38_
                                    d_50_c2_ = out39_
                                    generated = d_48_g2_
                                    insideConstrainedOut = d_49_i2_
                                    currentConstrainedOut = d_50_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_51_g2_: _dafny.Seq
                                d_52_i2_: bool
                                d_53_c2_: _dafny.Seq
                                out40_: _dafny.Seq
                                out41_: bool
                                out42_: _dafny.Seq
                                out40_, out41_, out42_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_45_next_)
                                d_51_g2_ = out40_
                                d_52_i2_ = out41_
                                d_53_c2_ = out42_
                                generated = d_51_g2_
                                insideConstrainedOut = d_52_i2_
                                currentConstrainedOut = d_53_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_54_gRolled_: _dafny.Seq
            d_55_cRolled_: _dafny.Seq
            out43_: _dafny.Seq
            out44_: _dafny.Seq
            out43_, out44_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_54_gRolled_ = out43_
            d_55_cRolled_ = out44_
            generated = d_54_gRolled_
            currentConstrainedOut = d_55_cRolled_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_56_constrainedPrompt_: _dafny.Seq
                d_56_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_57_next_: _dafny.Seq
                out45_: _dafny.Seq
                out45_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_56_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_57_next_ = out45_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_57_next_) != (eosToken):
                    d_58_g2_: _dafny.Seq
                    d_59_i2_: bool
                    d_60_c2_: _dafny.Seq
                    out46_: _dafny.Seq
                    out47_: bool
                    out48_: _dafny.Seq
                    out46_, out47_, out48_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_57_next_)
                    d_58_g2_ = out46_
                    d_59_i2_ = out47_
                    d_60_c2_ = out48_
                    generated = d_58_g2_
                    insideConstrainedOut = d_59_i2_
                    currentConstrainedOut = d_60_c2_
                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                        d_61_gR2_: _dafny.Seq
                        d_62_cR2_: _dafny.Seq
                        out49_: _dafny.Seq
                        out50_: _dafny.Seq
                        out49_, out50_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_61_gR2_ = out49_
                        d_62_cR2_ = out50_
                        generated = d_61_gR2_
                        currentConstrainedOut = d_62_cR2_
                elif True:
                    d_63_gR2_: _dafny.Seq
                    d_64_cR2_: _dafny.Seq
                    out51_: _dafny.Seq
                    out52_: _dafny.Seq
                    out51_, out52_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                    d_63_gR2_ = out51_
                    d_64_cR2_ = out52_
                    generated = d_63_gR2_
                    currentConstrainedOut = d_64_cR2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_65_g2_: _dafny.Seq
                d_66_i2_: bool
                d_67_c2_: _dafny.Seq
                out53_: _dafny.Seq
                out54_: bool
                out55_: _dafny.Seq
                out53_, out54_, out55_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_65_g2_ = out53_
                d_66_i2_ = out54_
                d_67_c2_ = out55_
                generated = d_65_g2_
                insideConstrainedOut = d_66_i2_
                currentConstrainedOut = d_67_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

