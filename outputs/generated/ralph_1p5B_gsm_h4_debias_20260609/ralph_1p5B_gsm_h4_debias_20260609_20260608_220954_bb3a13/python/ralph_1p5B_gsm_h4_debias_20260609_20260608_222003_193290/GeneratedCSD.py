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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. At the end, write ONLY the final numeric expression inside << >> delimiters, like <<42>> or <<n1 + n2 * 3>>. Keep the final answer span short and exact.")))
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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remaining_: int
                        d_6_remaining_ = (maxSteps) - (d_1_steps_)
                        if (((d_6_remaining_) <= (50)) and (not(d_5_hasSeenOpenSpan_))) and ((d_6_remaining_) > (3)):
                            d_7_g2_: _dafny.Seq
                            d_8_i2_: bool
                            d_9_c2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_g2_ = out0_
                            d_8_i2_ = out1_
                            d_9_c2_ = out2_
                            generated = d_7_g2_
                            insideConstrainedOut = d_8_i2_
                            currentConstrainedOut = d_9_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                            d_5_hasSeenOpenSpan_ = True
                        elif True:
                            d_10_chunkBudget_: int
                            if (d_6_remaining_) < (d_2_freeChunkSize_):
                                d_10_chunkBudget_ = d_6_remaining_
                            elif True:
                                d_10_chunkBudget_ = d_2_freeChunkSize_
                            if (d_10_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            d_11_chunkGenerated_: _dafny.Seq
                            d_12_stoppedOnOpenSpan_: bool
                            d_13_stoppedOnEos_: bool
                            d_14_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_11_chunkGenerated_ = out3_
                            d_12_stoppedOnOpenSpan_ = out4_
                            d_13_stoppedOnEos_ = out5_
                            d_14_stepsUsed_ = out6_
                            generated = d_11_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                            if d_13_stoppedOnEos_:
                                if (not(d_5_hasSeenOpenSpan_)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                                    d_15_g2_: _dafny.Seq
                                    d_16_i2_: bool
                                    d_17_c2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_15_g2_ = out7_
                                    d_16_i2_ = out8_
                                    d_17_c2_ = out9_
                                    generated = d_15_g2_
                                    insideConstrainedOut = d_16_i2_
                                    currentConstrainedOut = d_17_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                                    d_5_hasSeenOpenSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif d_12_stoppedOnOpenSpan_:
                                d_18_g2_: _dafny.Seq
                                d_19_i2_: bool
                                d_20_c2_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_18_g2_ = out10_
                                d_19_i2_ = out11_
                                d_20_c2_ = out12_
                                generated = d_18_g2_
                                insideConstrainedOut = d_19_i2_
                                currentConstrainedOut = d_20_c2_
                                d_3_spanTokensUsed_ = 0
                                d_5_hasSeenOpenSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_21_g2_: _dafny.Seq
                        d_22_i2_: bool
                        d_23_c2_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_21_g2_ = out13_
                        d_22_i2_ = out14_
                        d_23_c2_ = out15_
                        generated = d_21_g2_
                        insideConstrainedOut = d_22_i2_
                        currentConstrainedOut = d_23_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokensUsed_ = 0
                    elif True:
                        d_24_isDeadEnd_: bool
                        out16_: bool
                        out16_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_24_isDeadEnd_ = out16_
                        if (d_24_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_)):
                            d_25_gRolled_: _dafny.Seq
                            d_26_cRolled_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: _dafny.Seq
                            out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_25_gRolled_ = out17_
                            d_26_cRolled_ = out18_
                            generated = d_25_gRolled_
                            currentConstrainedOut = d_26_cRolled_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_27_g2_: _dafny.Seq
                                d_28_i2_: bool
                                d_29_c2_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_27_g2_ = out19_
                                d_28_i2_ = out20_
                                d_29_c2_ = out21_
                                generated = d_27_g2_
                                insideConstrainedOut = d_28_i2_
                                currentConstrainedOut = d_29_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                if (d_1_steps_) < (maxSteps):
                                    d_30_constrainedPrompt_: _dafny.Seq
                                    d_30_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_31_next_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out22_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_30_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                    d_31_next_ = out22_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_31_next_) != (eosToken):
                                        d_32_g2_: _dafny.Seq
                                        d_33_i2_: bool
                                        d_34_c2_: _dafny.Seq
                                        out23_: _dafny.Seq
                                        out24_: bool
                                        out25_: _dafny.Seq
                                        out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_next_)
                                        d_32_g2_ = out23_
                                        d_33_i2_ = out24_
                                        d_34_c2_ = out25_
                                        generated = d_32_g2_
                                        insideConstrainedOut = d_33_i2_
                                        currentConstrainedOut = d_34_c2_
                                        d_3_spanTokensUsed_ = 1
                                    elif True:
                                        if (d_1_steps_) < (maxSteps):
                                            d_35_constrainedPrompt2_: _dafny.Seq
                                            d_35_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                            d_36_next2_: _dafny.Seq
                                            out26_: _dafny.Seq
                                            out26_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_35_constrainedPrompt2_, currentConstrainedOut, eosToken)
                                            d_36_next2_ = out26_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            if (d_36_next2_) != (eosToken):
                                                d_37_g3_: _dafny.Seq
                                                d_38_i3_: bool
                                                d_39_c3_: _dafny.Seq
                                                out27_: _dafny.Seq
                                                out28_: bool
                                                out29_: _dafny.Seq
                                                out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_36_next2_)
                                                d_37_g3_ = out27_
                                                d_38_i3_ = out28_
                                                d_39_c3_ = out29_
                                                generated = d_37_g3_
                                                insideConstrainedOut = d_38_i3_
                                                currentConstrainedOut = d_39_c3_
                                                d_3_spanTokensUsed_ = 1
                        elif True:
                            d_40_constrainedPrompt_: _dafny.Seq
                            d_40_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_41_next_: _dafny.Seq
                            out30_: _dafny.Seq
                            out30_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_40_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                            d_41_next_ = out30_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_41_next_) == (eosToken):
                                d_42_gRolled_: _dafny.Seq
                                d_43_cRolled_: _dafny.Seq
                                out31_: _dafny.Seq
                                out32_: _dafny.Seq
                                out31_, out32_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_42_gRolled_ = out31_
                                d_43_cRolled_ = out32_
                                generated = d_42_gRolled_
                                currentConstrainedOut = d_43_cRolled_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_44_g2_: _dafny.Seq
                                    d_45_i2_: bool
                                    d_46_c2_: _dafny.Seq
                                    out33_: _dafny.Seq
                                    out34_: bool
                                    out35_: _dafny.Seq
                                    out33_, out34_, out35_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_44_g2_ = out33_
                                    d_45_i2_ = out34_
                                    d_46_c2_ = out35_
                                    generated = d_44_g2_
                                    insideConstrainedOut = d_45_i2_
                                    currentConstrainedOut = d_46_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    if (d_1_steps_) < (maxSteps):
                                        d_47_constrainedPrompt2_: _dafny.Seq
                                        d_47_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                        d_48_next2_: _dafny.Seq
                                        out36_: _dafny.Seq
                                        out36_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_47_constrainedPrompt2_, currentConstrainedOut, eosToken)
                                        d_48_next2_ = out36_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if (d_48_next2_) != (eosToken):
                                            d_49_g3_: _dafny.Seq
                                            d_50_i3_: bool
                                            d_51_c3_: _dafny.Seq
                                            out37_: _dafny.Seq
                                            out38_: bool
                                            out39_: _dafny.Seq
                                            out37_, out38_, out39_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_48_next2_)
                                            d_49_g3_ = out37_
                                            d_50_i3_ = out38_
                                            d_51_c3_ = out39_
                                            generated = d_49_g3_
                                            insideConstrainedOut = d_50_i3_
                                            currentConstrainedOut = d_51_c3_
                                            d_3_spanTokensUsed_ = 1
                            elif True:
                                d_52_g2_: _dafny.Seq
                                d_53_i2_: bool
                                d_54_c2_: _dafny.Seq
                                out40_: _dafny.Seq
                                out41_: bool
                                out42_: _dafny.Seq
                                out40_, out41_, out42_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_41_next_)
                                d_52_g2_ = out40_
                                d_53_i2_ = out41_
                                d_54_c2_ = out42_
                                generated = d_52_g2_
                                insideConstrainedOut = d_53_i2_
                                currentConstrainedOut = d_54_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_55_gRolled_: _dafny.Seq
            d_56_cRolled_: _dafny.Seq
            out43_: _dafny.Seq
            out44_: _dafny.Seq
            out43_, out44_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_55_gRolled_ = out43_
            d_56_cRolled_ = out44_
            generated = d_55_gRolled_
            currentConstrainedOut = d_56_cRolled_
            if not((parser).IsCompletePrefix(currentConstrainedOut)):
                d_57_gR2_: _dafny.Seq
                d_58_cR2_: _dafny.Seq
                out45_: _dafny.Seq
                out46_: _dafny.Seq
                out45_, out46_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_57_gR2_ = out45_
                d_58_cR2_ = out46_
                generated = d_57_gR2_
                currentConstrainedOut = d_58_cR2_
                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                    d_59_gR3_: _dafny.Seq
                    d_60_cR3_: _dafny.Seq
                    out47_: _dafny.Seq
                    out48_: _dafny.Seq
                    out47_, out48_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                    d_59_gR3_ = out47_
                    d_60_cR3_ = out48_
                    generated = d_59_gR3_
                    currentConstrainedOut = d_60_cR3_
                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                    d_61_gR4_: _dafny.Seq
                    d_62_cR4_: _dafny.Seq
                    out49_: _dafny.Seq
                    out50_: _dafny.Seq
                    out49_, out50_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                    d_61_gR4_ = out49_
                    d_62_cR4_ = out50_
                    generated = d_61_gR4_
                    currentConstrainedOut = d_62_cR4_
                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                    d_63_constrainedPrompt_: _dafny.Seq
                    d_63_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_64_next_: _dafny.Seq
                    out51_: _dafny.Seq
                    out51_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_63_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_64_next_ = out51_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_64_next_) != (eosToken):
                        d_65_g2_: _dafny.Seq
                        d_66_i2_: bool
                        d_67_c2_: _dafny.Seq
                        out52_: _dafny.Seq
                        out53_: bool
                        out54_: _dafny.Seq
                        out52_, out53_, out54_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_64_next_)
                        d_65_g2_ = out52_
                        d_66_i2_ = out53_
                        d_67_c2_ = out54_
                        generated = d_65_g2_
                        insideConstrainedOut = d_66_i2_
                        currentConstrainedOut = d_67_c2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_68_g2_: _dafny.Seq
                d_69_i2_: bool
                d_70_c2_: _dafny.Seq
                out55_: _dafny.Seq
                out56_: bool
                out57_: _dafny.Seq
                out55_, out56_, out57_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_68_g2_ = out55_
                d_69_i2_ = out56_
                d_70_c2_ = out57_
                generated = d_68_g2_
                insideConstrainedOut = d_69_i2_
                currentConstrainedOut = d_70_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

