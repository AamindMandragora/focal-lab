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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write each calculation and the final answer inside << >> delimiters. Example: <<n1 + n2>>. Final answer: <<42>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 25
        d_3_spanTokensUsed_: int
        d_3_spanTokensUsed_ = 0
        d_4_spanMaxTokens_: int
        d_4_spanMaxTokens_ = 12
        d_5_hasSeenOpenSpan_: bool
        d_5_hasSeenOpenSpan_ = insideConstrained
        d_6_closeReserve_: int
        d_6_closeReserve_ = 25
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_7_remaining_: int
                    d_7_remaining_ = (maxSteps) - (d_1_steps_)
                    if (insideConstrainedOut) and ((d_7_remaining_) <= (d_6_closeReserve_)):
                        if not((parser).IsCompletePrefix(currentConstrainedOut)):
                            d_8_constrainedPromptEmerg_: _dafny.Seq
                            d_8_constrainedPromptEmerg_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_9_nextEmerg_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_8_constrainedPromptEmerg_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_9_nextEmerg_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_nextEmerg_) == (eosToken):
                                d_10_gRE_: _dafny.Seq
                                d_11_cRE_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: _dafny.Seq
                                out1_, out2_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_10_gRE_ = out1_
                                d_11_cRE_ = out2_
                                generated = d_10_gRE_
                                currentConstrainedOut = d_11_cRE_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_12_gRE2_: _dafny.Seq
                                    d_13_cRE2_: _dafny.Seq
                                    out3_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out3_, out4_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_12_gRE2_ = out3_
                                    d_13_cRE2_ = out4_
                                    generated = d_12_gRE2_
                                    currentConstrainedOut = d_13_cRE2_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_14_gRE3_: _dafny.Seq
                                    d_15_cRE3_: _dafny.Seq
                                    out5_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out5_, out6_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_14_gRE3_ = out5_
                                    d_15_cRE3_ = out6_
                                    generated = d_14_gRE3_
                                    currentConstrainedOut = d_15_cRE3_
                            elif True:
                                d_16_gE_: _dafny.Seq
                                d_17_iE_: bool
                                d_18_cE_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_nextEmerg_)
                                d_16_gE_ = out7_
                                d_17_iE_ = out8_
                                d_18_cE_ = out9_
                                generated = d_16_gE_
                                insideConstrainedOut = d_17_iE_
                                currentConstrainedOut = d_18_cE_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_19_gC_: _dafny.Seq
                            d_20_iC_: bool
                            d_21_cC_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_gC_ = out10_
                            d_20_iC_ = out11_
                            d_21_cC_ = out12_
                            generated = d_19_gC_
                            insideConstrainedOut = d_20_iC_
                            currentConstrainedOut = d_21_cC_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                    elif not(insideConstrainedOut):
                        if (((d_7_remaining_) <= (65)) and (not(d_5_hasSeenOpenSpan_))) and ((d_7_remaining_) > (2)):
                            d_22_g2_: _dafny.Seq
                            d_23_i2_: bool
                            d_24_c2_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_22_g2_ = out13_
                            d_23_i2_ = out14_
                            d_24_c2_ = out15_
                            generated = d_22_g2_
                            insideConstrainedOut = d_23_i2_
                            currentConstrainedOut = d_24_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                            d_5_hasSeenOpenSpan_ = True
                        elif True:
                            d_25_chunkBudget_: int
                            if (d_7_remaining_) < (d_2_freeChunkSize_):
                                d_25_chunkBudget_ = d_7_remaining_
                            elif True:
                                d_25_chunkBudget_ = d_2_freeChunkSize_
                            if (d_25_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            d_26_chunkGenerated_: _dafny.Seq
                            d_27_stoppedOnOpenSpan_: bool
                            d_28_stoppedOnEos_: bool
                            d_29_stepsUsed_: int
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: bool
                            out19_: int
                            out16_, out17_, out18_, out19_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_25_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_26_chunkGenerated_ = out16_
                            d_27_stoppedOnOpenSpan_ = out17_
                            d_28_stoppedOnEos_ = out18_
                            d_29_stepsUsed_ = out19_
                            generated = d_26_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_29_stepsUsed_)
                            if d_28_stoppedOnEos_:
                                if (not(d_5_hasSeenOpenSpan_)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                                    d_30_g2_: _dafny.Seq
                                    d_31_i2_: bool
                                    d_32_c2_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_30_g2_ = out20_
                                    d_31_i2_ = out21_
                                    d_32_c2_ = out22_
                                    generated = d_30_g2_
                                    insideConstrainedOut = d_31_i2_
                                    currentConstrainedOut = d_32_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                                    d_5_hasSeenOpenSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif d_27_stoppedOnOpenSpan_:
                                d_33_g2_: _dafny.Seq
                                d_34_i2_: bool
                                d_35_c2_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: bool
                                out25_: _dafny.Seq
                                out23_, out24_, out25_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_33_g2_ = out23_
                                d_34_i2_ = out24_
                                d_35_c2_ = out25_
                                generated = d_33_g2_
                                insideConstrainedOut = d_34_i2_
                                currentConstrainedOut = d_35_c2_
                                d_3_spanTokensUsed_ = 0
                                d_5_hasSeenOpenSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_36_g2_: _dafny.Seq
                        d_37_i2_: bool
                        d_38_c2_: _dafny.Seq
                        out26_: _dafny.Seq
                        out27_: bool
                        out28_: _dafny.Seq
                        out26_, out27_, out28_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_36_g2_ = out26_
                        d_37_i2_ = out27_
                        d_38_c2_ = out28_
                        generated = d_36_g2_
                        insideConstrainedOut = d_37_i2_
                        currentConstrainedOut = d_38_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokensUsed_ = 0
                    elif True:
                        d_39_isDeadEnd_: bool
                        out29_: bool
                        out29_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_39_isDeadEnd_ = out29_
                        if (d_39_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_)):
                            d_40_gRolled_: _dafny.Seq
                            d_41_cRolled_: _dafny.Seq
                            out30_: _dafny.Seq
                            out31_: _dafny.Seq
                            out30_, out31_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_40_gRolled_ = out30_
                            d_41_cRolled_ = out31_
                            generated = d_40_gRolled_
                            currentConstrainedOut = d_41_cRolled_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_42_g3_: _dafny.Seq
                                d_43_c3_: _dafny.Seq
                                out32_: _dafny.Seq
                                out33_: _dafny.Seq
                                out32_, out33_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_42_g3_ = out32_
                                d_43_c3_ = out33_
                                generated = d_42_g3_
                                currentConstrainedOut = d_43_c3_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_44_g4_: _dafny.Seq
                                d_45_c4_: _dafny.Seq
                                out34_: _dafny.Seq
                                out35_: _dafny.Seq
                                out34_, out35_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_44_g4_ = out34_
                                d_45_c4_ = out35_
                                generated = d_44_g4_
                                currentConstrainedOut = d_45_c4_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_46_g5_: _dafny.Seq
                                d_47_c5_: _dafny.Seq
                                out36_: _dafny.Seq
                                out37_: _dafny.Seq
                                out36_, out37_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_46_g5_ = out36_
                                d_47_c5_ = out37_
                                generated = d_46_g5_
                                currentConstrainedOut = d_47_c5_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_48_g6_: _dafny.Seq
                                d_49_c6_: _dafny.Seq
                                out38_: _dafny.Seq
                                out39_: _dafny.Seq
                                out38_, out39_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_48_g6_ = out38_
                                d_49_c6_ = out39_
                                generated = d_48_g6_
                                currentConstrainedOut = d_49_c6_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_50_g7_: _dafny.Seq
                                d_51_c7_: _dafny.Seq
                                out40_: _dafny.Seq
                                out41_: _dafny.Seq
                                out40_, out41_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_50_g7_ = out40_
                                d_51_c7_ = out41_
                                generated = d_50_g7_
                                currentConstrainedOut = d_51_c7_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_52_g2_: _dafny.Seq
                                d_53_i2_: bool
                                d_54_c2_: _dafny.Seq
                                out42_: _dafny.Seq
                                out43_: bool
                                out44_: _dafny.Seq
                                out42_, out43_, out44_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_52_g2_ = out42_
                                d_53_i2_ = out43_
                                d_54_c2_ = out44_
                                generated = d_52_g2_
                                insideConstrainedOut = d_53_i2_
                                currentConstrainedOut = d_54_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif (d_1_steps_) < (maxSteps):
                                d_55_constrainedPrompt_: _dafny.Seq
                                d_55_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_56_next_: _dafny.Seq
                                out45_: _dafny.Seq
                                out45_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_55_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_56_next_ = out45_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_56_next_) == (eosToken):
                                    d_57_gR2_: _dafny.Seq
                                    d_58_cR2_: _dafny.Seq
                                    out46_: _dafny.Seq
                                    out47_: _dafny.Seq
                                    out46_, out47_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_57_gR2_ = out46_
                                    d_58_cR2_ = out47_
                                    generated = d_57_gR2_
                                    currentConstrainedOut = d_58_cR2_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_59_gR3_: _dafny.Seq
                                        d_60_cR3_: _dafny.Seq
                                        out48_: _dafny.Seq
                                        out49_: _dafny.Seq
                                        out48_, out49_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_59_gR3_ = out48_
                                        d_60_cR3_ = out49_
                                        generated = d_59_gR3_
                                        currentConstrainedOut = d_60_cR3_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_61_gR4_: _dafny.Seq
                                        d_62_cR4_: _dafny.Seq
                                        out50_: _dafny.Seq
                                        out51_: _dafny.Seq
                                        out50_, out51_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_61_gR4_ = out50_
                                        d_62_cR4_ = out51_
                                        generated = d_61_gR4_
                                        currentConstrainedOut = d_62_cR4_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_63_g2_: _dafny.Seq
                                        d_64_i2_: bool
                                        d_65_c2_: _dafny.Seq
                                        out52_: _dafny.Seq
                                        out53_: bool
                                        out54_: _dafny.Seq
                                        out52_, out53_, out54_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_63_g2_ = out52_
                                        d_64_i2_ = out53_
                                        d_65_c2_ = out54_
                                        generated = d_63_g2_
                                        insideConstrainedOut = d_64_i2_
                                        currentConstrainedOut = d_65_c2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_66_g2_: _dafny.Seq
                                    d_67_i2_: bool
                                    d_68_c2_: _dafny.Seq
                                    out55_: _dafny.Seq
                                    out56_: bool
                                    out57_: _dafny.Seq
                                    out55_, out56_, out57_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_56_next_)
                                    d_66_g2_ = out55_
                                    d_67_i2_ = out56_
                                    d_68_c2_ = out57_
                                    generated = d_66_g2_
                                    insideConstrainedOut = d_67_i2_
                                    currentConstrainedOut = d_68_c2_
                                    d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_69_g3_: _dafny.Seq
                                        d_70_i3_: bool
                                        d_71_c3_: _dafny.Seq
                                        out58_: _dafny.Seq
                                        out59_: bool
                                        out60_: _dafny.Seq
                                        out58_, out59_, out60_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_69_g3_ = out58_
                                        d_70_i3_ = out59_
                                        d_71_c3_ = out60_
                                        generated = d_69_g3_
                                        insideConstrainedOut = d_70_i3_
                                        currentConstrainedOut = d_71_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                        elif True:
                            d_72_constrainedPrompt_: _dafny.Seq
                            d_72_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_73_next_: _dafny.Seq
                            d_74_wasConstrained_: bool
                            out61_: _dafny.Seq
                            out62_: bool
                            out61_, out62_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_72_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_73_next_ = out61_
                            d_74_wasConstrained_ = out62_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_73_next_) == (eosToken):
                                d_75_gRolled_: _dafny.Seq
                                d_76_cRolled_: _dafny.Seq
                                out63_: _dafny.Seq
                                out64_: _dafny.Seq
                                out63_, out64_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_75_gRolled_ = out63_
                                d_76_cRolled_ = out64_
                                generated = d_75_gRolled_
                                currentConstrainedOut = d_76_cRolled_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_77_gR2_: _dafny.Seq
                                    d_78_cR2_: _dafny.Seq
                                    out65_: _dafny.Seq
                                    out66_: _dafny.Seq
                                    out65_, out66_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_77_gR2_ = out65_
                                    d_78_cR2_ = out66_
                                    generated = d_77_gR2_
                                    currentConstrainedOut = d_78_cR2_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_79_gR3_: _dafny.Seq
                                    d_80_cR3_: _dafny.Seq
                                    out67_: _dafny.Seq
                                    out68_: _dafny.Seq
                                    out67_, out68_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_79_gR3_ = out67_
                                    d_80_cR3_ = out68_
                                    generated = d_79_gR3_
                                    currentConstrainedOut = d_80_cR3_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_81_gR4_: _dafny.Seq
                                    d_82_cR4_: _dafny.Seq
                                    out69_: _dafny.Seq
                                    out70_: _dafny.Seq
                                    out69_, out70_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_81_gR4_ = out69_
                                    d_82_cR4_ = out70_
                                    generated = d_81_gR4_
                                    currentConstrainedOut = d_82_cR4_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_83_gR5_: _dafny.Seq
                                    d_84_cR5_: _dafny.Seq
                                    out71_: _dafny.Seq
                                    out72_: _dafny.Seq
                                    out71_, out72_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_83_gR5_ = out71_
                                    d_84_cR5_ = out72_
                                    generated = d_83_gR5_
                                    currentConstrainedOut = d_84_cR5_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_85_g2_: _dafny.Seq
                                    d_86_i2_: bool
                                    d_87_c2_: _dafny.Seq
                                    out73_: _dafny.Seq
                                    out74_: bool
                                    out75_: _dafny.Seq
                                    out73_, out74_, out75_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_85_g2_ = out73_
                                    d_86_i2_ = out74_
                                    d_87_c2_ = out75_
                                    generated = d_85_g2_
                                    insideConstrainedOut = d_86_i2_
                                    currentConstrainedOut = d_87_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_88_g2_: _dafny.Seq
                                d_89_i2_: bool
                                d_90_c2_: _dafny.Seq
                                out76_: _dafny.Seq
                                out77_: bool
                                out78_: _dafny.Seq
                                out76_, out77_, out78_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_73_next_)
                                d_88_g2_ = out76_
                                d_89_i2_ = out77_
                                d_90_c2_ = out78_
                                generated = d_88_g2_
                                insideConstrainedOut = d_89_i2_
                                currentConstrainedOut = d_90_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_91_gRolled_: _dafny.Seq
            d_92_cRolled_: _dafny.Seq
            out79_: _dafny.Seq
            out80_: _dafny.Seq
            out79_, out80_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
            d_91_gRolled_ = out79_
            d_92_cRolled_ = out80_
            generated = d_91_gRolled_
            currentConstrainedOut = d_92_cRolled_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_93_g2_: _dafny.Seq
                d_94_c2_: _dafny.Seq
                out81_: _dafny.Seq
                out82_: _dafny.Seq
                out81_, out82_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_93_g2_ = out81_
                d_94_c2_ = out82_
                generated = d_93_g2_
                currentConstrainedOut = d_94_c2_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_95_g3_: _dafny.Seq
                d_96_c3_: _dafny.Seq
                out83_: _dafny.Seq
                out84_: _dafny.Seq
                out83_, out84_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_95_g3_ = out83_
                d_96_c3_ = out84_
                generated = d_95_g3_
                currentConstrainedOut = d_96_c3_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_97_g4_: _dafny.Seq
                d_98_c4_: _dafny.Seq
                out85_: _dafny.Seq
                out86_: _dafny.Seq
                out85_, out86_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_97_g4_ = out85_
                d_98_c4_ = out86_
                generated = d_97_g4_
                currentConstrainedOut = d_98_c4_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_99_g5_: _dafny.Seq
                d_100_c5_: _dafny.Seq
                out87_: _dafny.Seq
                out88_: _dafny.Seq
                out87_, out88_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_99_g5_ = out87_
                d_100_c5_ = out88_
                generated = d_99_g5_
                currentConstrainedOut = d_100_c5_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_101_constrainedPromptPost_: _dafny.Seq
                d_101_constrainedPromptPost_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_102_nextPost_: _dafny.Seq
                out89_: _dafny.Seq
                out89_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_101_constrainedPromptPost_, currentConstrainedOut, eosToken)
                d_102_nextPost_ = out89_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_102_nextPost_) != (eosToken):
                    d_103_gp_: _dafny.Seq
                    d_104_ip_: bool
                    d_105_cp_: _dafny.Seq
                    out90_: _dafny.Seq
                    out91_: bool
                    out92_: _dafny.Seq
                    out90_, out91_, out92_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_102_nextPost_)
                    d_103_gp_ = out90_
                    d_104_ip_ = out91_
                    d_105_cp_ = out92_
                    generated = d_103_gp_
                    insideConstrainedOut = d_104_ip_
                    currentConstrainedOut = d_105_cp_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_106_g2_: _dafny.Seq
                d_107_i2_: bool
                d_108_c2_: _dafny.Seq
                out93_: _dafny.Seq
                out94_: bool
                out95_: _dafny.Seq
                out93_, out94_, out95_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_106_g2_ = out93_
                d_107_i2_ = out94_
                d_108_c2_ = out95_
                generated = d_106_g2_
                insideConstrainedOut = d_107_i2_
                currentConstrainedOut = d_108_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

