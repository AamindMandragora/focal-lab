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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step. Put ONLY the final numeric answer inside << >> delimiters. Example: The answer is <<42>>. Do not put intermediate calculations in << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 15
        d_3_spanTokensUsed_: int
        d_3_spanTokensUsed_ = 0
        d_4_spanMaxTokens_: int
        d_4_spanMaxTokens_ = 8
        d_5_hasSeenOpenSpan_: bool
        d_5_hasSeenOpenSpan_ = insideConstrained
        d_6_closeReserve_: int
        d_6_closeReserve_ = 4
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_7_remaining_: int
                    d_7_remaining_ = (maxSteps) - (d_1_steps_)
                    if (insideConstrainedOut) and ((d_7_remaining_) <= (d_6_closeReserve_)):
                        d_8_gR_: _dafny.Seq
                        d_9_cR_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: _dafny.Seq
                        out0_, out1_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_8_gR_ = out0_
                        d_9_cR_ = out1_
                        generated = d_8_gR_
                        currentConstrainedOut = d_9_cR_
                        if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                            d_10_gR2_: _dafny.Seq
                            d_11_cR2_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: _dafny.Seq
                            out2_, out3_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_10_gR2_ = out2_
                            d_11_cR2_ = out3_
                            generated = d_10_gR2_
                            currentConstrainedOut = d_11_cR2_
                        if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                            d_12_gR3_: _dafny.Seq
                            d_13_cR3_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: _dafny.Seq
                            out4_, out5_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_12_gR3_ = out4_
                            d_13_cR3_ = out5_
                            generated = d_12_gR3_
                            currentConstrainedOut = d_13_cR3_
                        if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                            d_14_gR4_: _dafny.Seq
                            d_15_cR4_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: _dafny.Seq
                            out6_, out7_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_14_gR4_ = out6_
                            d_15_cR4_ = out7_
                            generated = d_14_gR4_
                            currentConstrainedOut = d_15_cR4_
                        if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                            d_16_gR5_: _dafny.Seq
                            d_17_cR5_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: _dafny.Seq
                            out8_, out9_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_16_gR5_ = out8_
                            d_17_cR5_ = out9_
                            generated = d_16_gR5_
                            currentConstrainedOut = d_17_cR5_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_18_g2_: _dafny.Seq
                            d_19_i2_: bool
                            d_20_c2_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_18_g2_ = out10_
                            d_19_i2_ = out11_
                            d_20_c2_ = out12_
                            generated = d_18_g2_
                            insideConstrainedOut = d_19_i2_
                            currentConstrainedOut = d_20_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                            d_21_constrainedPromptE_: _dafny.Seq
                            d_21_constrainedPromptE_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_22_nextE_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_21_constrainedPromptE_, currentConstrainedOut, eosToken)
                            d_22_nextE_ = out13_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_nextE_) != (eosToken):
                                d_23_gE_: _dafny.Seq
                                d_24_iE_: bool
                                d_25_cE_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextE_)
                                d_23_gE_ = out14_
                                d_24_iE_ = out15_
                                d_25_cE_ = out16_
                                generated = d_23_gE_
                                insideConstrainedOut = d_24_iE_
                                currentConstrainedOut = d_25_cE_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_26_g2_: _dafny.Seq
                                    d_27_i2_: bool
                                    d_28_c2_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_26_g2_ = out17_
                                    d_27_i2_ = out18_
                                    d_28_c2_ = out19_
                                    generated = d_26_g2_
                                    insideConstrainedOut = d_27_i2_
                                    currentConstrainedOut = d_28_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    if not(insideConstrainedOut):
                        if (((d_7_remaining_) <= (70)) and (not(d_5_hasSeenOpenSpan_))) and ((d_7_remaining_) > ((d_6_closeReserve_) + (1))):
                            d_29_g2_: _dafny.Seq
                            d_30_i2_: bool
                            d_31_c2_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_29_g2_ = out20_
                            d_30_i2_ = out21_
                            d_31_c2_ = out22_
                            generated = d_29_g2_
                            insideConstrainedOut = d_30_i2_
                            currentConstrainedOut = d_31_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                            d_5_hasSeenOpenSpan_ = True
                        elif True:
                            d_32_chunkBudget_: int
                            if (d_7_remaining_) < (d_2_freeChunkSize_):
                                d_32_chunkBudget_ = d_7_remaining_
                            elif True:
                                d_32_chunkBudget_ = d_2_freeChunkSize_
                            if (d_32_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            d_33_chunkGenerated_: _dafny.Seq
                            d_34_stoppedOnOpenSpan_: bool
                            d_35_stoppedOnEos_: bool
                            d_36_stepsUsed_: int
                            out23_: _dafny.Seq
                            out24_: bool
                            out25_: bool
                            out26_: int
                            out23_, out24_, out25_, out26_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_32_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_33_chunkGenerated_ = out23_
                            d_34_stoppedOnOpenSpan_ = out24_
                            d_35_stoppedOnEos_ = out25_
                            d_36_stepsUsed_ = out26_
                            generated = d_33_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_36_stepsUsed_)
                            if d_35_stoppedOnEos_:
                                if (not(d_5_hasSeenOpenSpan_)) and ((((d_1_steps_) + (d_6_closeReserve_)) + (2)) <= (maxSteps)):
                                    d_37_g2_: _dafny.Seq
                                    d_38_i2_: bool
                                    d_39_c2_: _dafny.Seq
                                    out27_: _dafny.Seq
                                    out28_: bool
                                    out29_: _dafny.Seq
                                    out27_, out28_, out29_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_37_g2_ = out27_
                                    d_38_i2_ = out28_
                                    d_39_c2_ = out29_
                                    generated = d_37_g2_
                                    insideConstrainedOut = d_38_i2_
                                    currentConstrainedOut = d_39_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                                    d_5_hasSeenOpenSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif d_34_stoppedOnOpenSpan_:
                                d_40_g2_: _dafny.Seq
                                d_41_i2_: bool
                                d_42_c2_: _dafny.Seq
                                out30_: _dafny.Seq
                                out31_: bool
                                out32_: _dafny.Seq
                                out30_, out31_, out32_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_40_g2_ = out30_
                                d_41_i2_ = out31_
                                d_42_c2_ = out32_
                                generated = d_40_g2_
                                insideConstrainedOut = d_41_i2_
                                currentConstrainedOut = d_42_c2_
                                d_3_spanTokensUsed_ = 0
                                d_5_hasSeenOpenSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_43_g2_: _dafny.Seq
                        d_44_i2_: bool
                        d_45_c2_: _dafny.Seq
                        out33_: _dafny.Seq
                        out34_: bool
                        out35_: _dafny.Seq
                        out33_, out34_, out35_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_43_g2_ = out33_
                        d_44_i2_ = out34_
                        d_45_c2_ = out35_
                        generated = d_43_g2_
                        insideConstrainedOut = d_44_i2_
                        currentConstrainedOut = d_45_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokensUsed_ = 0
                    elif True:
                        d_46_isDeadEnd_: bool
                        out36_: bool
                        out36_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_46_isDeadEnd_ = out36_
                        if (d_46_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_)):
                            d_47_gRolled_: _dafny.Seq
                            d_48_cRolled_: _dafny.Seq
                            out37_: _dafny.Seq
                            out38_: _dafny.Seq
                            out37_, out38_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_47_gRolled_ = out37_
                            d_48_cRolled_ = out38_
                            generated = d_47_gRolled_
                            currentConstrainedOut = d_48_cRolled_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_49_g3_: _dafny.Seq
                                d_50_c3_: _dafny.Seq
                                out39_: _dafny.Seq
                                out40_: _dafny.Seq
                                out39_, out40_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_49_g3_ = out39_
                                d_50_c3_ = out40_
                                generated = d_49_g3_
                                currentConstrainedOut = d_50_c3_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_51_g4_: _dafny.Seq
                                d_52_c4_: _dafny.Seq
                                out41_: _dafny.Seq
                                out42_: _dafny.Seq
                                out41_, out42_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_51_g4_ = out41_
                                d_52_c4_ = out42_
                                generated = d_51_g4_
                                currentConstrainedOut = d_52_c4_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_53_g5_: _dafny.Seq
                                d_54_c5_: _dafny.Seq
                                out43_: _dafny.Seq
                                out44_: _dafny.Seq
                                out43_, out44_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_53_g5_ = out43_
                                d_54_c5_ = out44_
                                generated = d_53_g5_
                                currentConstrainedOut = d_54_c5_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_55_g6_: _dafny.Seq
                                d_56_c6_: _dafny.Seq
                                out45_: _dafny.Seq
                                out46_: _dafny.Seq
                                out45_, out46_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_55_g6_ = out45_
                                d_56_c6_ = out46_
                                generated = d_55_g6_
                                currentConstrainedOut = d_56_c6_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_57_g7_: _dafny.Seq
                                d_58_c7_: _dafny.Seq
                                out47_: _dafny.Seq
                                out48_: _dafny.Seq
                                out47_, out48_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_57_g7_ = out47_
                                d_58_c7_ = out48_
                                generated = d_57_g7_
                                currentConstrainedOut = d_58_c7_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_59_g2_: _dafny.Seq
                                d_60_i2_: bool
                                d_61_c2_: _dafny.Seq
                                out49_: _dafny.Seq
                                out50_: bool
                                out51_: _dafny.Seq
                                out49_, out50_, out51_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_59_g2_ = out49_
                                d_60_i2_ = out50_
                                d_61_c2_ = out51_
                                generated = d_59_g2_
                                insideConstrainedOut = d_60_i2_
                                currentConstrainedOut = d_61_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif (d_1_steps_) < (maxSteps):
                                d_62_constrainedPrompt_: _dafny.Seq
                                d_62_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_63_next_: _dafny.Seq
                                out52_: _dafny.Seq
                                out52_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_62_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_63_next_ = out52_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_63_next_) == (eosToken):
                                    d_64_gR2_: _dafny.Seq
                                    d_65_cR2_: _dafny.Seq
                                    out53_: _dafny.Seq
                                    out54_: _dafny.Seq
                                    out53_, out54_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_64_gR2_ = out53_
                                    d_65_cR2_ = out54_
                                    generated = d_64_gR2_
                                    currentConstrainedOut = d_65_cR2_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_66_gR3_: _dafny.Seq
                                        d_67_cR3_: _dafny.Seq
                                        out55_: _dafny.Seq
                                        out56_: _dafny.Seq
                                        out55_, out56_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_66_gR3_ = out55_
                                        d_67_cR3_ = out56_
                                        generated = d_66_gR3_
                                        currentConstrainedOut = d_67_cR3_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_68_gR4_: _dafny.Seq
                                        d_69_cR4_: _dafny.Seq
                                        out57_: _dafny.Seq
                                        out58_: _dafny.Seq
                                        out57_, out58_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_68_gR4_ = out57_
                                        d_69_cR4_ = out58_
                                        generated = d_68_gR4_
                                        currentConstrainedOut = d_69_cR4_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_70_g2_: _dafny.Seq
                                        d_71_i2_: bool
                                        d_72_c2_: _dafny.Seq
                                        out59_: _dafny.Seq
                                        out60_: bool
                                        out61_: _dafny.Seq
                                        out59_, out60_, out61_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_70_g2_ = out59_
                                        d_71_i2_ = out60_
                                        d_72_c2_ = out61_
                                        generated = d_70_g2_
                                        insideConstrainedOut = d_71_i2_
                                        currentConstrainedOut = d_72_c2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_73_g2_: _dafny.Seq
                                    d_74_i2_: bool
                                    d_75_c2_: _dafny.Seq
                                    out62_: _dafny.Seq
                                    out63_: bool
                                    out64_: _dafny.Seq
                                    out62_, out63_, out64_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_63_next_)
                                    d_73_g2_ = out62_
                                    d_74_i2_ = out63_
                                    d_75_c2_ = out64_
                                    generated = d_73_g2_
                                    insideConstrainedOut = d_74_i2_
                                    currentConstrainedOut = d_75_c2_
                                    d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_76_g3_: _dafny.Seq
                                        d_77_i3_: bool
                                        d_78_c3_: _dafny.Seq
                                        out65_: _dafny.Seq
                                        out66_: bool
                                        out67_: _dafny.Seq
                                        out65_, out66_, out67_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_76_g3_ = out65_
                                        d_77_i3_ = out66_
                                        d_78_c3_ = out67_
                                        generated = d_76_g3_
                                        insideConstrainedOut = d_77_i3_
                                        currentConstrainedOut = d_78_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                        elif True:
                            d_79_constrainedPrompt_: _dafny.Seq
                            d_79_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_80_next_: _dafny.Seq
                            out68_: _dafny.Seq
                            out68_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_79_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_80_next_ = out68_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_80_next_) == (eosToken):
                                d_81_gRolled_: _dafny.Seq
                                d_82_cRolled_: _dafny.Seq
                                out69_: _dafny.Seq
                                out70_: _dafny.Seq
                                out69_, out70_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_81_gRolled_ = out69_
                                d_82_cRolled_ = out70_
                                generated = d_81_gRolled_
                                currentConstrainedOut = d_82_cRolled_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_83_gR2_: _dafny.Seq
                                    d_84_cR2_: _dafny.Seq
                                    out71_: _dafny.Seq
                                    out72_: _dafny.Seq
                                    out71_, out72_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_83_gR2_ = out71_
                                    d_84_cR2_ = out72_
                                    generated = d_83_gR2_
                                    currentConstrainedOut = d_84_cR2_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_85_gR3_: _dafny.Seq
                                    d_86_cR3_: _dafny.Seq
                                    out73_: _dafny.Seq
                                    out74_: _dafny.Seq
                                    out73_, out74_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_85_gR3_ = out73_
                                    d_86_cR3_ = out74_
                                    generated = d_85_gR3_
                                    currentConstrainedOut = d_86_cR3_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_87_gR4_: _dafny.Seq
                                    d_88_cR4_: _dafny.Seq
                                    out75_: _dafny.Seq
                                    out76_: _dafny.Seq
                                    out75_, out76_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_87_gR4_ = out75_
                                    d_88_cR4_ = out76_
                                    generated = d_87_gR4_
                                    currentConstrainedOut = d_88_cR4_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_89_gR5_: _dafny.Seq
                                    d_90_cR5_: _dafny.Seq
                                    out77_: _dafny.Seq
                                    out78_: _dafny.Seq
                                    out77_, out78_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_89_gR5_ = out77_
                                    d_90_cR5_ = out78_
                                    generated = d_89_gR5_
                                    currentConstrainedOut = d_90_cR5_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_91_g2_: _dafny.Seq
                                    d_92_i2_: bool
                                    d_93_c2_: _dafny.Seq
                                    out79_: _dafny.Seq
                                    out80_: bool
                                    out81_: _dafny.Seq
                                    out79_, out80_, out81_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_91_g2_ = out79_
                                    d_92_i2_ = out80_
                                    d_93_c2_ = out81_
                                    generated = d_91_g2_
                                    insideConstrainedOut = d_92_i2_
                                    currentConstrainedOut = d_93_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_94_g2_: _dafny.Seq
                                d_95_i2_: bool
                                d_96_c2_: _dafny.Seq
                                out82_: _dafny.Seq
                                out83_: bool
                                out84_: _dafny.Seq
                                out82_, out83_, out84_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_80_next_)
                                d_94_g2_ = out82_
                                d_95_i2_ = out83_
                                d_96_c2_ = out84_
                                generated = d_94_g2_
                                insideConstrainedOut = d_95_i2_
                                currentConstrainedOut = d_96_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_97_gRolled_: _dafny.Seq
            d_98_cRolled_: _dafny.Seq
            out85_: _dafny.Seq
            out86_: _dafny.Seq
            out85_, out86_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
            d_97_gRolled_ = out85_
            d_98_cRolled_ = out86_
            generated = d_97_gRolled_
            currentConstrainedOut = d_98_cRolled_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_99_g2_: _dafny.Seq
                d_100_c2_: _dafny.Seq
                out87_: _dafny.Seq
                out88_: _dafny.Seq
                out87_, out88_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_99_g2_ = out87_
                d_100_c2_ = out88_
                generated = d_99_g2_
                currentConstrainedOut = d_100_c2_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_101_g3_: _dafny.Seq
                d_102_c3_: _dafny.Seq
                out89_: _dafny.Seq
                out90_: _dafny.Seq
                out89_, out90_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_101_g3_ = out89_
                d_102_c3_ = out90_
                generated = d_101_g3_
                currentConstrainedOut = d_102_c3_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_103_g4_: _dafny.Seq
                d_104_c4_: _dafny.Seq
                out91_: _dafny.Seq
                out92_: _dafny.Seq
                out91_, out92_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_103_g4_ = out91_
                d_104_c4_ = out92_
                generated = d_103_g4_
                currentConstrainedOut = d_104_c4_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_105_g5_: _dafny.Seq
                d_106_c5_: _dafny.Seq
                out93_: _dafny.Seq
                out94_: _dafny.Seq
                out93_, out94_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_105_g5_ = out93_
                d_106_c5_ = out94_
                generated = d_105_g5_
                currentConstrainedOut = d_106_c5_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_107_constrainedPromptPost_: _dafny.Seq
                d_107_constrainedPromptPost_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_108_nextPost_: _dafny.Seq
                out95_: _dafny.Seq
                out95_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_107_constrainedPromptPost_, currentConstrainedOut, eosToken)
                d_108_nextPost_ = out95_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_108_nextPost_) != (eosToken):
                    d_109_gp_: _dafny.Seq
                    d_110_ip_: bool
                    d_111_cp_: _dafny.Seq
                    out96_: _dafny.Seq
                    out97_: bool
                    out98_: _dafny.Seq
                    out96_, out97_, out98_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_108_nextPost_)
                    d_109_gp_ = out96_
                    d_110_ip_ = out97_
                    d_111_cp_ = out98_
                    generated = d_109_gp_
                    insideConstrainedOut = d_110_ip_
                    currentConstrainedOut = d_111_cp_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_112_g2_: _dafny.Seq
                d_113_i2_: bool
                d_114_c2_: _dafny.Seq
                out99_: _dafny.Seq
                out100_: bool
                out101_: _dafny.Seq
                out99_, out100_, out101_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_112_g2_ = out99_
                d_113_i2_ = out100_
                d_114_c2_ = out101_
                generated = d_112_g2_
                insideConstrainedOut = d_113_i2_
                currentConstrainedOut = d_114_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

