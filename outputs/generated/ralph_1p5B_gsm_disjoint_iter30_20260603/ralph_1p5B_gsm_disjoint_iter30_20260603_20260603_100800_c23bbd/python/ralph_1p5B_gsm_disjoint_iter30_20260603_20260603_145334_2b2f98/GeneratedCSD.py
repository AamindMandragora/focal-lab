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
        d_4_spanMaxTokens_ = 8
        d_5_hasSeenOpenSpan_: bool
        d_5_hasSeenOpenSpan_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remaining_: int
                        d_6_remaining_ = (maxSteps) - (d_1_steps_)
                        if (((d_6_remaining_) <= (65)) and (not(d_5_hasSeenOpenSpan_))) and ((d_6_remaining_) > (2)):
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
                            d_25_gR_: _dafny.Seq
                            d_26_cR_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: _dafny.Seq
                            out17_, out18_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_25_gR_ = out17_
                            d_26_cR_ = out18_
                            generated = d_25_gR_
                            currentConstrainedOut = d_26_cR_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_27_g2_: _dafny.Seq
                                d_28_c2_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: _dafny.Seq
                                out19_, out20_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_27_g2_ = out19_
                                d_28_c2_ = out20_
                                generated = d_27_g2_
                                currentConstrainedOut = d_28_c2_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_29_g3_: _dafny.Seq
                                d_30_c3_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: _dafny.Seq
                                out21_, out22_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_29_g3_ = out21_
                                d_30_c3_ = out22_
                                generated = d_29_g3_
                                currentConstrainedOut = d_30_c3_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_31_g4_: _dafny.Seq
                                d_32_c4_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: _dafny.Seq
                                out23_, out24_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_31_g4_ = out23_
                                d_32_c4_ = out24_
                                generated = d_31_g4_
                                currentConstrainedOut = d_32_c4_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_33_g5_: _dafny.Seq
                                d_34_c5_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: _dafny.Seq
                                out25_, out26_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_33_g5_ = out25_
                                d_34_c5_ = out26_
                                generated = d_33_g5_
                                currentConstrainedOut = d_34_c5_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_35_g6_: _dafny.Seq
                                d_36_c6_: _dafny.Seq
                                out27_: _dafny.Seq
                                out28_: _dafny.Seq
                                out27_, out28_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_35_g6_ = out27_
                                d_36_c6_ = out28_
                                generated = d_35_g6_
                                currentConstrainedOut = d_36_c6_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_37_g7_: _dafny.Seq
                                d_38_c7_: _dafny.Seq
                                out29_: _dafny.Seq
                                out30_: _dafny.Seq
                                out29_, out30_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_37_g7_ = out29_
                                d_38_c7_ = out30_
                                generated = d_37_g7_
                                currentConstrainedOut = d_38_c7_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_39_g8_: _dafny.Seq
                                d_40_c8_: _dafny.Seq
                                out31_: _dafny.Seq
                                out32_: _dafny.Seq
                                out31_, out32_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_39_g8_ = out31_
                                d_40_c8_ = out32_
                                generated = d_39_g8_
                                currentConstrainedOut = d_40_c8_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_41_g2_: _dafny.Seq
                                d_42_i2_: bool
                                d_43_c2_: _dafny.Seq
                                out33_: _dafny.Seq
                                out34_: bool
                                out35_: _dafny.Seq
                                out33_, out34_, out35_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_41_g2_ = out33_
                                d_42_i2_ = out34_
                                d_43_c2_ = out35_
                                generated = d_41_g2_
                                insideConstrainedOut = d_42_i2_
                                currentConstrainedOut = d_43_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif (d_1_steps_) < (maxSteps):
                                d_44_next_: _dafny.Seq
                                out36_: _dafny.Seq
                                out36_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_44_next_ = out36_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_44_next_) == (eosToken):
                                    d_45_gR2_: _dafny.Seq
                                    d_46_cR2_: _dafny.Seq
                                    out37_: _dafny.Seq
                                    out38_: _dafny.Seq
                                    out37_, out38_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_45_gR2_ = out37_
                                    d_46_cR2_ = out38_
                                    generated = d_45_gR2_
                                    currentConstrainedOut = d_46_cR2_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_47_g3_: _dafny.Seq
                                        d_48_c3_: _dafny.Seq
                                        out39_: _dafny.Seq
                                        out40_: _dafny.Seq
                                        out39_, out40_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_47_g3_ = out39_
                                        d_48_c3_ = out40_
                                        generated = d_47_g3_
                                        currentConstrainedOut = d_48_c3_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_49_g4_: _dafny.Seq
                                        d_50_c4_: _dafny.Seq
                                        out41_: _dafny.Seq
                                        out42_: _dafny.Seq
                                        out41_, out42_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_49_g4_ = out41_
                                        d_50_c4_ = out42_
                                        generated = d_49_g4_
                                        currentConstrainedOut = d_50_c4_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_51_g5_: _dafny.Seq
                                        d_52_c5_: _dafny.Seq
                                        out43_: _dafny.Seq
                                        out44_: _dafny.Seq
                                        out43_, out44_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_51_g5_ = out43_
                                        d_52_c5_ = out44_
                                        generated = d_51_g5_
                                        currentConstrainedOut = d_52_c5_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_53_g2_: _dafny.Seq
                                        d_54_i2_: bool
                                        d_55_c2_: _dafny.Seq
                                        out45_: _dafny.Seq
                                        out46_: bool
                                        out47_: _dafny.Seq
                                        out45_, out46_, out47_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_53_g2_ = out45_
                                        d_54_i2_ = out46_
                                        d_55_c2_ = out47_
                                        generated = d_53_g2_
                                        insideConstrainedOut = d_54_i2_
                                        currentConstrainedOut = d_55_c2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    d_56_g2_: _dafny.Seq
                                    d_57_i2_: bool
                                    d_58_c2_: _dafny.Seq
                                    out48_: _dafny.Seq
                                    out49_: bool
                                    out50_: _dafny.Seq
                                    out48_, out49_, out50_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_44_next_)
                                    d_56_g2_ = out48_
                                    d_57_i2_ = out49_
                                    d_58_c2_ = out50_
                                    generated = d_56_g2_
                                    insideConstrainedOut = d_57_i2_
                                    currentConstrainedOut = d_58_c2_
                                    d_3_spanTokensUsed_ = 1
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_59_g3_: _dafny.Seq
                                        d_60_i3_: bool
                                        d_61_c3_: _dafny.Seq
                                        out51_: _dafny.Seq
                                        out52_: bool
                                        out53_: _dafny.Seq
                                        out51_, out52_, out53_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_59_g3_ = out51_
                                        d_60_i3_ = out52_
                                        d_61_c3_ = out53_
                                        generated = d_59_g3_
                                        insideConstrainedOut = d_60_i3_
                                        currentConstrainedOut = d_61_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                        elif True:
                            d_62_next_: _dafny.Seq
                            d_63_wasConstrained_: bool
                            out54_: _dafny.Seq
                            out55_: bool
                            out54_, out55_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_62_next_ = out54_
                            d_63_wasConstrained_ = out55_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_62_next_) == (eosToken):
                                d_64_gR_: _dafny.Seq
                                d_65_cR_: _dafny.Seq
                                out56_: _dafny.Seq
                                out57_: _dafny.Seq
                                out56_, out57_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_64_gR_ = out56_
                                d_65_cR_ = out57_
                                generated = d_64_gR_
                                currentConstrainedOut = d_65_cR_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_66_g2_: _dafny.Seq
                                    d_67_c2_: _dafny.Seq
                                    out58_: _dafny.Seq
                                    out59_: _dafny.Seq
                                    out58_, out59_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_66_g2_ = out58_
                                    d_67_c2_ = out59_
                                    generated = d_66_g2_
                                    currentConstrainedOut = d_67_c2_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_68_g3_: _dafny.Seq
                                    d_69_c3_: _dafny.Seq
                                    out60_: _dafny.Seq
                                    out61_: _dafny.Seq
                                    out60_, out61_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_68_g3_ = out60_
                                    d_69_c3_ = out61_
                                    generated = d_68_g3_
                                    currentConstrainedOut = d_69_c3_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_70_g4_: _dafny.Seq
                                    d_71_c4_: _dafny.Seq
                                    out62_: _dafny.Seq
                                    out63_: _dafny.Seq
                                    out62_, out63_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_70_g4_ = out62_
                                    d_71_c4_ = out63_
                                    generated = d_70_g4_
                                    currentConstrainedOut = d_71_c4_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_72_g5_: _dafny.Seq
                                    d_73_c5_: _dafny.Seq
                                    out64_: _dafny.Seq
                                    out65_: _dafny.Seq
                                    out64_, out65_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_72_g5_ = out64_
                                    d_73_c5_ = out65_
                                    generated = d_72_g5_
                                    currentConstrainedOut = d_73_c5_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_74_g6_: _dafny.Seq
                                    d_75_c6_: _dafny.Seq
                                    out66_: _dafny.Seq
                                    out67_: _dafny.Seq
                                    out66_, out67_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_74_g6_ = out66_
                                    d_75_c6_ = out67_
                                    generated = d_74_g6_
                                    currentConstrainedOut = d_75_c6_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_76_g2_: _dafny.Seq
                                    d_77_i2_: bool
                                    d_78_c2_: _dafny.Seq
                                    out68_: _dafny.Seq
                                    out69_: bool
                                    out70_: _dafny.Seq
                                    out68_, out69_, out70_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_76_g2_ = out68_
                                    d_77_i2_ = out69_
                                    d_78_c2_ = out70_
                                    generated = d_76_g2_
                                    insideConstrainedOut = d_77_i2_
                                    currentConstrainedOut = d_78_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_79_g2_: _dafny.Seq
                                d_80_i2_: bool
                                d_81_c2_: _dafny.Seq
                                out71_: _dafny.Seq
                                out72_: bool
                                out73_: _dafny.Seq
                                out71_, out72_, out73_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_62_next_)
                                d_79_g2_ = out71_
                                d_80_i2_ = out72_
                                d_81_c2_ = out73_
                                generated = d_79_g2_
                                insideConstrainedOut = d_80_i2_
                                currentConstrainedOut = d_81_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_82_gR_: _dafny.Seq
            d_83_cR_: _dafny.Seq
            out74_: _dafny.Seq
            out75_: _dafny.Seq
            out74_, out75_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
            d_82_gR_ = out74_
            d_83_cR_ = out75_
            generated = d_82_gR_
            currentConstrainedOut = d_83_cR_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_84_g2_: _dafny.Seq
                d_85_c2_: _dafny.Seq
                out76_: _dafny.Seq
                out77_: _dafny.Seq
                out76_, out77_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_84_g2_ = out76_
                d_85_c2_ = out77_
                generated = d_84_g2_
                currentConstrainedOut = d_85_c2_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_86_g3_: _dafny.Seq
                d_87_c3_: _dafny.Seq
                out78_: _dafny.Seq
                out79_: _dafny.Seq
                out78_, out79_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_86_g3_ = out78_
                d_87_c3_ = out79_
                generated = d_86_g3_
                currentConstrainedOut = d_87_c3_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_88_g4_: _dafny.Seq
                d_89_c4_: _dafny.Seq
                out80_: _dafny.Seq
                out81_: _dafny.Seq
                out80_, out81_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_88_g4_ = out80_
                d_89_c4_ = out81_
                generated = d_88_g4_
                currentConstrainedOut = d_89_c4_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_90_g5_: _dafny.Seq
                d_91_c5_: _dafny.Seq
                out82_: _dafny.Seq
                out83_: _dafny.Seq
                out82_, out83_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_90_g5_ = out82_
                d_91_c5_ = out83_
                generated = d_90_g5_
                currentConstrainedOut = d_91_c5_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_92_g6_: _dafny.Seq
                d_93_c6_: _dafny.Seq
                out84_: _dafny.Seq
                out85_: _dafny.Seq
                out84_, out85_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_92_g6_ = out84_
                d_93_c6_ = out85_
                generated = d_92_g6_
                currentConstrainedOut = d_93_c6_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_94_g7_: _dafny.Seq
                d_95_c7_: _dafny.Seq
                out86_: _dafny.Seq
                out87_: _dafny.Seq
                out86_, out87_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_94_g7_ = out86_
                d_95_c7_ = out87_
                generated = d_94_g7_
                currentConstrainedOut = d_95_c7_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_96_g8_: _dafny.Seq
                d_97_c8_: _dafny.Seq
                out88_: _dafny.Seq
                out89_: _dafny.Seq
                out88_, out89_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_96_g8_ = out88_
                d_97_c8_ = out89_
                generated = d_96_g8_
                currentConstrainedOut = d_97_c8_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_98_g9_: _dafny.Seq
                d_99_c9_: _dafny.Seq
                out90_: _dafny.Seq
                out91_: _dafny.Seq
                out90_, out91_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_98_g9_ = out90_
                d_99_c9_ = out91_
                generated = d_98_g9_
                currentConstrainedOut = d_99_c9_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_100_g10_: _dafny.Seq
                d_101_c10_: _dafny.Seq
                out92_: _dafny.Seq
                out93_: _dafny.Seq
                out92_, out93_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_100_g10_ = out92_
                d_101_c10_ = out93_
                generated = d_100_g10_
                currentConstrainedOut = d_101_c10_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_102_next_: _dafny.Seq
                out94_: _dafny.Seq
                out94_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                d_102_next_ = out94_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_102_next_) != (eosToken):
                    d_103_g2_: _dafny.Seq
                    d_104_i2_: bool
                    d_105_c2_: _dafny.Seq
                    out95_: _dafny.Seq
                    out96_: bool
                    out97_: _dafny.Seq
                    out95_, out96_, out97_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_102_next_)
                    d_103_g2_ = out95_
                    d_104_i2_ = out96_
                    d_105_c2_ = out97_
                    generated = d_103_g2_
                    insideConstrainedOut = d_104_i2_
                    currentConstrainedOut = d_105_c2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_106_g2_: _dafny.Seq
                d_107_i2_: bool
                d_108_c2_: _dafny.Seq
                out98_: _dafny.Seq
                out99_: bool
                out100_: _dafny.Seq
                out98_, out99_, out100_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_106_g2_ = out98_
                d_107_i2_ = out99_
                d_108_c2_ = out100_
                generated = d_106_g2_
                insideConstrainedOut = d_107_i2_
                currentConstrainedOut = d_108_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

