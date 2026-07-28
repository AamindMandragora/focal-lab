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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final numeric answer inside << >>. Example: <<42>>. Keep expressions simple.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 25
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
                            if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_37_gFinal_: _dafny.Seq
                                d_38_cFinal_: _dafny.Seq
                                out29_: _dafny.Seq
                                out30_: _dafny.Seq
                                out29_, out30_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_37_gFinal_ = out29_
                                d_38_cFinal_ = out30_
                                generated = d_37_gFinal_
                                currentConstrainedOut = d_38_cFinal_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_39_g2_: _dafny.Seq
                                d_40_i2_: bool
                                d_41_c2_: _dafny.Seq
                                out31_: _dafny.Seq
                                out32_: bool
                                out33_: _dafny.Seq
                                out31_, out32_, out33_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_39_g2_ = out31_
                                d_40_i2_ = out32_
                                d_41_c2_ = out33_
                                generated = d_39_g2_
                                insideConstrainedOut = d_40_i2_
                                currentConstrainedOut = d_41_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif (d_1_steps_) < (maxSteps):
                                d_42_constrainedPrompt_: _dafny.Seq
                                d_42_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_43_next_: _dafny.Seq
                                out34_: _dafny.Seq
                                out34_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_42_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_43_next_ = out34_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_43_next_) == (eosToken):
                                    d_44_gR2_: _dafny.Seq
                                    d_45_cR2_: _dafny.Seq
                                    out35_: _dafny.Seq
                                    out36_: _dafny.Seq
                                    out35_, out36_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_44_gR2_ = out35_
                                    d_45_cR2_ = out36_
                                    generated = d_44_gR2_
                                    currentConstrainedOut = d_45_cR2_
                                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                        d_46_gF2_: _dafny.Seq
                                        d_47_cF2_: _dafny.Seq
                                        out37_: _dafny.Seq
                                        out38_: _dafny.Seq
                                        out37_, out38_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                        d_46_gF2_ = out37_
                                        d_47_cF2_ = out38_
                                        generated = d_46_gF2_
                                        currentConstrainedOut = d_47_cF2_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_48_g3_: _dafny.Seq
                                        d_49_i3_: bool
                                        d_50_c3_: _dafny.Seq
                                        out39_: _dafny.Seq
                                        out40_: bool
                                        out41_: _dafny.Seq
                                        out39_, out40_, out41_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_48_g3_ = out39_
                                        d_49_i3_ = out40_
                                        d_50_c3_ = out41_
                                        generated = d_48_g3_
                                        insideConstrainedOut = d_49_i3_
                                        currentConstrainedOut = d_50_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_51_g2_: _dafny.Seq
                                    d_52_i2_: bool
                                    d_53_c2_: _dafny.Seq
                                    out42_: _dafny.Seq
                                    out43_: bool
                                    out44_: _dafny.Seq
                                    out42_, out43_, out44_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_43_next_)
                                    d_51_g2_ = out42_
                                    d_52_i2_ = out43_
                                    d_53_c2_ = out44_
                                    generated = d_51_g2_
                                    insideConstrainedOut = d_52_i2_
                                    currentConstrainedOut = d_53_c2_
                                    d_3_spanTokensUsed_ = 1
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_54_g3_: _dafny.Seq
                                        d_55_i3_: bool
                                        d_56_c3_: _dafny.Seq
                                        out45_: _dafny.Seq
                                        out46_: bool
                                        out47_: _dafny.Seq
                                        out45_, out46_, out47_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_54_g3_ = out45_
                                        d_55_i3_ = out46_
                                        d_56_c3_ = out47_
                                        generated = d_54_g3_
                                        insideConstrainedOut = d_55_i3_
                                        currentConstrainedOut = d_56_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                        elif True:
                            d_57_constrainedPrompt_: _dafny.Seq
                            d_57_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_58_next_: _dafny.Seq
                            d_59_wasConstrained_: bool
                            out48_: _dafny.Seq
                            out49_: bool
                            out48_, out49_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_57_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_58_next_ = out48_
                            d_59_wasConstrained_ = out49_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_58_next_) == (eosToken):
                                d_60_gR_: _dafny.Seq
                                d_61_cR_: _dafny.Seq
                                out50_: _dafny.Seq
                                out51_: _dafny.Seq
                                out50_, out51_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_60_gR_ = out50_
                                d_61_cR_ = out51_
                                generated = d_60_gR_
                                currentConstrainedOut = d_61_cR_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_62_g2_: _dafny.Seq
                                    d_63_c2_: _dafny.Seq
                                    out52_: _dafny.Seq
                                    out53_: _dafny.Seq
                                    out52_, out53_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_62_g2_ = out52_
                                    d_63_c2_ = out53_
                                    generated = d_62_g2_
                                    currentConstrainedOut = d_63_c2_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_64_g3_: _dafny.Seq
                                    d_65_c3_: _dafny.Seq
                                    out54_: _dafny.Seq
                                    out55_: _dafny.Seq
                                    out54_, out55_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_64_g3_ = out54_
                                    d_65_c3_ = out55_
                                    generated = d_64_g3_
                                    currentConstrainedOut = d_65_c3_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_66_g4_: _dafny.Seq
                                    d_67_c4_: _dafny.Seq
                                    out56_: _dafny.Seq
                                    out57_: _dafny.Seq
                                    out56_, out57_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_66_g4_ = out56_
                                    d_67_c4_ = out57_
                                    generated = d_66_g4_
                                    currentConstrainedOut = d_67_c4_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_68_g5_: _dafny.Seq
                                    d_69_c5_: _dafny.Seq
                                    out58_: _dafny.Seq
                                    out59_: _dafny.Seq
                                    out58_, out59_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_68_g5_ = out58_
                                    d_69_c5_ = out59_
                                    generated = d_68_g5_
                                    currentConstrainedOut = d_69_c5_
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_70_gFinal_: _dafny.Seq
                                    d_71_cFinal_: _dafny.Seq
                                    out60_: _dafny.Seq
                                    out61_: _dafny.Seq
                                    out60_, out61_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_70_gFinal_ = out60_
                                    d_71_cFinal_ = out61_
                                    generated = d_70_gFinal_
                                    currentConstrainedOut = d_71_cFinal_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_72_g2_: _dafny.Seq
                                    d_73_i2_: bool
                                    d_74_c2_: _dafny.Seq
                                    out62_: _dafny.Seq
                                    out63_: bool
                                    out64_: _dafny.Seq
                                    out62_, out63_, out64_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_72_g2_ = out62_
                                    d_73_i2_ = out63_
                                    d_74_c2_ = out64_
                                    generated = d_72_g2_
                                    insideConstrainedOut = d_73_i2_
                                    currentConstrainedOut = d_74_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_75_g2_: _dafny.Seq
                                d_76_i2_: bool
                                d_77_c2_: _dafny.Seq
                                out65_: _dafny.Seq
                                out66_: bool
                                out67_: _dafny.Seq
                                out65_, out66_, out67_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_58_next_)
                                d_75_g2_ = out65_
                                d_76_i2_ = out66_
                                d_77_c2_ = out67_
                                generated = d_75_g2_
                                insideConstrainedOut = d_76_i2_
                                currentConstrainedOut = d_77_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_78_gFinal_: _dafny.Seq
            d_79_cFinal_: _dafny.Seq
            out68_: _dafny.Seq
            out69_: _dafny.Seq
            out68_, out69_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_78_gFinal_ = out68_
            d_79_cFinal_ = out69_
            generated = d_78_gFinal_
            currentConstrainedOut = d_79_cFinal_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_80_constrainedPrompt_: _dafny.Seq
                d_80_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_81_next_: _dafny.Seq
                out70_: _dafny.Seq
                out70_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_80_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_81_next_ = out70_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_81_next_) != (eosToken):
                    d_82_g2_: _dafny.Seq
                    d_83_i2_: bool
                    d_84_c2_: _dafny.Seq
                    out71_: _dafny.Seq
                    out72_: bool
                    out73_: _dafny.Seq
                    out71_, out72_, out73_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_81_next_)
                    d_82_g2_ = out71_
                    d_83_i2_ = out72_
                    d_84_c2_ = out73_
                    generated = d_82_g2_
                    insideConstrainedOut = d_83_i2_
                    currentConstrainedOut = d_84_c2_
                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                    d_85_gR2_: _dafny.Seq
                    d_86_cR2_: _dafny.Seq
                    out74_: _dafny.Seq
                    out75_: _dafny.Seq
                    out74_, out75_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                    d_85_gR2_ = out74_
                    d_86_cR2_ = out75_
                    generated = d_85_gR2_
                    currentConstrainedOut = d_86_cR2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_87_g2_: _dafny.Seq
                d_88_i2_: bool
                d_89_c2_: _dafny.Seq
                out76_: _dafny.Seq
                out77_: bool
                out78_: _dafny.Seq
                out76_, out77_, out78_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_87_g2_ = out76_
                d_88_i2_ = out77_
                d_89_c2_ = out78_
                generated = d_87_g2_
                insideConstrainedOut = d_88_i2_
                currentConstrainedOut = d_89_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

