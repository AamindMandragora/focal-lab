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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final numeric answer inside << >>. Example: <<42>>. Keep the answer expression short.")))
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
        d_6_rollbackCount_: int
        d_6_rollbackCount_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_remaining_: int
                        d_7_remaining_ = (maxSteps) - (d_1_steps_)
                        if (((d_7_remaining_) <= (65)) and (not(d_5_hasSeenOpenSpan_))) and ((d_7_remaining_) > (2)):
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
                            d_6_rollbackCount_ = 0
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
                                    d_6_rollbackCount_ = 0
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
                                d_6_rollbackCount_ = 0
                                d_5_hasSeenOpenSpan_ = True
                    elif True:
                        d_22_remaining_: int
                        d_22_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_22_remaining_) <= (3):
                            d_23_gFinal_: _dafny.Seq
                            d_24_cFinal_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: _dafny.Seq
                            out13_, out14_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_23_gFinal_ = out13_
                            d_24_cFinal_ = out14_
                            generated = d_23_gFinal_
                            currentConstrainedOut = d_24_cFinal_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_25_g2_: _dafny.Seq
                                d_26_i2_: bool
                                d_27_c2_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_25_g2_ = out15_
                                d_26_i2_ = out16_
                                d_27_c2_ = out17_
                                generated = d_25_g2_
                                insideConstrainedOut = d_26_i2_
                                currentConstrainedOut = d_27_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif (parser).IsCompletePrefix(currentConstrainedOut):
                            d_28_g2_: _dafny.Seq
                            d_29_i2_: bool
                            d_30_c2_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: bool
                            out20_: _dafny.Seq
                            out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_28_g2_ = out18_
                            d_29_i2_ = out19_
                            d_30_c2_ = out20_
                            generated = d_28_g2_
                            insideConstrainedOut = d_29_i2_
                            currentConstrainedOut = d_30_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                            d_6_rollbackCount_ = 0
                        elif True:
                            d_31_effectiveMax_: int
                            if (d_6_rollbackCount_) >= (2):
                                d_31_effectiveMax_ = 4
                            elif True:
                                d_31_effectiveMax_ = d_4_spanMaxTokens_
                            d_32_isDeadEnd_: bool
                            out21_: bool
                            out21_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_32_isDeadEnd_ = out21_
                            if (d_32_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (d_31_effectiveMax_)):
                                d_33_gR1_: _dafny.Seq
                                d_34_cR1_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: _dafny.Seq
                                out22_, out23_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_33_gR1_ = out22_
                                d_34_cR1_ = out23_
                                generated = d_33_gR1_
                                currentConstrainedOut = d_34_cR1_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_35_gR2_: _dafny.Seq
                                    d_36_cR2_: _dafny.Seq
                                    out24_: _dafny.Seq
                                    out25_: _dafny.Seq
                                    out24_, out25_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_35_gR2_ = out24_
                                    d_36_cR2_ = out25_
                                    generated = d_35_gR2_
                                    currentConstrainedOut = d_36_cR2_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_37_gR3_: _dafny.Seq
                                    d_38_cR3_: _dafny.Seq
                                    out26_: _dafny.Seq
                                    out27_: _dafny.Seq
                                    out26_, out27_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_37_gR3_ = out26_
                                    d_38_cR3_ = out27_
                                    generated = d_37_gR3_
                                    currentConstrainedOut = d_38_cR3_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_39_gR4_: _dafny.Seq
                                    d_40_cR4_: _dafny.Seq
                                    out28_: _dafny.Seq
                                    out29_: _dafny.Seq
                                    out28_, out29_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_39_gR4_ = out28_
                                    d_40_cR4_ = out29_
                                    generated = d_39_gR4_
                                    currentConstrainedOut = d_40_cR4_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_41_gR5_: _dafny.Seq
                                    d_42_cR5_: _dafny.Seq
                                    out30_: _dafny.Seq
                                    out31_: _dafny.Seq
                                    out30_, out31_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_41_gR5_ = out30_
                                    d_42_cR5_ = out31_
                                    generated = d_41_gR5_
                                    currentConstrainedOut = d_42_cR5_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_43_gR6_: _dafny.Seq
                                    d_44_cR6_: _dafny.Seq
                                    out32_: _dafny.Seq
                                    out33_: _dafny.Seq
                                    out32_, out33_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_43_gR6_ = out32_
                                    d_44_cR6_ = out33_
                                    generated = d_43_gR6_
                                    currentConstrainedOut = d_44_cR6_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_45_g2_: _dafny.Seq
                                    d_46_i2_: bool
                                    d_47_c2_: _dafny.Seq
                                    out34_: _dafny.Seq
                                    out35_: bool
                                    out36_: _dafny.Seq
                                    out34_, out35_, out36_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_45_g2_ = out34_
                                    d_46_i2_ = out35_
                                    d_47_c2_ = out36_
                                    generated = d_45_g2_
                                    insideConstrainedOut = d_46_i2_
                                    currentConstrainedOut = d_47_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_6_rollbackCount_ = 0
                                elif (d_1_steps_) < (maxSteps):
                                    d_48_gFinal_: _dafny.Seq
                                    d_49_cFinal_: _dafny.Seq
                                    out37_: _dafny.Seq
                                    out38_: _dafny.Seq
                                    out37_, out38_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_48_gFinal_ = out37_
                                    d_49_cFinal_ = out38_
                                    generated = d_48_gFinal_
                                    currentConstrainedOut = d_49_cFinal_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_50_g2_: _dafny.Seq
                                        d_51_i2_: bool
                                        d_52_c2_: _dafny.Seq
                                        out39_: _dafny.Seq
                                        out40_: bool
                                        out41_: _dafny.Seq
                                        out39_, out40_, out41_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_50_g2_ = out39_
                                        d_51_i2_ = out40_
                                        d_52_c2_ = out41_
                                        generated = d_50_g2_
                                        insideConstrainedOut = d_51_i2_
                                        currentConstrainedOut = d_52_c2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_6_rollbackCount_ = 0
                                    elif True:
                                        d_6_rollbackCount_ = (d_6_rollbackCount_) + (1)
                                        d_53_constrainedPrompt_: _dafny.Seq
                                        d_53_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                        d_54_next_: _dafny.Seq
                                        out42_: _dafny.Seq
                                        out42_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_53_constrainedPrompt_, currentConstrainedOut, eosToken)
                                        d_54_next_ = out42_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if (d_54_next_) == (eosToken):
                                            d_55_gFinal2_: _dafny.Seq
                                            d_56_cFinal2_: _dafny.Seq
                                            out43_: _dafny.Seq
                                            out44_: _dafny.Seq
                                            out43_, out44_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                            d_55_gFinal2_ = out43_
                                            d_56_cFinal2_ = out44_
                                            generated = d_55_gFinal2_
                                            currentConstrainedOut = d_56_cFinal2_
                                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                                d_57_g3_: _dafny.Seq
                                                d_58_i3_: bool
                                                d_59_c3_: _dafny.Seq
                                                out45_: _dafny.Seq
                                                out46_: bool
                                                out47_: _dafny.Seq
                                                out45_, out46_, out47_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_57_g3_ = out45_
                                                d_58_i3_ = out46_
                                                d_59_c3_ = out47_
                                                generated = d_57_g3_
                                                insideConstrainedOut = d_58_i3_
                                                currentConstrainedOut = d_59_c3_
                                                d_1_steps_ = (d_1_steps_) + (1)
                                                d_6_rollbackCount_ = 0
                                        elif True:
                                            d_60_g2_: _dafny.Seq
                                            d_61_i2_: bool
                                            d_62_c2_: _dafny.Seq
                                            out48_: _dafny.Seq
                                            out49_: bool
                                            out50_: _dafny.Seq
                                            out48_, out49_, out50_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_54_next_)
                                            d_60_g2_ = out48_
                                            d_61_i2_ = out49_
                                            d_62_c2_ = out50_
                                            generated = d_60_g2_
                                            insideConstrainedOut = d_61_i2_
                                            currentConstrainedOut = d_62_c2_
                                            d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                                d_63_g3_: _dafny.Seq
                                                d_64_i3_: bool
                                                d_65_c3_: _dafny.Seq
                                                out51_: _dafny.Seq
                                                out52_: bool
                                                out53_: _dafny.Seq
                                                out51_, out52_, out53_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_63_g3_ = out51_
                                                d_64_i3_ = out52_
                                                d_65_c3_ = out53_
                                                generated = d_63_g3_
                                                insideConstrainedOut = d_64_i3_
                                                currentConstrainedOut = d_65_c3_
                                                d_1_steps_ = (d_1_steps_) + (1)
                                                d_6_rollbackCount_ = 0
                                                d_3_spanTokensUsed_ = 0
                            elif True:
                                d_66_constrainedPrompt_: _dafny.Seq
                                d_66_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_67_next_: _dafny.Seq
                                d_68_wasConstrained_: bool
                                out54_: _dafny.Seq
                                out55_: bool
                                out54_, out55_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_66_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_67_next_ = out54_
                                d_68_wasConstrained_ = out55_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_67_next_) == (eosToken):
                                    d_69_gR1_: _dafny.Seq
                                    d_70_cR1_: _dafny.Seq
                                    out56_: _dafny.Seq
                                    out57_: _dafny.Seq
                                    out56_, out57_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_69_gR1_ = out56_
                                    d_70_cR1_ = out57_
                                    generated = d_69_gR1_
                                    currentConstrainedOut = d_70_cR1_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_71_gR2_: _dafny.Seq
                                        d_72_cR2_: _dafny.Seq
                                        out58_: _dafny.Seq
                                        out59_: _dafny.Seq
                                        out58_, out59_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_71_gR2_ = out58_
                                        d_72_cR2_ = out59_
                                        generated = d_71_gR2_
                                        currentConstrainedOut = d_72_cR2_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_73_gR3_: _dafny.Seq
                                        d_74_cR3_: _dafny.Seq
                                        out60_: _dafny.Seq
                                        out61_: _dafny.Seq
                                        out60_, out61_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_73_gR3_ = out60_
                                        d_74_cR3_ = out61_
                                        generated = d_73_gR3_
                                        currentConstrainedOut = d_74_cR3_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_75_gR4_: _dafny.Seq
                                        d_76_cR4_: _dafny.Seq
                                        out62_: _dafny.Seq
                                        out63_: _dafny.Seq
                                        out62_, out63_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_75_gR4_ = out62_
                                        d_76_cR4_ = out63_
                                        generated = d_75_gR4_
                                        currentConstrainedOut = d_76_cR4_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_77_gR5_: _dafny.Seq
                                        d_78_cR5_: _dafny.Seq
                                        out64_: _dafny.Seq
                                        out65_: _dafny.Seq
                                        out64_, out65_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_77_gR5_ = out64_
                                        d_78_cR5_ = out65_
                                        generated = d_77_gR5_
                                        currentConstrainedOut = d_78_cR5_
                                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                        d_79_gFinal_: _dafny.Seq
                                        d_80_cFinal_: _dafny.Seq
                                        out66_: _dafny.Seq
                                        out67_: _dafny.Seq
                                        out66_, out67_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                        d_79_gFinal_ = out66_
                                        d_80_cFinal_ = out67_
                                        generated = d_79_gFinal_
                                        currentConstrainedOut = d_80_cFinal_
                                        if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                            d_6_rollbackCount_ = (d_6_rollbackCount_) + (1)
                                    d_3_spanTokensUsed_ = 0
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_81_g2_: _dafny.Seq
                                        d_82_i2_: bool
                                        d_83_c2_: _dafny.Seq
                                        out68_: _dafny.Seq
                                        out69_: bool
                                        out70_: _dafny.Seq
                                        out68_, out69_, out70_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_81_g2_ = out68_
                                        d_82_i2_ = out69_
                                        d_83_c2_ = out70_
                                        generated = d_81_g2_
                                        insideConstrainedOut = d_82_i2_
                                        currentConstrainedOut = d_83_c2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_6_rollbackCount_ = 0
                                elif True:
                                    d_84_g2_: _dafny.Seq
                                    d_85_i2_: bool
                                    d_86_c2_: _dafny.Seq
                                    out71_: _dafny.Seq
                                    out72_: bool
                                    out73_: _dafny.Seq
                                    out71_, out72_, out73_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_67_next_)
                                    d_84_g2_ = out71_
                                    d_85_i2_ = out72_
                                    d_86_c2_ = out73_
                                    generated = d_84_g2_
                                    insideConstrainedOut = d_85_i2_
                                    currentConstrainedOut = d_86_c2_
                                    d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_87_gFinal_: _dafny.Seq
            d_88_cFinal_: _dafny.Seq
            out74_: _dafny.Seq
            out75_: _dafny.Seq
            out74_, out75_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_87_gFinal_ = out74_
            d_88_cFinal_ = out75_
            generated = d_87_gFinal_
            currentConstrainedOut = d_88_cFinal_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_89_constrainedPrompt_: _dafny.Seq
                d_89_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_90_next_: _dafny.Seq
                out76_: _dafny.Seq
                out76_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_89_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_90_next_ = out76_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_90_next_) != (eosToken):
                    d_91_g2_: _dafny.Seq
                    d_92_i2_: bool
                    d_93_c2_: _dafny.Seq
                    out77_: _dafny.Seq
                    out78_: bool
                    out79_: _dafny.Seq
                    out77_, out78_, out79_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_90_next_)
                    d_91_g2_ = out77_
                    d_92_i2_ = out78_
                    d_93_c2_ = out79_
                    generated = d_91_g2_
                    insideConstrainedOut = d_92_i2_
                    currentConstrainedOut = d_93_c2_
                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                        d_94_gF2_: _dafny.Seq
                        d_95_cF2_: _dafny.Seq
                        out80_: _dafny.Seq
                        out81_: _dafny.Seq
                        out80_, out81_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_94_gF2_ = out80_
                        d_95_cF2_ = out81_
                        generated = d_94_gF2_
                        currentConstrainedOut = d_95_cF2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_96_g2_: _dafny.Seq
                d_97_i2_: bool
                d_98_c2_: _dafny.Seq
                out82_: _dafny.Seq
                out83_: bool
                out84_: _dafny.Seq
                out82_, out83_, out84_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_96_g2_ = out82_
                d_97_i2_ = out83_
                d_98_c2_ = out84_
                generated = d_96_g2_
                insideConstrainedOut = d_97_i2_
                currentConstrainedOut = d_98_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        if ((maxSteps) > (0)) and ((d_1_steps_) == (0)):
            cost = 1
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

