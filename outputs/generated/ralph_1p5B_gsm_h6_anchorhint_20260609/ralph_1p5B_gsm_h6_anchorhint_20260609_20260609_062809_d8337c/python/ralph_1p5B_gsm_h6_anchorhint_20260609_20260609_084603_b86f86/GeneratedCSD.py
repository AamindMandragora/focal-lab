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
        d_4_spanMaxTokens_ = 10
        d_5_hasSeenOpenSpan_: bool
        d_5_hasSeenOpenSpan_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remaining_: int
                        d_6_remaining_ = (maxSteps) - (d_1_steps_)
                        if (((d_6_remaining_) <= (65)) and (not(d_5_hasSeenOpenSpan_))) and ((d_6_remaining_) > (3)):
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
                    elif True:
                        d_21_remaining_: int
                        d_21_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_21_remaining_) <= (10):
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_22_gFinal_: _dafny.Seq
                            d_23_cFinal_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: _dafny.Seq
                            out13_, out14_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_22_gFinal_ = out13_
                            d_23_cFinal_ = out14_
                            generated = d_22_gFinal_
                            currentConstrainedOut = d_23_cFinal_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_24_g2_: _dafny.Seq
                                d_25_i2_: bool
                                d_26_c2_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_24_g2_ = out15_
                                d_25_i2_ = out16_
                                d_26_c2_ = out17_
                                generated = d_24_g2_
                                insideConstrainedOut = d_25_i2_
                                currentConstrainedOut = d_26_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif (parser).IsCompletePrefix(currentConstrainedOut):
                            d_27_g2_: _dafny.Seq
                            d_28_i2_: bool
                            d_29_c2_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: bool
                            out20_: _dafny.Seq
                            out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_27_g2_ = out18_
                            d_28_i2_ = out19_
                            d_29_c2_ = out20_
                            generated = d_27_g2_
                            insideConstrainedOut = d_28_i2_
                            currentConstrainedOut = d_29_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                        elif True:
                            d_30_isDeadEnd_: bool
                            out21_: bool
                            out21_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_30_isDeadEnd_ = out21_
                            if (d_30_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_)):
                                d_31_gR1_: _dafny.Seq
                                d_32_cR1_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: _dafny.Seq
                                out22_, out23_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_31_gR1_ = out22_
                                d_32_cR1_ = out23_
                                generated = d_31_gR1_
                                currentConstrainedOut = d_32_cR1_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_33_gR2_: _dafny.Seq
                                    d_34_cR2_: _dafny.Seq
                                    out24_: _dafny.Seq
                                    out25_: _dafny.Seq
                                    out24_, out25_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_33_gR2_ = out24_
                                    d_34_cR2_ = out25_
                                    generated = d_33_gR2_
                                    currentConstrainedOut = d_34_cR2_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_35_gR3_: _dafny.Seq
                                    d_36_cR3_: _dafny.Seq
                                    out26_: _dafny.Seq
                                    out27_: _dafny.Seq
                                    out26_, out27_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_35_gR3_ = out26_
                                    d_36_cR3_ = out27_
                                    generated = d_35_gR3_
                                    currentConstrainedOut = d_36_cR3_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_37_gR4_: _dafny.Seq
                                    d_38_cR4_: _dafny.Seq
                                    out28_: _dafny.Seq
                                    out29_: _dafny.Seq
                                    out28_, out29_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_37_gR4_ = out28_
                                    d_38_cR4_ = out29_
                                    generated = d_37_gR4_
                                    currentConstrainedOut = d_38_cR4_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_39_gR5_: _dafny.Seq
                                    d_40_cR5_: _dafny.Seq
                                    out30_: _dafny.Seq
                                    out31_: _dafny.Seq
                                    out30_, out31_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_39_gR5_ = out30_
                                    d_40_cR5_ = out31_
                                    generated = d_39_gR5_
                                    currentConstrainedOut = d_40_cR5_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_41_gR6_: _dafny.Seq
                                    d_42_cR6_: _dafny.Seq
                                    out32_: _dafny.Seq
                                    out33_: _dafny.Seq
                                    out32_, out33_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_41_gR6_ = out32_
                                    d_42_cR6_ = out33_
                                    generated = d_41_gR6_
                                    currentConstrainedOut = d_42_cR6_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_43_g2_: _dafny.Seq
                                    d_44_i2_: bool
                                    d_45_c2_: _dafny.Seq
                                    out34_: _dafny.Seq
                                    out35_: bool
                                    out36_: _dafny.Seq
                                    out34_, out35_, out36_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_43_g2_ = out34_
                                    d_44_i2_ = out35_
                                    d_45_c2_ = out36_
                                    generated = d_43_g2_
                                    insideConstrainedOut = d_44_i2_
                                    currentConstrainedOut = d_45_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif (d_1_steps_) < (maxSteps):
                                    d_46_constrainedPrompt_: _dafny.Seq
                                    d_46_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_47_next_: _dafny.Seq
                                    out37_: _dafny.Seq
                                    out37_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_46_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_47_next_ = out37_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_47_next_) == (eosToken):
                                        d_48_gF2_: _dafny.Seq
                                        d_49_cF2_: _dafny.Seq
                                        out38_: _dafny.Seq
                                        out39_: _dafny.Seq
                                        out38_, out39_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                        d_48_gF2_ = out38_
                                        d_49_cF2_ = out39_
                                        generated = d_48_gF2_
                                        currentConstrainedOut = d_49_cF2_
                                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                            d_50_g3_: _dafny.Seq
                                            d_51_i3_: bool
                                            d_52_c3_: _dafny.Seq
                                            out40_: _dafny.Seq
                                            out41_: bool
                                            out42_: _dafny.Seq
                                            out40_, out41_, out42_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_50_g3_ = out40_
                                            d_51_i3_ = out41_
                                            d_52_c3_ = out42_
                                            generated = d_50_g3_
                                            insideConstrainedOut = d_51_i3_
                                            currentConstrainedOut = d_52_c3_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            d_3_spanTokensUsed_ = 0
                                    elif True:
                                        d_53_g2_: _dafny.Seq
                                        d_54_i2_: bool
                                        d_55_c2_: _dafny.Seq
                                        out43_: _dafny.Seq
                                        out44_: bool
                                        out45_: _dafny.Seq
                                        out43_, out44_, out45_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_47_next_)
                                        d_53_g2_ = out43_
                                        d_54_i2_ = out44_
                                        d_55_c2_ = out45_
                                        generated = d_53_g2_
                                        insideConstrainedOut = d_54_i2_
                                        currentConstrainedOut = d_55_c2_
                                        d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                            d_56_g3_: _dafny.Seq
                                            d_57_i3_: bool
                                            d_58_c3_: _dafny.Seq
                                            out46_: _dafny.Seq
                                            out47_: bool
                                            out48_: _dafny.Seq
                                            out46_, out47_, out48_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_56_g3_ = out46_
                                            d_57_i3_ = out47_
                                            d_58_c3_ = out48_
                                            generated = d_56_g3_
                                            insideConstrainedOut = d_57_i3_
                                            currentConstrainedOut = d_58_c3_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            d_3_spanTokensUsed_ = 0
                            elif True:
                                d_59_constrainedPrompt_: _dafny.Seq
                                d_59_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_60_next_: _dafny.Seq
                                out49_: _dafny.Seq
                                out49_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_59_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_60_next_ = out49_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_60_next_) == (eosToken):
                                    d_61_gR1_: _dafny.Seq
                                    d_62_cR1_: _dafny.Seq
                                    out50_: _dafny.Seq
                                    out51_: _dafny.Seq
                                    out50_, out51_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_61_gR1_ = out50_
                                    d_62_cR1_ = out51_
                                    generated = d_61_gR1_
                                    currentConstrainedOut = d_62_cR1_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_63_gR2_: _dafny.Seq
                                        d_64_cR2_: _dafny.Seq
                                        out52_: _dafny.Seq
                                        out53_: _dafny.Seq
                                        out52_, out53_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_63_gR2_ = out52_
                                        d_64_cR2_ = out53_
                                        generated = d_63_gR2_
                                        currentConstrainedOut = d_64_cR2_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_65_gR3_: _dafny.Seq
                                        d_66_cR3_: _dafny.Seq
                                        out54_: _dafny.Seq
                                        out55_: _dafny.Seq
                                        out54_, out55_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_65_gR3_ = out54_
                                        d_66_cR3_ = out55_
                                        generated = d_65_gR3_
                                        currentConstrainedOut = d_66_cR3_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_67_gR4_: _dafny.Seq
                                        d_68_cR4_: _dafny.Seq
                                        out56_: _dafny.Seq
                                        out57_: _dafny.Seq
                                        out56_, out57_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_67_gR4_ = out56_
                                        d_68_cR4_ = out57_
                                        generated = d_67_gR4_
                                        currentConstrainedOut = d_68_cR4_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_69_gR5_: _dafny.Seq
                                        d_70_cR5_: _dafny.Seq
                                        out58_: _dafny.Seq
                                        out59_: _dafny.Seq
                                        out58_, out59_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_69_gR5_ = out58_
                                        d_70_cR5_ = out59_
                                        generated = d_69_gR5_
                                        currentConstrainedOut = d_70_cR5_
                                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                        d_71_gFinal_: _dafny.Seq
                                        d_72_cFinal_: _dafny.Seq
                                        out60_: _dafny.Seq
                                        out61_: _dafny.Seq
                                        out60_, out61_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                        d_71_gFinal_ = out60_
                                        d_72_cFinal_ = out61_
                                        generated = d_71_gFinal_
                                        currentConstrainedOut = d_72_cFinal_
                                    d_3_spanTokensUsed_ = 0
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_73_g2_: _dafny.Seq
                                        d_74_i2_: bool
                                        d_75_c2_: _dafny.Seq
                                        out62_: _dafny.Seq
                                        out63_: bool
                                        out64_: _dafny.Seq
                                        out62_, out63_, out64_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_73_g2_ = out62_
                                        d_74_i2_ = out63_
                                        d_75_c2_ = out64_
                                        generated = d_73_g2_
                                        insideConstrainedOut = d_74_i2_
                                        currentConstrainedOut = d_75_c2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_76_g2_: _dafny.Seq
                                    d_77_i2_: bool
                                    d_78_c2_: _dafny.Seq
                                    out65_: _dafny.Seq
                                    out66_: bool
                                    out67_: _dafny.Seq
                                    out65_, out66_, out67_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_60_next_)
                                    d_76_g2_ = out65_
                                    d_77_i2_ = out66_
                                    d_78_c2_ = out67_
                                    generated = d_76_g2_
                                    insideConstrainedOut = d_77_i2_
                                    currentConstrainedOut = d_78_c2_
                                    d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_79_gFinal_: _dafny.Seq
            d_80_cFinal_: _dafny.Seq
            out68_: _dafny.Seq
            out69_: _dafny.Seq
            out68_, out69_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_79_gFinal_ = out68_
            d_80_cFinal_ = out69_
            generated = d_79_gFinal_
            currentConstrainedOut = d_80_cFinal_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (2)) < (maxSteps)):
                d_81_constrainedPrompt_: _dafny.Seq
                d_81_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_82_next_: _dafny.Seq
                out70_: _dafny.Seq
                out70_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_81_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_82_next_ = out70_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_82_next_) != (eosToken):
                    d_83_g2_: _dafny.Seq
                    d_84_i2_: bool
                    d_85_c2_: _dafny.Seq
                    out71_: _dafny.Seq
                    out72_: bool
                    out73_: _dafny.Seq
                    out71_, out72_, out73_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_82_next_)
                    d_83_g2_ = out71_
                    d_84_i2_ = out72_
                    d_85_c2_ = out73_
                    generated = d_83_g2_
                    insideConstrainedOut = d_84_i2_
                    currentConstrainedOut = d_85_c2_
                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                        d_86_gF2_: _dafny.Seq
                        d_87_cF2_: _dafny.Seq
                        out74_: _dafny.Seq
                        out75_: _dafny.Seq
                        out74_, out75_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_86_gF2_ = out74_
                        d_87_cF2_ = out75_
                        generated = d_86_gF2_
                        currentConstrainedOut = d_87_cF2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_88_g2_: _dafny.Seq
                d_89_i2_: bool
                d_90_c2_: _dafny.Seq
                out76_: _dafny.Seq
                out77_: bool
                out78_: _dafny.Seq
                out76_, out77_, out78_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_88_g2_ = out76_
                d_89_i2_ = out77_
                d_90_c2_ = out78_
                generated = d_88_g2_
                insideConstrainedOut = d_89_i2_
                currentConstrainedOut = d_90_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

