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
                            d_25_gR1_: _dafny.Seq
                            d_26_cR1_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: _dafny.Seq
                            out17_, out18_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_25_gR1_ = out17_
                            d_26_cR1_ = out18_
                            generated = d_25_gR1_
                            currentConstrainedOut = d_26_cR1_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_27_gR2_: _dafny.Seq
                                d_28_cR2_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: _dafny.Seq
                                out19_, out20_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_27_gR2_ = out19_
                                d_28_cR2_ = out20_
                                generated = d_27_gR2_
                                currentConstrainedOut = d_28_cR2_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_29_gR3_: _dafny.Seq
                                d_30_cR3_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: _dafny.Seq
                                out21_, out22_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_29_gR3_ = out21_
                                d_30_cR3_ = out22_
                                generated = d_29_gR3_
                                currentConstrainedOut = d_30_cR3_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_31_gR4_: _dafny.Seq
                                d_32_cR4_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: _dafny.Seq
                                out23_, out24_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_31_gR4_ = out23_
                                d_32_cR4_ = out24_
                                generated = d_31_gR4_
                                currentConstrainedOut = d_32_cR4_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_33_gR5_: _dafny.Seq
                                d_34_cR5_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: _dafny.Seq
                                out25_, out26_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_33_gR5_ = out25_
                                d_34_cR5_ = out26_
                                generated = d_33_gR5_
                                currentConstrainedOut = d_34_cR5_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_35_gR6_: _dafny.Seq
                                d_36_cR6_: _dafny.Seq
                                out27_: _dafny.Seq
                                out28_: _dafny.Seq
                                out27_, out28_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_35_gR6_ = out27_
                                d_36_cR6_ = out28_
                                generated = d_35_gR6_
                                currentConstrainedOut = d_36_cR6_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_37_g2_: _dafny.Seq
                                d_38_i2_: bool
                                d_39_c2_: _dafny.Seq
                                out29_: _dafny.Seq
                                out30_: bool
                                out31_: _dafny.Seq
                                out29_, out30_, out31_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_37_g2_ = out29_
                                d_38_i2_ = out30_
                                d_39_c2_ = out31_
                                generated = d_37_g2_
                                insideConstrainedOut = d_38_i2_
                                currentConstrainedOut = d_39_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif (d_1_steps_) < (maxSteps):
                                d_40_constrainedPrompt_: _dafny.Seq
                                d_40_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_41_next_: _dafny.Seq
                                out32_: _dafny.Seq
                                out32_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_40_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_41_next_ = out32_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_41_next_) == (eosToken):
                                    d_42_gFinal_: _dafny.Seq
                                    d_43_cFinal_: _dafny.Seq
                                    out33_: _dafny.Seq
                                    out34_: _dafny.Seq
                                    out33_, out34_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_42_gFinal_ = out33_
                                    d_43_cFinal_ = out34_
                                    generated = d_42_gFinal_
                                    currentConstrainedOut = d_43_cFinal_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_44_g3_: _dafny.Seq
                                        d_45_i3_: bool
                                        d_46_c3_: _dafny.Seq
                                        out35_: _dafny.Seq
                                        out36_: bool
                                        out37_: _dafny.Seq
                                        out35_, out36_, out37_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_44_g3_ = out35_
                                        d_45_i3_ = out36_
                                        d_46_c3_ = out37_
                                        generated = d_44_g3_
                                        insideConstrainedOut = d_45_i3_
                                        currentConstrainedOut = d_46_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                                elif True:
                                    d_47_g2_: _dafny.Seq
                                    d_48_i2_: bool
                                    d_49_c2_: _dafny.Seq
                                    out38_: _dafny.Seq
                                    out39_: bool
                                    out40_: _dafny.Seq
                                    out38_, out39_, out40_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_41_next_)
                                    d_47_g2_ = out38_
                                    d_48_i2_ = out39_
                                    d_49_c2_ = out40_
                                    generated = d_47_g2_
                                    insideConstrainedOut = d_48_i2_
                                    currentConstrainedOut = d_49_c2_
                                    d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_50_g3_: _dafny.Seq
                                        d_51_i3_: bool
                                        d_52_c3_: _dafny.Seq
                                        out41_: _dafny.Seq
                                        out42_: bool
                                        out43_: _dafny.Seq
                                        out41_, out42_, out43_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_50_g3_ = out41_
                                        d_51_i3_ = out42_
                                        d_52_c3_ = out43_
                                        generated = d_50_g3_
                                        insideConstrainedOut = d_51_i3_
                                        currentConstrainedOut = d_52_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                        elif True:
                            d_53_constrainedPrompt_: _dafny.Seq
                            d_53_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_54_next_: _dafny.Seq
                            d_55_wasConstrained_: bool
                            out44_: _dafny.Seq
                            out45_: bool
                            out44_, out45_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_53_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_54_next_ = out44_
                            d_55_wasConstrained_ = out45_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_54_next_) == (eosToken):
                                d_56_gR1_: _dafny.Seq
                                d_57_cR1_: _dafny.Seq
                                out46_: _dafny.Seq
                                out47_: _dafny.Seq
                                out46_, out47_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_56_gR1_ = out46_
                                d_57_cR1_ = out47_
                                generated = d_56_gR1_
                                currentConstrainedOut = d_57_cR1_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_58_gR2_: _dafny.Seq
                                    d_59_cR2_: _dafny.Seq
                                    out48_: _dafny.Seq
                                    out49_: _dafny.Seq
                                    out48_, out49_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_58_gR2_ = out48_
                                    d_59_cR2_ = out49_
                                    generated = d_58_gR2_
                                    currentConstrainedOut = d_59_cR2_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_60_gR3_: _dafny.Seq
                                    d_61_cR3_: _dafny.Seq
                                    out50_: _dafny.Seq
                                    out51_: _dafny.Seq
                                    out50_, out51_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_60_gR3_ = out50_
                                    d_61_cR3_ = out51_
                                    generated = d_60_gR3_
                                    currentConstrainedOut = d_61_cR3_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_62_gR4_: _dafny.Seq
                                    d_63_cR4_: _dafny.Seq
                                    out52_: _dafny.Seq
                                    out53_: _dafny.Seq
                                    out52_, out53_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_62_gR4_ = out52_
                                    d_63_cR4_ = out53_
                                    generated = d_62_gR4_
                                    currentConstrainedOut = d_63_cR4_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_64_gR5_: _dafny.Seq
                                    d_65_cR5_: _dafny.Seq
                                    out54_: _dafny.Seq
                                    out55_: _dafny.Seq
                                    out54_, out55_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_64_gR5_ = out54_
                                    d_65_cR5_ = out55_
                                    generated = d_64_gR5_
                                    currentConstrainedOut = d_65_cR5_
                                if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_66_gFinal_: _dafny.Seq
                                    d_67_cFinal_: _dafny.Seq
                                    out56_: _dafny.Seq
                                    out57_: _dafny.Seq
                                    out56_, out57_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_66_gFinal_ = out56_
                                    d_67_cFinal_ = out57_
                                    generated = d_66_gFinal_
                                    currentConstrainedOut = d_67_cFinal_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_68_g2_: _dafny.Seq
                                    d_69_i2_: bool
                                    d_70_c2_: _dafny.Seq
                                    out58_: _dafny.Seq
                                    out59_: bool
                                    out60_: _dafny.Seq
                                    out58_, out59_, out60_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_68_g2_ = out58_
                                    d_69_i2_ = out59_
                                    d_70_c2_ = out60_
                                    generated = d_68_g2_
                                    insideConstrainedOut = d_69_i2_
                                    currentConstrainedOut = d_70_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_71_g2_: _dafny.Seq
                                d_72_i2_: bool
                                d_73_c2_: _dafny.Seq
                                out61_: _dafny.Seq
                                out62_: bool
                                out63_: _dafny.Seq
                                out61_, out62_, out63_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_54_next_)
                                d_71_g2_ = out61_
                                d_72_i2_ = out62_
                                d_73_c2_ = out63_
                                generated = d_71_g2_
                                insideConstrainedOut = d_72_i2_
                                currentConstrainedOut = d_73_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_74_gFinal_: _dafny.Seq
            d_75_cFinal_: _dafny.Seq
            out64_: _dafny.Seq
            out65_: _dafny.Seq
            out64_, out65_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_74_gFinal_ = out64_
            d_75_cFinal_ = out65_
            generated = d_74_gFinal_
            currentConstrainedOut = d_75_cFinal_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_76_constrainedPrompt_: _dafny.Seq
                d_76_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_77_next_: _dafny.Seq
                out66_: _dafny.Seq
                out66_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_76_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_77_next_ = out66_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_77_next_) != (eosToken):
                    d_78_g2_: _dafny.Seq
                    d_79_i2_: bool
                    d_80_c2_: _dafny.Seq
                    out67_: _dafny.Seq
                    out68_: bool
                    out69_: _dafny.Seq
                    out67_, out68_, out69_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_77_next_)
                    d_78_g2_ = out67_
                    d_79_i2_ = out68_
                    d_80_c2_ = out69_
                    generated = d_78_g2_
                    insideConstrainedOut = d_79_i2_
                    currentConstrainedOut = d_80_c2_
                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                        d_81_gF2_: _dafny.Seq
                        d_82_cF2_: _dafny.Seq
                        out70_: _dafny.Seq
                        out71_: _dafny.Seq
                        out70_, out71_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_81_gF2_ = out70_
                        d_82_cF2_ = out71_
                        generated = d_81_gF2_
                        currentConstrainedOut = d_82_cF2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_83_g2_: _dafny.Seq
                d_84_i2_: bool
                d_85_c2_: _dafny.Seq
                out72_: _dafny.Seq
                out73_: bool
                out74_: _dafny.Seq
                out72_, out73_, out74_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_83_g2_ = out72_
                d_84_i2_ = out73_
                d_85_c2_ = out74_
                generated = d_83_g2_
                insideConstrainedOut = d_84_i2_
                currentConstrainedOut = d_85_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

