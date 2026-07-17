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
        d_6_totalSpanSteps_: int
        d_6_totalSpanSteps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (insideConstrainedOut) and (((maxSteps) - (d_1_steps_)) <= (4)):
                        d_7_gR_: _dafny.Seq
                        d_8_cR_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: _dafny.Seq
                        out0_, out1_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_7_gR_ = out0_
                        d_8_cR_ = out1_
                        generated = d_7_gR_
                        currentConstrainedOut = d_8_cR_
                        if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((maxSteps) - (d_1_steps_)) >= (1)):
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_10_next_: _dafny.Seq
                            out2_: _dafny.Seq
                            out2_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_10_next_ = out2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) != (eosToken):
                                d_11_g2_: _dafny.Seq
                                d_12_i2_: bool
                                d_13_c2_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                d_11_g2_ = out3_
                                d_12_i2_ = out4_
                                d_13_c2_ = out5_
                                generated = d_11_g2_
                                insideConstrainedOut = d_12_i2_
                                currentConstrainedOut = d_13_c2_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((maxSteps) - (d_1_steps_)) >= (1)):
                            d_14_g2_: _dafny.Seq
                            d_15_i2_: bool
                            d_16_c2_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_14_g2_ = out6_
                            d_15_i2_ = out7_
                            d_16_c2_ = out8_
                            generated = d_14_g2_
                            insideConstrainedOut = d_15_i2_
                            currentConstrainedOut = d_16_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif (insideConstrainedOut) and ((d_6_totalSpanSteps_) >= (50)):
                        d_17_gR_: _dafny.Seq
                        d_18_cR_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: _dafny.Seq
                        out9_, out10_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_17_gR_ = out9_
                        d_18_cR_ = out10_
                        generated = d_17_gR_
                        currentConstrainedOut = d_18_cR_
                        d_6_totalSpanSteps_ = 0
                        d_3_spanTokensUsed_ = 0
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_19_g2_: _dafny.Seq
                            d_20_i2_: bool
                            d_21_c2_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_g2_ = out11_
                            d_20_i2_ = out12_
                            d_21_c2_ = out13_
                            generated = d_19_g2_
                            insideConstrainedOut = d_20_i2_
                            currentConstrainedOut = d_21_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_23_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_23_next_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_23_next_) != (eosToken):
                                d_24_g2_: _dafny.Seq
                                d_25_i2_: bool
                                d_26_c2_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_24_g2_ = out15_
                                d_25_i2_ = out16_
                                d_26_c2_ = out17_
                                generated = d_24_g2_
                                insideConstrainedOut = d_25_i2_
                                currentConstrainedOut = d_26_c2_
                                d_6_totalSpanSteps_ = 1
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_27_g3_: _dafny.Seq
                                    d_28_i3_: bool
                                    d_29_c3_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_27_g3_ = out18_
                                    d_28_i3_ = out19_
                                    d_29_c3_ = out20_
                                    generated = d_27_g3_
                                    insideConstrainedOut = d_28_i3_
                                    currentConstrainedOut = d_29_c3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_6_totalSpanSteps_ = 0
                                    d_3_spanTokensUsed_ = 0
                    elif not(insideConstrainedOut):
                        d_30_remaining_: int
                        d_30_remaining_ = (maxSteps) - (d_1_steps_)
                        if (((d_30_remaining_) <= (65)) and (not(d_5_hasSeenOpenSpan_))) and ((d_30_remaining_) > (2)):
                            d_31_g2_: _dafny.Seq
                            d_32_i2_: bool
                            d_33_c2_: _dafny.Seq
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: _dafny.Seq
                            out21_, out22_, out23_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_31_g2_ = out21_
                            d_32_i2_ = out22_
                            d_33_c2_ = out23_
                            generated = d_31_g2_
                            insideConstrainedOut = d_32_i2_
                            currentConstrainedOut = d_33_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                            d_6_totalSpanSteps_ = 0
                            d_5_hasSeenOpenSpan_ = True
                        elif True:
                            d_34_chunkBudget_: int
                            if (d_30_remaining_) < (d_2_freeChunkSize_):
                                d_34_chunkBudget_ = d_30_remaining_
                            elif True:
                                d_34_chunkBudget_ = d_2_freeChunkSize_
                            if (d_34_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            d_35_chunkGenerated_: _dafny.Seq
                            d_36_stoppedOnOpenSpan_: bool
                            d_37_stoppedOnEos_: bool
                            d_38_stepsUsed_: int
                            out24_: _dafny.Seq
                            out25_: bool
                            out26_: bool
                            out27_: int
                            out24_, out25_, out26_, out27_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_34_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_35_chunkGenerated_ = out24_
                            d_36_stoppedOnOpenSpan_ = out25_
                            d_37_stoppedOnEos_ = out26_
                            d_38_stepsUsed_ = out27_
                            generated = d_35_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_38_stepsUsed_)
                            if d_37_stoppedOnEos_:
                                if (not(d_5_hasSeenOpenSpan_)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                                    d_39_g2_: _dafny.Seq
                                    d_40_i2_: bool
                                    d_41_c2_: _dafny.Seq
                                    out28_: _dafny.Seq
                                    out29_: bool
                                    out30_: _dafny.Seq
                                    out28_, out29_, out30_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_39_g2_ = out28_
                                    d_40_i2_ = out29_
                                    d_41_c2_ = out30_
                                    generated = d_39_g2_
                                    insideConstrainedOut = d_40_i2_
                                    currentConstrainedOut = d_41_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                                    d_6_totalSpanSteps_ = 0
                                    d_5_hasSeenOpenSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif d_36_stoppedOnOpenSpan_:
                                d_42_g2_: _dafny.Seq
                                d_43_i2_: bool
                                d_44_c2_: _dafny.Seq
                                out31_: _dafny.Seq
                                out32_: bool
                                out33_: _dafny.Seq
                                out31_, out32_, out33_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_42_g2_ = out31_
                                d_43_i2_ = out32_
                                d_44_c2_ = out33_
                                generated = d_42_g2_
                                insideConstrainedOut = d_43_i2_
                                currentConstrainedOut = d_44_c2_
                                d_3_spanTokensUsed_ = 0
                                d_6_totalSpanSteps_ = 0
                                d_5_hasSeenOpenSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
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
                        d_3_spanTokensUsed_ = 0
                        d_6_totalSpanSteps_ = 0
                    elif True:
                        d_48_isDeadEnd_: bool
                        out37_: bool
                        out37_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_48_isDeadEnd_ = out37_
                        if (d_48_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_)):
                            d_49_gRolled_: _dafny.Seq
                            d_50_cRolled_: _dafny.Seq
                            out38_: _dafny.Seq
                            out39_: _dafny.Seq
                            out38_, out39_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_49_gRolled_ = out38_
                            d_50_cRolled_ = out39_
                            generated = d_49_gRolled_
                            currentConstrainedOut = d_50_cRolled_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_51_g3_: _dafny.Seq
                                d_52_c3_: _dafny.Seq
                                out40_: _dafny.Seq
                                out41_: _dafny.Seq
                                out40_, out41_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_51_g3_ = out40_
                                d_52_c3_ = out41_
                                generated = d_51_g3_
                                currentConstrainedOut = d_52_c3_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_53_g4_: _dafny.Seq
                                d_54_c4_: _dafny.Seq
                                out42_: _dafny.Seq
                                out43_: _dafny.Seq
                                out42_, out43_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_53_g4_ = out42_
                                d_54_c4_ = out43_
                                generated = d_53_g4_
                                currentConstrainedOut = d_54_c4_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_55_g5_: _dafny.Seq
                                d_56_c5_: _dafny.Seq
                                out44_: _dafny.Seq
                                out45_: _dafny.Seq
                                out44_, out45_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_55_g5_ = out44_
                                d_56_c5_ = out45_
                                generated = d_55_g5_
                                currentConstrainedOut = d_56_c5_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_57_g6_: _dafny.Seq
                                d_58_c6_: _dafny.Seq
                                out46_: _dafny.Seq
                                out47_: _dafny.Seq
                                out46_, out47_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_57_g6_ = out46_
                                d_58_c6_ = out47_
                                generated = d_57_g6_
                                currentConstrainedOut = d_58_c6_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_59_g7_: _dafny.Seq
                                d_60_c7_: _dafny.Seq
                                out48_: _dafny.Seq
                                out49_: _dafny.Seq
                                out48_, out49_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_59_g7_ = out48_
                                d_60_c7_ = out49_
                                generated = d_59_g7_
                                currentConstrainedOut = d_60_c7_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_61_g2_: _dafny.Seq
                                d_62_i2_: bool
                                d_63_c2_: _dafny.Seq
                                out50_: _dafny.Seq
                                out51_: bool
                                out52_: _dafny.Seq
                                out50_, out51_, out52_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_61_g2_ = out50_
                                d_62_i2_ = out51_
                                d_63_c2_ = out52_
                                generated = d_61_g2_
                                insideConstrainedOut = d_62_i2_
                                currentConstrainedOut = d_63_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_6_totalSpanSteps_ = (d_6_totalSpanSteps_) + (1)
                            elif (d_1_steps_) < (maxSteps):
                                d_64_constrainedPrompt_: _dafny.Seq
                                d_64_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_65_next_: _dafny.Seq
                                out53_: _dafny.Seq
                                out53_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_64_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_65_next_ = out53_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_6_totalSpanSteps_ = (d_6_totalSpanSteps_) + (1)
                                if (d_65_next_) == (eosToken):
                                    d_66_gR2_: _dafny.Seq
                                    d_67_cR2_: _dafny.Seq
                                    out54_: _dafny.Seq
                                    out55_: _dafny.Seq
                                    out54_, out55_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_66_gR2_ = out54_
                                    d_67_cR2_ = out55_
                                    generated = d_66_gR2_
                                    currentConstrainedOut = d_67_cR2_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_68_gR3_: _dafny.Seq
                                        d_69_cR3_: _dafny.Seq
                                        out56_: _dafny.Seq
                                        out57_: _dafny.Seq
                                        out56_, out57_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_68_gR3_ = out56_
                                        d_69_cR3_ = out57_
                                        generated = d_68_gR3_
                                        currentConstrainedOut = d_69_cR3_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_70_gR4_: _dafny.Seq
                                        d_71_cR4_: _dafny.Seq
                                        out58_: _dafny.Seq
                                        out59_: _dafny.Seq
                                        out58_, out59_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_70_gR4_ = out58_
                                        d_71_cR4_ = out59_
                                        generated = d_70_gR4_
                                        currentConstrainedOut = d_71_cR4_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_72_g2_: _dafny.Seq
                                        d_73_i2_: bool
                                        d_74_c2_: _dafny.Seq
                                        out60_: _dafny.Seq
                                        out61_: bool
                                        out62_: _dafny.Seq
                                        out60_, out61_, out62_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_72_g2_ = out60_
                                        d_73_i2_ = out61_
                                        d_74_c2_ = out62_
                                        generated = d_72_g2_
                                        insideConstrainedOut = d_73_i2_
                                        currentConstrainedOut = d_74_c2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_6_totalSpanSteps_ = (d_6_totalSpanSteps_) + (1)
                                elif True:
                                    d_75_g2_: _dafny.Seq
                                    d_76_i2_: bool
                                    d_77_c2_: _dafny.Seq
                                    out63_: _dafny.Seq
                                    out64_: bool
                                    out65_: _dafny.Seq
                                    out63_, out64_, out65_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_65_next_)
                                    d_75_g2_ = out63_
                                    d_76_i2_ = out64_
                                    d_77_c2_ = out65_
                                    generated = d_75_g2_
                                    insideConstrainedOut = d_76_i2_
                                    currentConstrainedOut = d_77_c2_
                                    d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_78_g3_: _dafny.Seq
                                        d_79_i3_: bool
                                        d_80_c3_: _dafny.Seq
                                        out66_: _dafny.Seq
                                        out67_: bool
                                        out68_: _dafny.Seq
                                        out66_, out67_, out68_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_78_g3_ = out66_
                                        d_79_i3_ = out67_
                                        d_80_c3_ = out68_
                                        generated = d_78_g3_
                                        insideConstrainedOut = d_79_i3_
                                        currentConstrainedOut = d_80_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_6_totalSpanSteps_ = (d_6_totalSpanSteps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                        elif True:
                            d_81_constrainedPrompt_: _dafny.Seq
                            d_81_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_82_next_: _dafny.Seq
                            d_83_wasConstrained_: bool
                            out69_: _dafny.Seq
                            out70_: bool
                            out69_, out70_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_81_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_82_next_ = out69_
                            d_83_wasConstrained_ = out70_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_6_totalSpanSteps_ = (d_6_totalSpanSteps_) + (1)
                            if (d_82_next_) == (eosToken):
                                d_84_gRolled_: _dafny.Seq
                                d_85_cRolled_: _dafny.Seq
                                out71_: _dafny.Seq
                                out72_: _dafny.Seq
                                out71_, out72_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_84_gRolled_ = out71_
                                d_85_cRolled_ = out72_
                                generated = d_84_gRolled_
                                currentConstrainedOut = d_85_cRolled_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_86_gR2_: _dafny.Seq
                                    d_87_cR2_: _dafny.Seq
                                    out73_: _dafny.Seq
                                    out74_: _dafny.Seq
                                    out73_, out74_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_86_gR2_ = out73_
                                    d_87_cR2_ = out74_
                                    generated = d_86_gR2_
                                    currentConstrainedOut = d_87_cR2_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_88_gR3_: _dafny.Seq
                                    d_89_cR3_: _dafny.Seq
                                    out75_: _dafny.Seq
                                    out76_: _dafny.Seq
                                    out75_, out76_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_88_gR3_ = out75_
                                    d_89_cR3_ = out76_
                                    generated = d_88_gR3_
                                    currentConstrainedOut = d_89_cR3_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_90_gR4_: _dafny.Seq
                                    d_91_cR4_: _dafny.Seq
                                    out77_: _dafny.Seq
                                    out78_: _dafny.Seq
                                    out77_, out78_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_90_gR4_ = out77_
                                    d_91_cR4_ = out78_
                                    generated = d_90_gR4_
                                    currentConstrainedOut = d_91_cR4_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_92_gR5_: _dafny.Seq
                                    d_93_cR5_: _dafny.Seq
                                    out79_: _dafny.Seq
                                    out80_: _dafny.Seq
                                    out79_, out80_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_92_gR5_ = out79_
                                    d_93_cR5_ = out80_
                                    generated = d_92_gR5_
                                    currentConstrainedOut = d_93_cR5_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_94_g2_: _dafny.Seq
                                    d_95_i2_: bool
                                    d_96_c2_: _dafny.Seq
                                    out81_: _dafny.Seq
                                    out82_: bool
                                    out83_: _dafny.Seq
                                    out81_, out82_, out83_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_94_g2_ = out81_
                                    d_95_i2_ = out82_
                                    d_96_c2_ = out83_
                                    generated = d_94_g2_
                                    insideConstrainedOut = d_95_i2_
                                    currentConstrainedOut = d_96_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_6_totalSpanSteps_ = (d_6_totalSpanSteps_) + (1)
                            elif True:
                                d_97_g2_: _dafny.Seq
                                d_98_i2_: bool
                                d_99_c2_: _dafny.Seq
                                out84_: _dafny.Seq
                                out85_: bool
                                out86_: _dafny.Seq
                                out84_, out85_, out86_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_82_next_)
                                d_97_g2_ = out84_
                                d_98_i2_ = out85_
                                d_99_c2_ = out86_
                                generated = d_97_g2_
                                insideConstrainedOut = d_98_i2_
                                currentConstrainedOut = d_99_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_100_gR_: _dafny.Seq
            d_101_cR_: _dafny.Seq
            out87_: _dafny.Seq
            out88_: _dafny.Seq
            out87_, out88_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_100_gR_ = out87_
            d_101_cR_ = out88_
            generated = d_100_gR_
            currentConstrainedOut = d_101_cR_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_102_constrainedPrompt_: _dafny.Seq
                d_102_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_103_next_: _dafny.Seq
                out89_: _dafny.Seq
                out89_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_102_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_103_next_ = out89_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_103_next_) != (eosToken):
                    d_104_g2_: _dafny.Seq
                    d_105_i2_: bool
                    d_106_c2_: _dafny.Seq
                    out90_: _dafny.Seq
                    out91_: bool
                    out92_: _dafny.Seq
                    out90_, out91_, out92_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_103_next_)
                    d_104_g2_ = out90_
                    d_105_i2_ = out91_
                    d_106_c2_ = out92_
                    generated = d_104_g2_
                    insideConstrainedOut = d_105_i2_
                    currentConstrainedOut = d_106_c2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_107_g2_: _dafny.Seq
                d_108_i2_: bool
                d_109_c2_: _dafny.Seq
                out93_: _dafny.Seq
                out94_: bool
                out95_: _dafny.Seq
                out93_, out94_, out95_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_107_g2_ = out93_
                d_108_i2_ = out94_
                d_109_c2_ = out95_
                generated = d_107_g2_
                insideConstrainedOut = d_108_i2_
                currentConstrainedOut = d_109_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

