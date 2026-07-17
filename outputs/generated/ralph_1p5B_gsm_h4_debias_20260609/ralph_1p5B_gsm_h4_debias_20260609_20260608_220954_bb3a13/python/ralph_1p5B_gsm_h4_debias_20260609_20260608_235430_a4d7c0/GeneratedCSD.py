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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (insideConstrainedOut) and (((maxSteps) - (d_1_steps_)) <= (4)):
                        d_6_gR_: _dafny.Seq
                        d_7_cR_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: _dafny.Seq
                        out0_, out1_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_6_gR_ = out0_
                        d_7_cR_ = out1_
                        generated = d_6_gR_
                        currentConstrainedOut = d_7_cR_
                        if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((maxSteps) - (d_1_steps_)) >= (1)):
                            d_8_constrainedPrompt_: _dafny.Seq
                            d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_9_next_: _dafny.Seq
                            out2_: _dafny.Seq
                            out2_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_9_next_ = out2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) != (eosToken):
                                d_10_g2_: _dafny.Seq
                                d_11_i2_: bool
                                d_12_c2_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                                d_10_g2_ = out3_
                                d_11_i2_ = out4_
                                d_12_c2_ = out5_
                                generated = d_10_g2_
                                insideConstrainedOut = d_11_i2_
                                currentConstrainedOut = d_12_c2_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((maxSteps) - (d_1_steps_)) >= (1)):
                            d_13_g2_: _dafny.Seq
                            d_14_i2_: bool
                            d_15_c2_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_g2_ = out6_
                            d_14_i2_ = out7_
                            d_15_c2_ = out8_
                            generated = d_13_g2_
                            insideConstrainedOut = d_14_i2_
                            currentConstrainedOut = d_15_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif not(insideConstrainedOut):
                        d_16_remaining_: int
                        d_16_remaining_ = (maxSteps) - (d_1_steps_)
                        if ((((d_16_remaining_) <= (200)) or ((d_1_steps_) >= (300))) and (not(d_5_hasSeenOpenSpan_))) and ((d_16_remaining_) > (4)):
                            d_17_g2_: _dafny.Seq
                            d_18_i2_: bool
                            d_19_c2_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_17_g2_ = out9_
                            d_18_i2_ = out10_
                            d_19_c2_ = out11_
                            generated = d_17_g2_
                            insideConstrainedOut = d_18_i2_
                            currentConstrainedOut = d_19_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                            d_5_hasSeenOpenSpan_ = True
                        elif True:
                            d_20_chunkBudget_: int
                            if (d_16_remaining_) < (d_2_freeChunkSize_):
                                d_20_chunkBudget_ = d_16_remaining_
                            elif True:
                                d_20_chunkBudget_ = d_2_freeChunkSize_
                            if (d_20_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            d_21_chunkGenerated_: _dafny.Seq
                            d_22_stoppedOnOpenSpan_: bool
                            d_23_stoppedOnEos_: bool
                            d_24_stepsUsed_: int
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: bool
                            out15_: int
                            out12_, out13_, out14_, out15_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_20_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_21_chunkGenerated_ = out12_
                            d_22_stoppedOnOpenSpan_ = out13_
                            d_23_stoppedOnEos_ = out14_
                            d_24_stepsUsed_ = out15_
                            generated = d_21_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_24_stepsUsed_)
                            if d_23_stoppedOnEos_:
                                if (not(d_5_hasSeenOpenSpan_)) and (((d_1_steps_) + (1)) < (maxSteps)):
                                    d_25_g2_: _dafny.Seq
                                    d_26_i2_: bool
                                    d_27_c2_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_25_g2_ = out16_
                                    d_26_i2_ = out17_
                                    d_27_c2_ = out18_
                                    generated = d_25_g2_
                                    insideConstrainedOut = d_26_i2_
                                    currentConstrainedOut = d_27_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                                    d_5_hasSeenOpenSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif d_22_stoppedOnOpenSpan_:
                                d_28_g2_: _dafny.Seq
                                d_29_i2_: bool
                                d_30_c2_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_28_g2_ = out19_
                                d_29_i2_ = out20_
                                d_30_c2_ = out21_
                                generated = d_28_g2_
                                insideConstrainedOut = d_29_i2_
                                currentConstrainedOut = d_30_c2_
                                d_3_spanTokensUsed_ = 0
                                d_5_hasSeenOpenSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_31_g2_: _dafny.Seq
                        d_32_i2_: bool
                        d_33_c2_: _dafny.Seq
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: _dafny.Seq
                        out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_31_g2_ = out22_
                        d_32_i2_ = out23_
                        d_33_c2_ = out24_
                        generated = d_31_g2_
                        insideConstrainedOut = d_32_i2_
                        currentConstrainedOut = d_33_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokensUsed_ = 0
                    elif True:
                        d_34_isDeadEnd_: bool
                        out25_: bool
                        out25_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_34_isDeadEnd_ = out25_
                        if (d_34_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_)):
                            d_35_gRolled_: _dafny.Seq
                            d_36_cRolled_: _dafny.Seq
                            out26_: _dafny.Seq
                            out27_: _dafny.Seq
                            out26_, out27_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_35_gRolled_ = out26_
                            d_36_cRolled_ = out27_
                            generated = d_35_gRolled_
                            currentConstrainedOut = d_36_cRolled_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_37_g3_: _dafny.Seq
                                d_38_c3_: _dafny.Seq
                                out28_: _dafny.Seq
                                out29_: _dafny.Seq
                                out28_, out29_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_37_g3_ = out28_
                                d_38_c3_ = out29_
                                generated = d_37_g3_
                                currentConstrainedOut = d_38_c3_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_39_g4_: _dafny.Seq
                                d_40_c4_: _dafny.Seq
                                out30_: _dafny.Seq
                                out31_: _dafny.Seq
                                out30_, out31_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_39_g4_ = out30_
                                d_40_c4_ = out31_
                                generated = d_39_g4_
                                currentConstrainedOut = d_40_c4_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_41_g5_: _dafny.Seq
                                d_42_c5_: _dafny.Seq
                                out32_: _dafny.Seq
                                out33_: _dafny.Seq
                                out32_, out33_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_41_g5_ = out32_
                                d_42_c5_ = out33_
                                generated = d_41_g5_
                                currentConstrainedOut = d_42_c5_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_43_g6_: _dafny.Seq
                                d_44_c6_: _dafny.Seq
                                out34_: _dafny.Seq
                                out35_: _dafny.Seq
                                out34_, out35_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_43_g6_ = out34_
                                d_44_c6_ = out35_
                                generated = d_43_g6_
                                currentConstrainedOut = d_44_c6_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_45_g7_: _dafny.Seq
                                d_46_c7_: _dafny.Seq
                                out36_: _dafny.Seq
                                out37_: _dafny.Seq
                                out36_, out37_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_45_g7_ = out36_
                                d_46_c7_ = out37_
                                generated = d_45_g7_
                                currentConstrainedOut = d_46_c7_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_47_g2_: _dafny.Seq
                                d_48_i2_: bool
                                d_49_c2_: _dafny.Seq
                                out38_: _dafny.Seq
                                out39_: bool
                                out40_: _dafny.Seq
                                out38_, out39_, out40_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_47_g2_ = out38_
                                d_48_i2_ = out39_
                                d_49_c2_ = out40_
                                generated = d_47_g2_
                                insideConstrainedOut = d_48_i2_
                                currentConstrainedOut = d_49_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif (d_1_steps_) < (maxSteps):
                                d_50_constrainedPrompt_: _dafny.Seq
                                d_50_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_51_next_: _dafny.Seq
                                out41_: _dafny.Seq
                                out41_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_50_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_51_next_ = out41_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_51_next_) == (eosToken):
                                    d_52_gR2_: _dafny.Seq
                                    d_53_cR2_: _dafny.Seq
                                    out42_: _dafny.Seq
                                    out43_: _dafny.Seq
                                    out42_, out43_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_52_gR2_ = out42_
                                    d_53_cR2_ = out43_
                                    generated = d_52_gR2_
                                    currentConstrainedOut = d_53_cR2_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_54_gR3_: _dafny.Seq
                                        d_55_cR3_: _dafny.Seq
                                        out44_: _dafny.Seq
                                        out45_: _dafny.Seq
                                        out44_, out45_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_54_gR3_ = out44_
                                        d_55_cR3_ = out45_
                                        generated = d_54_gR3_
                                        currentConstrainedOut = d_55_cR3_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_56_gR4_: _dafny.Seq
                                        d_57_cR4_: _dafny.Seq
                                        out46_: _dafny.Seq
                                        out47_: _dafny.Seq
                                        out46_, out47_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_56_gR4_ = out46_
                                        d_57_cR4_ = out47_
                                        generated = d_56_gR4_
                                        currentConstrainedOut = d_57_cR4_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_58_g2_: _dafny.Seq
                                        d_59_i2_: bool
                                        d_60_c2_: _dafny.Seq
                                        out48_: _dafny.Seq
                                        out49_: bool
                                        out50_: _dafny.Seq
                                        out48_, out49_, out50_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_58_g2_ = out48_
                                        d_59_i2_ = out49_
                                        d_60_c2_ = out50_
                                        generated = d_58_g2_
                                        insideConstrainedOut = d_59_i2_
                                        currentConstrainedOut = d_60_c2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_61_g2_: _dafny.Seq
                                    d_62_i2_: bool
                                    d_63_c2_: _dafny.Seq
                                    out51_: _dafny.Seq
                                    out52_: bool
                                    out53_: _dafny.Seq
                                    out51_, out52_, out53_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_51_next_)
                                    d_61_g2_ = out51_
                                    d_62_i2_ = out52_
                                    d_63_c2_ = out53_
                                    generated = d_61_g2_
                                    insideConstrainedOut = d_62_i2_
                                    currentConstrainedOut = d_63_c2_
                                    d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_64_g3_: _dafny.Seq
                                        d_65_i3_: bool
                                        d_66_c3_: _dafny.Seq
                                        out54_: _dafny.Seq
                                        out55_: bool
                                        out56_: _dafny.Seq
                                        out54_, out55_, out56_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_64_g3_ = out54_
                                        d_65_i3_ = out55_
                                        d_66_c3_ = out56_
                                        generated = d_64_g3_
                                        insideConstrainedOut = d_65_i3_
                                        currentConstrainedOut = d_66_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                        elif True:
                            d_67_constrainedPrompt_: _dafny.Seq
                            d_67_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_68_next_: _dafny.Seq
                            d_69_wasConstrained_: bool
                            out57_: _dafny.Seq
                            out58_: bool
                            out57_, out58_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_67_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_68_next_ = out57_
                            d_69_wasConstrained_ = out58_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_68_next_) == (eosToken):
                                d_70_gRolled_: _dafny.Seq
                                d_71_cRolled_: _dafny.Seq
                                out59_: _dafny.Seq
                                out60_: _dafny.Seq
                                out59_, out60_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_70_gRolled_ = out59_
                                d_71_cRolled_ = out60_
                                generated = d_70_gRolled_
                                currentConstrainedOut = d_71_cRolled_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_72_gR2_: _dafny.Seq
                                    d_73_cR2_: _dafny.Seq
                                    out61_: _dafny.Seq
                                    out62_: _dafny.Seq
                                    out61_, out62_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_72_gR2_ = out61_
                                    d_73_cR2_ = out62_
                                    generated = d_72_gR2_
                                    currentConstrainedOut = d_73_cR2_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_74_gR3_: _dafny.Seq
                                    d_75_cR3_: _dafny.Seq
                                    out63_: _dafny.Seq
                                    out64_: _dafny.Seq
                                    out63_, out64_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_74_gR3_ = out63_
                                    d_75_cR3_ = out64_
                                    generated = d_74_gR3_
                                    currentConstrainedOut = d_75_cR3_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_76_gR4_: _dafny.Seq
                                    d_77_cR4_: _dafny.Seq
                                    out65_: _dafny.Seq
                                    out66_: _dafny.Seq
                                    out65_, out66_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_76_gR4_ = out65_
                                    d_77_cR4_ = out66_
                                    generated = d_76_gR4_
                                    currentConstrainedOut = d_77_cR4_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_78_gR5_: _dafny.Seq
                                    d_79_cR5_: _dafny.Seq
                                    out67_: _dafny.Seq
                                    out68_: _dafny.Seq
                                    out67_, out68_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_78_gR5_ = out67_
                                    d_79_cR5_ = out68_
                                    generated = d_78_gR5_
                                    currentConstrainedOut = d_79_cR5_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_80_g2_: _dafny.Seq
                                    d_81_i2_: bool
                                    d_82_c2_: _dafny.Seq
                                    out69_: _dafny.Seq
                                    out70_: bool
                                    out71_: _dafny.Seq
                                    out69_, out70_, out71_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_80_g2_ = out69_
                                    d_81_i2_ = out70_
                                    d_82_c2_ = out71_
                                    generated = d_80_g2_
                                    insideConstrainedOut = d_81_i2_
                                    currentConstrainedOut = d_82_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_83_g2_: _dafny.Seq
                                d_84_i2_: bool
                                d_85_c2_: _dafny.Seq
                                out72_: _dafny.Seq
                                out73_: bool
                                out74_: _dafny.Seq
                                out72_, out73_, out74_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_68_next_)
                                d_83_g2_ = out72_
                                d_84_i2_ = out73_
                                d_85_c2_ = out74_
                                generated = d_83_g2_
                                insideConstrainedOut = d_84_i2_
                                currentConstrainedOut = d_85_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_86_gRolled_: _dafny.Seq
            d_87_cRolled_: _dafny.Seq
            out75_: _dafny.Seq
            out76_: _dafny.Seq
            out75_, out76_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
            d_86_gRolled_ = out75_
            d_87_cRolled_ = out76_
            generated = d_86_gRolled_
            currentConstrainedOut = d_87_cRolled_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_88_g2_: _dafny.Seq
                d_89_c2_: _dafny.Seq
                out77_: _dafny.Seq
                out78_: _dafny.Seq
                out77_, out78_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_88_g2_ = out77_
                d_89_c2_ = out78_
                generated = d_88_g2_
                currentConstrainedOut = d_89_c2_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_90_g3_: _dafny.Seq
                d_91_c3_: _dafny.Seq
                out79_: _dafny.Seq
                out80_: _dafny.Seq
                out79_, out80_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_90_g3_ = out79_
                d_91_c3_ = out80_
                generated = d_90_g3_
                currentConstrainedOut = d_91_c3_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_92_g4_: _dafny.Seq
                d_93_c4_: _dafny.Seq
                out81_: _dafny.Seq
                out82_: _dafny.Seq
                out81_, out82_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_92_g4_ = out81_
                d_93_c4_ = out82_
                generated = d_92_g4_
                currentConstrainedOut = d_93_c4_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_94_g5_: _dafny.Seq
                d_95_c5_: _dafny.Seq
                out83_: _dafny.Seq
                out84_: _dafny.Seq
                out83_, out84_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_94_g5_ = out83_
                d_95_c5_ = out84_
                generated = d_94_g5_
                currentConstrainedOut = d_95_c5_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_96_constrainedPrompt_: _dafny.Seq
                d_96_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_97_next_: _dafny.Seq
                out85_: _dafny.Seq
                out85_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_96_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_97_next_ = out85_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_97_next_) != (eosToken):
                    d_98_g2_: _dafny.Seq
                    d_99_i2_: bool
                    d_100_c2_: _dafny.Seq
                    out86_: _dafny.Seq
                    out87_: bool
                    out88_: _dafny.Seq
                    out86_, out87_, out88_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_97_next_)
                    d_98_g2_ = out86_
                    d_99_i2_ = out87_
                    d_100_c2_ = out88_
                    generated = d_98_g2_
                    insideConstrainedOut = d_99_i2_
                    currentConstrainedOut = d_100_c2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_101_g2_: _dafny.Seq
                d_102_i2_: bool
                d_103_c2_: _dafny.Seq
                out89_: _dafny.Seq
                out90_: bool
                out91_: _dafny.Seq
                out89_, out90_, out91_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_101_g2_ = out89_
                d_102_i2_ = out90_
                d_103_c2_ = out91_
                generated = d_101_g2_
                insideConstrainedOut = d_102_i2_
                currentConstrainedOut = d_103_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

