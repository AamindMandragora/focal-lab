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
        d_6_emptySpanEosCount_: int
        d_6_emptySpanEosCount_ = 0
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
                            d_5_hasSeenOpenSpan_ = True
                            d_6_emptySpanEosCount_ = 0
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
                                    d_5_hasSeenOpenSpan_ = True
                                    d_6_emptySpanEosCount_ = 0
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
                                d_5_hasSeenOpenSpan_ = True
                                d_6_emptySpanEosCount_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_22_g2_: _dafny.Seq
                        d_23_i2_: bool
                        d_24_c2_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_22_g2_ = out13_
                        d_23_i2_ = out14_
                        d_24_c2_ = out15_
                        generated = d_22_g2_
                        insideConstrainedOut = d_23_i2_
                        currentConstrainedOut = d_24_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokensUsed_ = 0
                        d_6_emptySpanEosCount_ = 0
                    elif True:
                        d_25_isDeadEnd_: bool
                        out16_: bool
                        out16_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_25_isDeadEnd_ = out16_
                        if (d_25_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_)):
                            d_26_gRolled_: _dafny.Seq
                            d_27_cRolled_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: _dafny.Seq
                            out17_, out18_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_26_gRolled_ = out17_
                            d_27_cRolled_ = out18_
                            generated = d_26_gRolled_
                            currentConstrainedOut = d_27_cRolled_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_28_g3_: _dafny.Seq
                                d_29_c3_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: _dafny.Seq
                                out19_, out20_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_28_g3_ = out19_
                                d_29_c3_ = out20_
                                generated = d_28_g3_
                                currentConstrainedOut = d_29_c3_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_30_g4_: _dafny.Seq
                                d_31_c4_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: _dafny.Seq
                                out21_, out22_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_30_g4_ = out21_
                                d_31_c4_ = out22_
                                generated = d_30_g4_
                                currentConstrainedOut = d_31_c4_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_32_g5_: _dafny.Seq
                                d_33_c5_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: _dafny.Seq
                                out23_, out24_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_32_g5_ = out23_
                                d_33_c5_ = out24_
                                generated = d_32_g5_
                                currentConstrainedOut = d_33_c5_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_34_g6_: _dafny.Seq
                                d_35_c6_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: _dafny.Seq
                                out25_, out26_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_34_g6_ = out25_
                                d_35_c6_ = out26_
                                generated = d_34_g6_
                                currentConstrainedOut = d_35_c6_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_36_g7_: _dafny.Seq
                                d_37_c7_: _dafny.Seq
                                out27_: _dafny.Seq
                                out28_: _dafny.Seq
                                out27_, out28_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_36_g7_ = out27_
                                d_37_c7_ = out28_
                                generated = d_36_g7_
                                currentConstrainedOut = d_37_c7_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_38_g2_: _dafny.Seq
                                d_39_i2_: bool
                                d_40_c2_: _dafny.Seq
                                out29_: _dafny.Seq
                                out30_: bool
                                out31_: _dafny.Seq
                                out29_, out30_, out31_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_38_g2_ = out29_
                                d_39_i2_ = out30_
                                d_40_c2_ = out31_
                                generated = d_38_g2_
                                insideConstrainedOut = d_39_i2_
                                currentConstrainedOut = d_40_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_6_emptySpanEosCount_ = 0
                            elif (d_1_steps_) < (maxSteps):
                                d_41_constrainedPrompt_: _dafny.Seq
                                d_41_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_42_next_: _dafny.Seq
                                out32_: _dafny.Seq
                                out32_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_41_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_42_next_ = out32_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_42_next_) == (eosToken):
                                    d_6_emptySpanEosCount_ = (d_6_emptySpanEosCount_) + (1)
                                    if (((len(currentConstrainedOut)) == (0)) and ((d_6_emptySpanEosCount_) >= (2))) and ((d_1_steps_) < (maxSteps)):
                                        d_43_candidates_: _dafny.Seq
                                        out33_: _dafny.Seq
                                        out33_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_41_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                                        d_43_candidates_ = out33_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if ((len(d_43_candidates_)) > (0)) and (((d_43_candidates_)[0]) != (eosToken)):
                                            d_44_isValid_: bool
                                            out34_: bool
                                            out34_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, (d_43_candidates_)[0])
                                            d_44_isValid_ = out34_
                                            if d_44_isValid_:
                                                d_45_g2_: _dafny.Seq
                                                d_46_i2_: bool
                                                d_47_c2_: _dafny.Seq
                                                out35_: _dafny.Seq
                                                out36_: bool
                                                out37_: _dafny.Seq
                                                out35_, out36_, out37_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, (d_43_candidates_)[0])
                                                d_45_g2_ = out35_
                                                d_46_i2_ = out36_
                                                d_47_c2_ = out37_
                                                generated = d_45_g2_
                                                insideConstrainedOut = d_46_i2_
                                                currentConstrainedOut = d_47_c2_
                                                d_3_spanTokensUsed_ = 1
                                                d_6_emptySpanEosCount_ = 0
                                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                                    d_48_g3_: _dafny.Seq
                                                    d_49_i3_: bool
                                                    d_50_c3_: _dafny.Seq
                                                    out38_: _dafny.Seq
                                                    out39_: bool
                                                    out40_: _dafny.Seq
                                                    out38_, out39_, out40_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                    d_48_g3_ = out38_
                                                    d_49_i3_ = out39_
                                                    d_50_c3_ = out40_
                                                    generated = d_48_g3_
                                                    insideConstrainedOut = d_49_i3_
                                                    currentConstrainedOut = d_50_c3_
                                                    d_1_steps_ = (d_1_steps_) + (1)
                                            elif True:
                                                raise _dafny.Break("0")
                                        elif True:
                                            raise _dafny.Break("0")
                                    elif True:
                                        d_51_gR2_: _dafny.Seq
                                        d_52_cR2_: _dafny.Seq
                                        out41_: _dafny.Seq
                                        out42_: _dafny.Seq
                                        out41_, out42_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_51_gR2_ = out41_
                                        d_52_cR2_ = out42_
                                        generated = d_51_gR2_
                                        currentConstrainedOut = d_52_cR2_
                                        if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                            d_53_gR3_: _dafny.Seq
                                            d_54_cR3_: _dafny.Seq
                                            out43_: _dafny.Seq
                                            out44_: _dafny.Seq
                                            out43_, out44_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                            d_53_gR3_ = out43_
                                            d_54_cR3_ = out44_
                                            generated = d_53_gR3_
                                            currentConstrainedOut = d_54_cR3_
                                        if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                            d_55_gR4_: _dafny.Seq
                                            d_56_cR4_: _dafny.Seq
                                            out45_: _dafny.Seq
                                            out46_: _dafny.Seq
                                            out45_, out46_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                            d_55_gR4_ = out45_
                                            d_56_cR4_ = out46_
                                            generated = d_55_gR4_
                                            currentConstrainedOut = d_56_cR4_
                                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                            d_57_g2_: _dafny.Seq
                                            d_58_i2_: bool
                                            d_59_c2_: _dafny.Seq
                                            out47_: _dafny.Seq
                                            out48_: bool
                                            out49_: _dafny.Seq
                                            out47_, out48_, out49_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_57_g2_ = out47_
                                            d_58_i2_ = out48_
                                            d_59_c2_ = out49_
                                            generated = d_57_g2_
                                            insideConstrainedOut = d_58_i2_
                                            currentConstrainedOut = d_59_c2_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            d_6_emptySpanEosCount_ = 0
                                        elif (len(currentConstrainedOut)) == (0):
                                            raise _dafny.Break("0")
                                elif True:
                                    d_6_emptySpanEosCount_ = 0
                                    d_60_g2_: _dafny.Seq
                                    d_61_i2_: bool
                                    d_62_c2_: _dafny.Seq
                                    out50_: _dafny.Seq
                                    out51_: bool
                                    out52_: _dafny.Seq
                                    out50_, out51_, out52_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_42_next_)
                                    d_60_g2_ = out50_
                                    d_61_i2_ = out51_
                                    d_62_c2_ = out52_
                                    generated = d_60_g2_
                                    insideConstrainedOut = d_61_i2_
                                    currentConstrainedOut = d_62_c2_
                                    d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_63_g3_: _dafny.Seq
                                        d_64_i3_: bool
                                        d_65_c3_: _dafny.Seq
                                        out53_: _dafny.Seq
                                        out54_: bool
                                        out55_: _dafny.Seq
                                        out53_, out54_, out55_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_63_g3_ = out53_
                                        d_64_i3_ = out54_
                                        d_65_c3_ = out55_
                                        generated = d_63_g3_
                                        insideConstrainedOut = d_64_i3_
                                        currentConstrainedOut = d_65_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                        elif True:
                            d_66_constrainedPrompt_: _dafny.Seq
                            d_66_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_67_next_: _dafny.Seq
                            d_68_wasConstrained_: bool
                            out56_: _dafny.Seq
                            out57_: bool
                            out56_, out57_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_66_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_67_next_ = out56_
                            d_68_wasConstrained_ = out57_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_67_next_) == (eosToken):
                                d_6_emptySpanEosCount_ = (d_6_emptySpanEosCount_) + (1)
                                if (((len(currentConstrainedOut)) == (0)) and ((d_6_emptySpanEosCount_) >= (2))) and ((d_1_steps_) < (maxSteps)):
                                    d_69_candidates_: _dafny.Seq
                                    out58_: _dafny.Seq
                                    out58_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_66_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                                    d_69_candidates_ = out58_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if ((len(d_69_candidates_)) > (0)) and (((d_69_candidates_)[0]) != (eosToken)):
                                        d_70_isValid_: bool
                                        out59_: bool
                                        out59_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, (d_69_candidates_)[0])
                                        d_70_isValid_ = out59_
                                        if d_70_isValid_:
                                            d_71_g2_: _dafny.Seq
                                            d_72_i2_: bool
                                            d_73_c2_: _dafny.Seq
                                            out60_: _dafny.Seq
                                            out61_: bool
                                            out62_: _dafny.Seq
                                            out60_, out61_, out62_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, (d_69_candidates_)[0])
                                            d_71_g2_ = out60_
                                            d_72_i2_ = out61_
                                            d_73_c2_ = out62_
                                            generated = d_71_g2_
                                            insideConstrainedOut = d_72_i2_
                                            currentConstrainedOut = d_73_c2_
                                            d_3_spanTokensUsed_ = 1
                                            d_6_emptySpanEosCount_ = 0
                                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                                d_74_g3_: _dafny.Seq
                                                d_75_i3_: bool
                                                d_76_c3_: _dafny.Seq
                                                out63_: _dafny.Seq
                                                out64_: bool
                                                out65_: _dafny.Seq
                                                out63_, out64_, out65_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_74_g3_ = out63_
                                                d_75_i3_ = out64_
                                                d_76_c3_ = out65_
                                                generated = d_74_g3_
                                                insideConstrainedOut = d_75_i3_
                                                currentConstrainedOut = d_76_c3_
                                                d_1_steps_ = (d_1_steps_) + (1)
                                        elif True:
                                            raise _dafny.Break("0")
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    d_77_gRolled_: _dafny.Seq
                                    d_78_cRolled_: _dafny.Seq
                                    out66_: _dafny.Seq
                                    out67_: _dafny.Seq
                                    out66_, out67_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_77_gRolled_ = out66_
                                    d_78_cRolled_ = out67_
                                    generated = d_77_gRolled_
                                    currentConstrainedOut = d_78_cRolled_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_79_gR2_: _dafny.Seq
                                        d_80_cR2_: _dafny.Seq
                                        out68_: _dafny.Seq
                                        out69_: _dafny.Seq
                                        out68_, out69_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_79_gR2_ = out68_
                                        d_80_cR2_ = out69_
                                        generated = d_79_gR2_
                                        currentConstrainedOut = d_80_cR2_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_81_gR3_: _dafny.Seq
                                        d_82_cR3_: _dafny.Seq
                                        out70_: _dafny.Seq
                                        out71_: _dafny.Seq
                                        out70_, out71_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_81_gR3_ = out70_
                                        d_82_cR3_ = out71_
                                        generated = d_81_gR3_
                                        currentConstrainedOut = d_82_cR3_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_83_gR4_: _dafny.Seq
                                        d_84_cR4_: _dafny.Seq
                                        out72_: _dafny.Seq
                                        out73_: _dafny.Seq
                                        out72_, out73_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_83_gR4_ = out72_
                                        d_84_cR4_ = out73_
                                        generated = d_83_gR4_
                                        currentConstrainedOut = d_84_cR4_
                                    if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                        d_85_gR5_: _dafny.Seq
                                        d_86_cR5_: _dafny.Seq
                                        out74_: _dafny.Seq
                                        out75_: _dafny.Seq
                                        out74_, out75_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                        d_85_gR5_ = out74_
                                        d_86_cR5_ = out75_
                                        generated = d_85_gR5_
                                        currentConstrainedOut = d_86_cR5_
                                    d_3_spanTokensUsed_ = 0
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
                                        d_6_emptySpanEosCount_ = 0
                                    elif (len(currentConstrainedOut)) == (0):
                                        raise _dafny.Break("0")
                            elif True:
                                d_6_emptySpanEosCount_ = 0
                                d_90_g2_: _dafny.Seq
                                d_91_i2_: bool
                                d_92_c2_: _dafny.Seq
                                out79_: _dafny.Seq
                                out80_: bool
                                out81_: _dafny.Seq
                                out79_, out80_, out81_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_67_next_)
                                d_90_g2_ = out79_
                                d_91_i2_ = out80_
                                d_92_c2_ = out81_
                                generated = d_90_g2_
                                insideConstrainedOut = d_91_i2_
                                currentConstrainedOut = d_92_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_93_gRolled_: _dafny.Seq
            d_94_cRolled_: _dafny.Seq
            out82_: _dafny.Seq
            out83_: _dafny.Seq
            out82_, out83_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
            d_93_gRolled_ = out82_
            d_94_cRolled_ = out83_
            generated = d_93_gRolled_
            currentConstrainedOut = d_94_cRolled_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_95_g2_: _dafny.Seq
                d_96_c2_: _dafny.Seq
                out84_: _dafny.Seq
                out85_: _dafny.Seq
                out84_, out85_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_95_g2_ = out84_
                d_96_c2_ = out85_
                generated = d_95_g2_
                currentConstrainedOut = d_96_c2_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_97_g3_: _dafny.Seq
                d_98_c3_: _dafny.Seq
                out86_: _dafny.Seq
                out87_: _dafny.Seq
                out86_, out87_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_97_g3_ = out86_
                d_98_c3_ = out87_
                generated = d_97_g3_
                currentConstrainedOut = d_98_c3_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_99_g4_: _dafny.Seq
                d_100_c4_: _dafny.Seq
                out88_: _dafny.Seq
                out89_: _dafny.Seq
                out88_, out89_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_99_g4_ = out88_
                d_100_c4_ = out89_
                generated = d_99_g4_
                currentConstrainedOut = d_100_c4_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_101_g5_: _dafny.Seq
                d_102_c5_: _dafny.Seq
                out90_: _dafny.Seq
                out91_: _dafny.Seq
                out90_, out91_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_101_g5_ = out90_
                d_102_c5_ = out91_
                generated = d_101_g5_
                currentConstrainedOut = d_102_c5_
            if ((not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) == (0))) and ((d_1_steps_) < (maxSteps)):
                d_103_constrainedPromptPost_: _dafny.Seq
                d_103_constrainedPromptPost_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_104_candidates_: _dafny.Seq
                out92_: _dafny.Seq
                out92_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_103_constrainedPromptPost_, currentConstrainedOut, 4, eosToken)
                d_104_candidates_ = out92_
                d_1_steps_ = (d_1_steps_) + (1)
                if ((len(d_104_candidates_)) > (0)) and (((d_104_candidates_)[0]) != (eosToken)):
                    d_105_isValidPost_: bool
                    out93_: bool
                    out93_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, (d_104_candidates_)[0])
                    d_105_isValidPost_ = out93_
                    if d_105_isValidPost_:
                        d_106_gF_: _dafny.Seq
                        d_107_iF_: bool
                        d_108_cF_: _dafny.Seq
                        out94_: _dafny.Seq
                        out95_: bool
                        out96_: _dafny.Seq
                        out94_, out95_, out96_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, (d_104_candidates_)[0])
                        d_106_gF_ = out94_
                        d_107_iF_ = out95_
                        d_108_cF_ = out96_
                        generated = d_106_gF_
                        insideConstrainedOut = d_107_iF_
                        currentConstrainedOut = d_108_cF_
            elif (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_109_constrainedPromptPost_: _dafny.Seq
                d_109_constrainedPromptPost_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_110_nextPost_: _dafny.Seq
                out97_: _dafny.Seq
                out97_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_109_constrainedPromptPost_, currentConstrainedOut, eosToken)
                d_110_nextPost_ = out97_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_110_nextPost_) != (eosToken):
                    d_111_g2_: _dafny.Seq
                    d_112_i2_: bool
                    d_113_c2_: _dafny.Seq
                    out98_: _dafny.Seq
                    out99_: bool
                    out100_: _dafny.Seq
                    out98_, out99_, out100_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_110_nextPost_)
                    d_111_g2_ = out98_
                    d_112_i2_ = out99_
                    d_113_c2_ = out100_
                    generated = d_111_g2_
                    insideConstrainedOut = d_112_i2_
                    currentConstrainedOut = d_113_c2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_114_g2_: _dafny.Seq
                d_115_i2_: bool
                d_116_c2_: _dafny.Seq
                out101_: _dafny.Seq
                out102_: bool
                out103_: _dafny.Seq
                out101_, out102_, out103_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_114_g2_ = out101_
                d_115_i2_ = out102_
                d_116_c2_ = out103_
                generated = d_114_g2_
                insideConstrainedOut = d_115_i2_
                currentConstrainedOut = d_116_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

