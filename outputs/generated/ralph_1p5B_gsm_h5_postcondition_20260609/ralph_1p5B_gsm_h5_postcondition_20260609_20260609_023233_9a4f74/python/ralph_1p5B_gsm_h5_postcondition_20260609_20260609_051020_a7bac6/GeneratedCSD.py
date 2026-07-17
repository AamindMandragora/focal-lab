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
        d_4_spanMaxTokens_ = 6
        d_5_hasSeenOpenSpan_: bool
        d_5_hasSeenOpenSpan_ = insideConstrained
        d_6_closeReserve_: int
        d_6_closeReserve_ = 5
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_7_remaining_: int
                    d_7_remaining_ = (maxSteps) - (d_1_steps_)
                    if (insideConstrainedOut) and ((d_7_remaining_) <= (d_6_closeReserve_)):
                        d_8_gRolled_: _dafny.Seq
                        d_9_cRolled_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: _dafny.Seq
                        out0_, out1_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_8_gRolled_ = out0_
                        d_9_cRolled_ = out1_
                        generated = d_8_gRolled_
                        currentConstrainedOut = d_9_cRolled_
                        if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                            d_10_constrainedPromptE_: _dafny.Seq
                            d_10_constrainedPromptE_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_11_nextE_: _dafny.Seq
                            out2_: _dafny.Seq
                            out2_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPromptE_, currentConstrainedOut, eosToken)
                            d_11_nextE_ = out2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_nextE_) != (eosToken):
                                d_12_gE_: _dafny.Seq
                                d_13_iE_: bool
                                d_14_cE_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_nextE_)
                                d_12_gE_ = out3_
                                d_13_iE_ = out4_
                                d_14_cE_ = out5_
                                generated = d_12_gE_
                                insideConstrainedOut = d_13_iE_
                                currentConstrainedOut = d_14_cE_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_15_g2_: _dafny.Seq
                            d_16_i2_: bool
                            d_17_c2_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_15_g2_ = out6_
                            d_16_i2_ = out7_
                            d_17_c2_ = out8_
                            generated = d_15_g2_
                            insideConstrainedOut = d_16_i2_
                            currentConstrainedOut = d_17_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    if not(insideConstrainedOut):
                        if (((d_7_remaining_) <= (65)) and (not(d_5_hasSeenOpenSpan_))) and ((d_7_remaining_) > ((d_6_closeReserve_) + (1))):
                            d_18_g2_: _dafny.Seq
                            d_19_i2_: bool
                            d_20_c2_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_18_g2_ = out9_
                            d_19_i2_ = out10_
                            d_20_c2_ = out11_
                            generated = d_18_g2_
                            insideConstrainedOut = d_19_i2_
                            currentConstrainedOut = d_20_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                            d_5_hasSeenOpenSpan_ = True
                        elif True:
                            d_21_chunkBudget_: int
                            if (d_7_remaining_) < (d_2_freeChunkSize_):
                                d_21_chunkBudget_ = d_7_remaining_
                            elif True:
                                d_21_chunkBudget_ = d_2_freeChunkSize_
                            if (d_21_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            d_22_chunkGenerated_: _dafny.Seq
                            d_23_stoppedOnOpenSpan_: bool
                            d_24_stoppedOnEos_: bool
                            d_25_stepsUsed_: int
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: bool
                            out15_: int
                            out12_, out13_, out14_, out15_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_21_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_22_chunkGenerated_ = out12_
                            d_23_stoppedOnOpenSpan_ = out13_
                            d_24_stoppedOnEos_ = out14_
                            d_25_stepsUsed_ = out15_
                            generated = d_22_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_25_stepsUsed_)
                            if d_24_stoppedOnEos_:
                                if (not(d_5_hasSeenOpenSpan_)) and ((((d_1_steps_) + (d_6_closeReserve_)) + (2)) <= (maxSteps)):
                                    d_26_g2_: _dafny.Seq
                                    d_27_i2_: bool
                                    d_28_c2_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_26_g2_ = out16_
                                    d_27_i2_ = out17_
                                    d_28_c2_ = out18_
                                    generated = d_26_g2_
                                    insideConstrainedOut = d_27_i2_
                                    currentConstrainedOut = d_28_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                                    d_5_hasSeenOpenSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif d_23_stoppedOnOpenSpan_:
                                d_29_g2_: _dafny.Seq
                                d_30_i2_: bool
                                d_31_c2_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_29_g2_ = out19_
                                d_30_i2_ = out20_
                                d_31_c2_ = out21_
                                generated = d_29_g2_
                                insideConstrainedOut = d_30_i2_
                                currentConstrainedOut = d_31_c2_
                                d_3_spanTokensUsed_ = 0
                                d_5_hasSeenOpenSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_32_g2_: _dafny.Seq
                        d_33_i2_: bool
                        d_34_c2_: _dafny.Seq
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: _dafny.Seq
                        out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_32_g2_ = out22_
                        d_33_i2_ = out23_
                        d_34_c2_ = out24_
                        generated = d_32_g2_
                        insideConstrainedOut = d_33_i2_
                        currentConstrainedOut = d_34_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokensUsed_ = 0
                    elif True:
                        d_35_isDeadEnd_: bool
                        out25_: bool
                        out25_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_35_isDeadEnd_ = out25_
                        if (d_35_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_)):
                            d_36_gRolled_: _dafny.Seq
                            d_37_cRolled_: _dafny.Seq
                            out26_: _dafny.Seq
                            out27_: _dafny.Seq
                            out26_, out27_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_36_gRolled_ = out26_
                            d_37_cRolled_ = out27_
                            generated = d_36_gRolled_
                            currentConstrainedOut = d_37_cRolled_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_38_g2_: _dafny.Seq
                                d_39_i2_: bool
                                d_40_c2_: _dafny.Seq
                                out28_: _dafny.Seq
                                out29_: bool
                                out30_: _dafny.Seq
                                out28_, out29_, out30_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_38_g2_ = out28_
                                d_39_i2_ = out29_
                                d_40_c2_ = out30_
                                generated = d_38_g2_
                                insideConstrainedOut = d_39_i2_
                                currentConstrainedOut = d_40_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif (d_1_steps_) < (maxSteps):
                                d_41_constrainedPrompt_: _dafny.Seq
                                d_41_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_42_next_: _dafny.Seq
                                out31_: _dafny.Seq
                                out31_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_41_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_42_next_ = out31_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_42_next_) == (eosToken):
                                    d_43_gR2_: _dafny.Seq
                                    d_44_cR2_: _dafny.Seq
                                    out32_: _dafny.Seq
                                    out33_: _dafny.Seq
                                    out32_, out33_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_43_gR2_ = out32_
                                    d_44_cR2_ = out33_
                                    generated = d_43_gR2_
                                    currentConstrainedOut = d_44_cR2_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_45_g3_: _dafny.Seq
                                        d_46_i3_: bool
                                        d_47_c3_: _dafny.Seq
                                        out34_: _dafny.Seq
                                        out35_: bool
                                        out36_: _dafny.Seq
                                        out34_, out35_, out36_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_45_g3_ = out34_
                                        d_46_i3_ = out35_
                                        d_47_c3_ = out36_
                                        generated = d_45_g3_
                                        insideConstrainedOut = d_46_i3_
                                        currentConstrainedOut = d_47_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_48_g2_: _dafny.Seq
                                    d_49_i2_: bool
                                    d_50_c2_: _dafny.Seq
                                    out37_: _dafny.Seq
                                    out38_: bool
                                    out39_: _dafny.Seq
                                    out37_, out38_, out39_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_42_next_)
                                    d_48_g2_ = out37_
                                    d_49_i2_ = out38_
                                    d_50_c2_ = out39_
                                    generated = d_48_g2_
                                    insideConstrainedOut = d_49_i2_
                                    currentConstrainedOut = d_50_c2_
                                    d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_51_g3_: _dafny.Seq
                                        d_52_i3_: bool
                                        d_53_c3_: _dafny.Seq
                                        out40_: _dafny.Seq
                                        out41_: bool
                                        out42_: _dafny.Seq
                                        out40_, out41_, out42_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_51_g3_ = out40_
                                        d_52_i3_ = out41_
                                        d_53_c3_ = out42_
                                        generated = d_51_g3_
                                        insideConstrainedOut = d_52_i3_
                                        currentConstrainedOut = d_53_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                        elif True:
                            d_54_constrainedPrompt_: _dafny.Seq
                            d_54_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_55_next_: _dafny.Seq
                            out43_: _dafny.Seq
                            out43_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_54_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_55_next_ = out43_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_55_next_) == (eosToken):
                                d_56_gRolled_: _dafny.Seq
                                d_57_cRolled_: _dafny.Seq
                                out44_: _dafny.Seq
                                out45_: _dafny.Seq
                                out44_, out45_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_56_gRolled_ = out44_
                                d_57_cRolled_ = out45_
                                generated = d_56_gRolled_
                                currentConstrainedOut = d_57_cRolled_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_58_g2_: _dafny.Seq
                                    d_59_i2_: bool
                                    d_60_c2_: _dafny.Seq
                                    out46_: _dafny.Seq
                                    out47_: bool
                                    out48_: _dafny.Seq
                                    out46_, out47_, out48_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_58_g2_ = out46_
                                    d_59_i2_ = out47_
                                    d_60_c2_ = out48_
                                    generated = d_58_g2_
                                    insideConstrainedOut = d_59_i2_
                                    currentConstrainedOut = d_60_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_61_g2_: _dafny.Seq
                                d_62_i2_: bool
                                d_63_c2_: _dafny.Seq
                                out49_: _dafny.Seq
                                out50_: bool
                                out51_: _dafny.Seq
                                out49_, out50_, out51_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_55_next_)
                                d_61_g2_ = out49_
                                d_62_i2_ = out50_
                                d_63_c2_ = out51_
                                generated = d_61_g2_
                                insideConstrainedOut = d_62_i2_
                                currentConstrainedOut = d_63_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_64_gRolled_: _dafny.Seq
            d_65_cRolled_: _dafny.Seq
            out52_: _dafny.Seq
            out53_: _dafny.Seq
            out52_, out53_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_64_gRolled_ = out52_
            d_65_cRolled_ = out53_
            generated = d_64_gRolled_
            currentConstrainedOut = d_65_cRolled_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_66_constrainedPromptPost_: _dafny.Seq
                d_66_constrainedPromptPost_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_67_nextPost_: _dafny.Seq
                out54_: _dafny.Seq
                out54_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_66_constrainedPromptPost_, currentConstrainedOut, eosToken)
                d_67_nextPost_ = out54_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_67_nextPost_) != (eosToken):
                    d_68_gp_: _dafny.Seq
                    d_69_ip_: bool
                    d_70_cp_: _dafny.Seq
                    out55_: _dafny.Seq
                    out56_: bool
                    out57_: _dafny.Seq
                    out55_, out56_, out57_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_67_nextPost_)
                    d_68_gp_ = out55_
                    d_69_ip_ = out56_
                    d_70_cp_ = out57_
                    generated = d_68_gp_
                    insideConstrainedOut = d_69_ip_
                    currentConstrainedOut = d_70_cp_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_71_g2_: _dafny.Seq
                d_72_i2_: bool
                d_73_c2_: _dafny.Seq
                out58_: _dafny.Seq
                out59_: bool
                out60_: _dafny.Seq
                out58_, out59_, out60_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_71_g2_ = out58_
                d_72_i2_ = out59_
                d_73_c2_ = out60_
                generated = d_71_g2_
                insideConstrainedOut = d_72_i2_
                currentConstrainedOut = d_73_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

