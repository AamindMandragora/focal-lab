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
        d_2_freeChunkSize_ = 20
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
                                if (not(d_5_hasSeenOpenSpan_)) and (((d_1_steps_) + (4)) <= (maxSteps)):
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
                        if d_24_isDeadEnd_:
                            d_25_remaining_: int
                            d_25_remaining_ = (maxSteps) - (d_1_steps_)
                            d_26_rbBudget_: int
                            if (d_25_remaining_) < (8):
                                d_26_rbBudget_ = d_25_remaining_
                            elif True:
                                d_26_rbBudget_ = 8
                            d_27_closeReserve_: int
                            if (d_26_rbBudget_) >= (2):
                                d_27_closeReserve_ = 2
                            elif True:
                                d_27_closeReserve_ = d_26_rbBudget_
                            if ((d_26_rbBudget_) > (0)) and ((d_27_closeReserve_) <= (d_26_rbBudget_)):
                                d_28_gRb_: _dafny.Seq
                                d_29_cRb_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: _dafny.Seq
                                out17_, out18_ = (d_0_helpers_).RollbackAndContinue(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_rbBudget_, d_27_closeReserve_, 3)
                                d_28_gRb_ = out17_
                                d_29_cRb_ = out18_
                                d_30_rbCost_: int
                                if ((d_26_rbBudget_) - (d_27_closeReserve_)) <= (d_26_rbBudget_):
                                    d_30_rbCost_ = (d_26_rbBudget_) - (d_27_closeReserve_)
                                elif True:
                                    d_30_rbCost_ = d_26_rbBudget_
                                d_1_steps_ = (d_1_steps_) + ((d_26_rbBudget_) - (d_27_closeReserve_))
                                generated = d_28_gRb_
                                currentConstrainedOut = d_29_cRb_
                                d_3_spanTokensUsed_ = len(d_29_cRb_)
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_31_g2_: _dafny.Seq
                                    d_32_i2_: bool
                                    d_33_c2_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_31_g2_ = out19_
                                    d_32_i2_ = out20_
                                    d_33_c2_ = out21_
                                    generated = d_31_g2_
                                    insideConstrainedOut = d_32_i2_
                                    currentConstrainedOut = d_33_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                                elif (d_1_steps_) < (maxSteps):
                                    d_34_constrainedPrompt_: _dafny.Seq
                                    d_34_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_35_next_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out22_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_34_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_35_next_ = out22_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_35_next_) != (eosToken):
                                        d_36_g2_: _dafny.Seq
                                        d_37_i2_: bool
                                        d_38_c2_: _dafny.Seq
                                        out23_: _dafny.Seq
                                        out24_: bool
                                        out25_: _dafny.Seq
                                        out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_35_next_)
                                        d_36_g2_ = out23_
                                        d_37_i2_ = out24_
                                        d_38_c2_ = out25_
                                        generated = d_36_g2_
                                        insideConstrainedOut = d_37_i2_
                                        currentConstrainedOut = d_38_c2_
                                        d_3_spanTokensUsed_ = len(d_38_c2_)
                                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                            d_39_g3_: _dafny.Seq
                                            d_40_i3_: bool
                                            d_41_c3_: _dafny.Seq
                                            out26_: _dafny.Seq
                                            out27_: bool
                                            out28_: _dafny.Seq
                                            out26_, out27_, out28_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_39_g3_ = out26_
                                            d_40_i3_ = out27_
                                            d_41_c3_ = out28_
                                            generated = d_39_g3_
                                            insideConstrainedOut = d_40_i3_
                                            currentConstrainedOut = d_41_c3_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            d_3_spanTokensUsed_ = 0
                            elif True:
                                d_42_gRolled_: _dafny.Seq
                                d_43_cRolled_: _dafny.Seq
                                out29_: _dafny.Seq
                                out30_: _dafny.Seq
                                out29_, out30_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_42_gRolled_ = out29_
                                d_43_cRolled_ = out30_
                                generated = d_42_gRolled_
                                currentConstrainedOut = d_43_cRolled_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_44_g2_: _dafny.Seq
                                    d_45_i2_: bool
                                    d_46_c2_: _dafny.Seq
                                    out31_: _dafny.Seq
                                    out32_: bool
                                    out33_: _dafny.Seq
                                    out31_, out32_, out33_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_44_g2_ = out31_
                                    d_45_i2_ = out32_
                                    d_46_c2_ = out33_
                                    generated = d_44_g2_
                                    insideConstrainedOut = d_45_i2_
                                    currentConstrainedOut = d_46_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                        elif (d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_):
                            d_47_gRolled_: _dafny.Seq
                            d_48_cRolled_: _dafny.Seq
                            out34_: _dafny.Seq
                            out35_: _dafny.Seq
                            out34_, out35_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_47_gRolled_ = out34_
                            d_48_cRolled_ = out35_
                            generated = d_47_gRolled_
                            currentConstrainedOut = d_48_cRolled_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_49_g2_: _dafny.Seq
                                d_50_i2_: bool
                                d_51_c2_: _dafny.Seq
                                out36_: _dafny.Seq
                                out37_: bool
                                out38_: _dafny.Seq
                                out36_, out37_, out38_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_49_g2_ = out36_
                                d_50_i2_ = out37_
                                d_51_c2_ = out38_
                                generated = d_49_g2_
                                insideConstrainedOut = d_50_i2_
                                currentConstrainedOut = d_51_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif (d_1_steps_) < (maxSteps):
                                d_52_constrainedPrompt_: _dafny.Seq
                                d_52_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_53_next_: _dafny.Seq
                                out39_: _dafny.Seq
                                out39_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_52_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_53_next_ = out39_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_53_next_) != (eosToken):
                                    d_54_g2_: _dafny.Seq
                                    d_55_i2_: bool
                                    d_56_c2_: _dafny.Seq
                                    out40_: _dafny.Seq
                                    out41_: bool
                                    out42_: _dafny.Seq
                                    out40_, out41_, out42_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_53_next_)
                                    d_54_g2_ = out40_
                                    d_55_i2_ = out41_
                                    d_56_c2_ = out42_
                                    generated = d_54_g2_
                                    insideConstrainedOut = d_55_i2_
                                    currentConstrainedOut = d_56_c2_
                                    d_3_spanTokensUsed_ = 1
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_57_g3_: _dafny.Seq
                                        d_58_i3_: bool
                                        d_59_c3_: _dafny.Seq
                                        out43_: _dafny.Seq
                                        out44_: bool
                                        out45_: _dafny.Seq
                                        out43_, out44_, out45_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_57_g3_ = out43_
                                        d_58_i3_ = out44_
                                        d_59_c3_ = out45_
                                        generated = d_57_g3_
                                        insideConstrainedOut = d_58_i3_
                                        currentConstrainedOut = d_59_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                        elif True:
                            d_60_constrainedPrompt_: _dafny.Seq
                            d_60_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_61_next_: _dafny.Seq
                            d_62_wasConstrained_: bool
                            out46_: _dafny.Seq
                            out47_: bool
                            out46_, out47_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_60_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_61_next_ = out46_
                            d_62_wasConstrained_ = out47_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_61_next_) == (eosToken):
                                d_63_gRolled_: _dafny.Seq
                                d_64_cRolled_: _dafny.Seq
                                out48_: _dafny.Seq
                                out49_: _dafny.Seq
                                out48_, out49_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_63_gRolled_ = out48_
                                d_64_cRolled_ = out49_
                                generated = d_63_gRolled_
                                currentConstrainedOut = d_64_cRolled_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_65_g2_: _dafny.Seq
                                    d_66_i2_: bool
                                    d_67_c2_: _dafny.Seq
                                    out50_: _dafny.Seq
                                    out51_: bool
                                    out52_: _dafny.Seq
                                    out50_, out51_, out52_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_65_g2_ = out50_
                                    d_66_i2_ = out51_
                                    d_67_c2_ = out52_
                                    generated = d_65_g2_
                                    insideConstrainedOut = d_66_i2_
                                    currentConstrainedOut = d_67_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif (d_1_steps_) < (maxSteps):
                                    d_68_cPrompt2_: _dafny.Seq
                                    d_68_cPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_69_next2_: _dafny.Seq
                                    out53_: _dafny.Seq
                                    out53_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_68_cPrompt2_, currentConstrainedOut, eosToken)
                                    d_69_next2_ = out53_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_69_next2_) != (eosToken):
                                        d_70_g2_: _dafny.Seq
                                        d_71_i2_: bool
                                        d_72_c2_: _dafny.Seq
                                        out54_: _dafny.Seq
                                        out55_: bool
                                        out56_: _dafny.Seq
                                        out54_, out55_, out56_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_69_next2_)
                                        d_70_g2_ = out54_
                                        d_71_i2_ = out55_
                                        d_72_c2_ = out56_
                                        generated = d_70_g2_
                                        insideConstrainedOut = d_71_i2_
                                        currentConstrainedOut = d_72_c2_
                                        d_3_spanTokensUsed_ = 1
                                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                            d_73_g3_: _dafny.Seq
                                            d_74_i3_: bool
                                            d_75_c3_: _dafny.Seq
                                            out57_: _dafny.Seq
                                            out58_: bool
                                            out59_: _dafny.Seq
                                            out57_, out58_, out59_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_73_g3_ = out57_
                                            d_74_i3_ = out58_
                                            d_75_c3_ = out59_
                                            generated = d_73_g3_
                                            insideConstrainedOut = d_74_i3_
                                            currentConstrainedOut = d_75_c3_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            d_3_spanTokensUsed_ = 0
                            elif True:
                                d_76_g2_: _dafny.Seq
                                d_77_i2_: bool
                                d_78_c2_: _dafny.Seq
                                out60_: _dafny.Seq
                                out61_: bool
                                out62_: _dafny.Seq
                                out60_, out61_, out62_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_61_next_)
                                d_76_g2_ = out60_
                                d_77_i2_ = out61_
                                d_78_c2_ = out62_
                                generated = d_76_g2_
                                insideConstrainedOut = d_77_i2_
                                currentConstrainedOut = d_78_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_79_g3_: _dafny.Seq
                                    d_80_i3_: bool
                                    d_81_c3_: _dafny.Seq
                                    out63_: _dafny.Seq
                                    out64_: bool
                                    out65_: _dafny.Seq
                                    out63_, out64_, out65_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_79_g3_ = out63_
                                    d_80_i3_ = out64_
                                    d_81_c3_ = out65_
                                    generated = d_79_g3_
                                    insideConstrainedOut = d_80_i3_
                                    currentConstrainedOut = d_81_c3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_82_gRolled_: _dafny.Seq
            d_83_cRolled_: _dafny.Seq
            out66_: _dafny.Seq
            out67_: _dafny.Seq
            out66_, out67_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_82_gRolled_ = out66_
            d_83_cRolled_ = out67_
            generated = d_82_gRolled_
            currentConstrainedOut = d_83_cRolled_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_84_constrainedPrompt_: _dafny.Seq
                d_84_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_85_next_: _dafny.Seq
                out68_: _dafny.Seq
                out68_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_84_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_85_next_ = out68_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_85_next_) != (eosToken):
                    d_86_g2_: _dafny.Seq
                    d_87_i2_: bool
                    d_88_c2_: _dafny.Seq
                    out69_: _dafny.Seq
                    out70_: bool
                    out71_: _dafny.Seq
                    out69_, out70_, out71_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_85_next_)
                    d_86_g2_ = out69_
                    d_87_i2_ = out70_
                    d_88_c2_ = out71_
                    generated = d_86_g2_
                    insideConstrainedOut = d_87_i2_
                    currentConstrainedOut = d_88_c2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_89_g2_: _dafny.Seq
                d_90_i2_: bool
                d_91_c2_: _dafny.Seq
                out72_: _dafny.Seq
                out73_: bool
                out74_: _dafny.Seq
                out72_, out73_, out74_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_89_g2_ = out72_
                d_90_i2_ = out73_
                d_91_c2_ = out74_
                generated = d_89_g2_
                insideConstrainedOut = d_90_i2_
                currentConstrainedOut = d_91_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

