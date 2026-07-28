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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final numeric answer inside << >> delimiters. The content between << and >> must be a valid arithmetic expression using only numbers, variables, +, -, *, /, (, ). Example: <<42>> or <<n1 * r + b>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 30
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
                        if (((d_6_remaining_) <= (50)) and (not(d_5_hasSeenOpenSpan_))) and ((d_6_remaining_) > (3)):
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
                        elif (d_6_remaining_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            d_10_chunkBudget_: int
                            if (d_6_remaining_) < (d_2_freeChunkSize_):
                                d_10_chunkBudget_ = d_6_remaining_
                            elif True:
                                d_10_chunkBudget_ = d_2_freeChunkSize_
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
                        if (d_1_steps_) < (maxSteps):
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
                            raise _dafny.Break("0")
                    elif True:
                        d_24_isDeadEnd_: bool
                        out16_: bool
                        out16_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_24_isDeadEnd_ = out16_
                        if (d_24_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_)):
                            d_25_gRolled_: _dafny.Seq
                            d_26_cRolled_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: _dafny.Seq
                            out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_25_gRolled_ = out17_
                            d_26_cRolled_ = out18_
                            generated = d_25_gRolled_
                            currentConstrainedOut = d_26_cRolled_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_27_g2_: _dafny.Seq
                                d_28_i2_: bool
                                d_29_c2_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_27_g2_ = out19_
                                d_28_i2_ = out20_
                                d_29_c2_ = out21_
                                generated = d_27_g2_
                                insideConstrainedOut = d_28_i2_
                                currentConstrainedOut = d_29_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif (d_1_steps_) < (maxSteps):
                                d_30_constrainedPrompt_: _dafny.Seq
                                d_30_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_31_next_: _dafny.Seq
                                out22_: _dafny.Seq
                                out22_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_30_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_31_next_ = out22_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_31_next_) == (eosToken):
                                    d_32_gR2_: _dafny.Seq
                                    d_33_cR2_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: _dafny.Seq
                                    out23_, out24_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_32_gR2_ = out23_
                                    d_33_cR2_ = out24_
                                    generated = d_32_gR2_
                                    currentConstrainedOut = d_33_cR2_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_34_g2_: _dafny.Seq
                                        d_35_i2_: bool
                                        d_36_c2_: _dafny.Seq
                                        out25_: _dafny.Seq
                                        out26_: bool
                                        out27_: _dafny.Seq
                                        out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_34_g2_ = out25_
                                        d_35_i2_ = out26_
                                        d_36_c2_ = out27_
                                        generated = d_34_g2_
                                        insideConstrainedOut = d_35_i2_
                                        currentConstrainedOut = d_36_c2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                                elif True:
                                    d_37_valid_: bool
                                    out28_: bool
                                    out28_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_31_next_)
                                    d_37_valid_ = out28_
                                    if d_37_valid_:
                                        d_38_g2_: _dafny.Seq
                                        d_39_i2_: bool
                                        d_40_c2_: _dafny.Seq
                                        out29_: _dafny.Seq
                                        out30_: bool
                                        out31_: _dafny.Seq
                                        out29_, out30_, out31_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_next_)
                                        d_38_g2_ = out29_
                                        d_39_i2_ = out30_
                                        d_40_c2_ = out31_
                                        generated = d_38_g2_
                                        insideConstrainedOut = d_39_i2_
                                        currentConstrainedOut = d_40_c2_
                                        d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                            d_41_g3_: _dafny.Seq
                                            d_42_i3_: bool
                                            d_43_c3_: _dafny.Seq
                                            out32_: _dafny.Seq
                                            out33_: bool
                                            out34_: _dafny.Seq
                                            out32_, out33_, out34_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_41_g3_ = out32_
                                            d_42_i3_ = out33_
                                            d_43_c3_ = out34_
                                            generated = d_41_g3_
                                            insideConstrainedOut = d_42_i3_
                                            currentConstrainedOut = d_43_c3_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            d_3_spanTokensUsed_ = 0
                        elif True:
                            d_44_constrainedPrompt_: _dafny.Seq
                            d_44_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_45_next_: _dafny.Seq
                            out35_: _dafny.Seq
                            out35_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_44_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_45_next_ = out35_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_45_next_) == (eosToken):
                                d_46_gRolled_: _dafny.Seq
                                d_47_cRolled_: _dafny.Seq
                                out36_: _dafny.Seq
                                out37_: _dafny.Seq
                                out36_, out37_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_46_gRolled_ = out36_
                                d_47_cRolled_ = out37_
                                generated = d_46_gRolled_
                                currentConstrainedOut = d_47_cRolled_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_48_g2_: _dafny.Seq
                                    d_49_i2_: bool
                                    d_50_c2_: _dafny.Seq
                                    out38_: _dafny.Seq
                                    out39_: bool
                                    out40_: _dafny.Seq
                                    out38_, out39_, out40_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_48_g2_ = out38_
                                    d_49_i2_ = out39_
                                    d_50_c2_ = out40_
                                    generated = d_48_g2_
                                    insideConstrainedOut = d_49_i2_
                                    currentConstrainedOut = d_50_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_51_valid_: bool
                                out41_: bool
                                out41_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_45_next_)
                                d_51_valid_ = out41_
                                if d_51_valid_:
                                    d_52_g2_: _dafny.Seq
                                    d_53_i2_: bool
                                    d_54_c2_: _dafny.Seq
                                    out42_: _dafny.Seq
                                    out43_: bool
                                    out44_: _dafny.Seq
                                    out42_, out43_, out44_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_45_next_)
                                    d_52_g2_ = out42_
                                    d_53_i2_ = out43_
                                    d_54_c2_ = out44_
                                    generated = d_52_g2_
                                    insideConstrainedOut = d_53_i2_
                                    currentConstrainedOut = d_54_c2_
                                    d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_55_g3_: _dafny.Seq
                                        d_56_i3_: bool
                                        d_57_c3_: _dafny.Seq
                                        out45_: _dafny.Seq
                                        out46_: bool
                                        out47_: _dafny.Seq
                                        out45_, out46_, out47_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_55_g3_ = out45_
                                        d_56_i3_ = out46_
                                        d_57_c3_ = out47_
                                        generated = d_55_g3_
                                        insideConstrainedOut = d_56_i3_
                                        currentConstrainedOut = d_57_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_58_gRolled_: _dafny.Seq
            d_59_cRolled_: _dafny.Seq
            out48_: _dafny.Seq
            out49_: _dafny.Seq
            out48_, out49_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_58_gRolled_ = out48_
            d_59_cRolled_ = out49_
            generated = d_58_gRolled_
            currentConstrainedOut = d_59_cRolled_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_60_constrainedPrompt_: _dafny.Seq
                d_60_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_61_next_: _dafny.Seq
                out50_: _dafny.Seq
                out50_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_60_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_61_next_ = out50_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_61_next_) != (eosToken):
                    d_62_valid_: bool
                    out51_: bool
                    out51_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_61_next_)
                    d_62_valid_ = out51_
                    if d_62_valid_:
                        d_63_g2_: _dafny.Seq
                        d_64_i2_: bool
                        d_65_c2_: _dafny.Seq
                        out52_: _dafny.Seq
                        out53_: bool
                        out54_: _dafny.Seq
                        out52_, out53_, out54_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_61_next_)
                        d_63_g2_ = out52_
                        d_64_i2_ = out53_
                        d_65_c2_ = out54_
                        generated = d_63_g2_
                        insideConstrainedOut = d_64_i2_
                        currentConstrainedOut = d_65_c2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_66_g2_: _dafny.Seq
                d_67_i2_: bool
                d_68_c2_: _dafny.Seq
                out55_: _dafny.Seq
                out56_: bool
                out57_: _dafny.Seq
                out55_, out56_, out57_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_66_g2_ = out55_
                d_67_i2_ = out56_
                d_68_c2_ = out57_
                generated = d_66_g2_
                insideConstrainedOut = d_67_i2_
                currentConstrainedOut = d_68_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

