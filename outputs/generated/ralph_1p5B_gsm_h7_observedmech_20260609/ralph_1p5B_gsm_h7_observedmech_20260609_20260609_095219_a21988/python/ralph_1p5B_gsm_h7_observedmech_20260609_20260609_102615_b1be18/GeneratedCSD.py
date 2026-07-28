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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final numeric answer expression inside << >> delimiters. Example: <<n1 * p1 + n2 * p2>>. Keep the expression inside << >> compact and complete.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 25
        d_3_spanTokensUsed_: int
        d_3_spanTokensUsed_ = 0
        d_4_spanMaxTokens_: int
        d_4_spanMaxTokens_ = 8
        d_5_hasSeenOpenSpan_: bool
        d_5_hasSeenOpenSpan_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remaining_: int
                        d_6_remaining_ = (maxSteps) - (d_1_steps_)
                        if (((d_6_remaining_) <= (60)) and (not(d_5_hasSeenOpenSpan_))) and ((d_6_remaining_) > (2)):
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
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                if (d_1_steps_) < (maxSteps):
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
                            elif True:
                                if (d_1_steps_) < (maxSteps):
                                    d_30_constrainedPrompt_: _dafny.Seq
                                    d_30_constrainedPrompt_ = (prompt) + (generated)
                                    d_31_next_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out22_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_30_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_31_next_ = out22_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_31_next_) == (eosToken):
                                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                            d_32_g2_: _dafny.Seq
                                            d_33_i2_: bool
                                            d_34_c2_: _dafny.Seq
                                            out23_: _dafny.Seq
                                            out24_: bool
                                            out25_: _dafny.Seq
                                            out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_32_g2_ = out23_
                                            d_33_i2_ = out24_
                                            d_34_c2_ = out25_
                                            generated = d_32_g2_
                                            insideConstrainedOut = d_33_i2_
                                            currentConstrainedOut = d_34_c2_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        d_35_g2_: _dafny.Seq
                                        d_36_i2_: bool
                                        d_37_c2_: _dafny.Seq
                                        out26_: _dafny.Seq
                                        out27_: bool
                                        out28_: _dafny.Seq
                                        out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_next_)
                                        d_35_g2_ = out26_
                                        d_36_i2_ = out27_
                                        d_37_c2_ = out28_
                                        generated = d_35_g2_
                                        insideConstrainedOut = d_36_i2_
                                        currentConstrainedOut = d_37_c2_
                                        d_3_spanTokensUsed_ = 1
                                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                            d_38_g3_: _dafny.Seq
                                            d_39_i3_: bool
                                            d_40_c3_: _dafny.Seq
                                            out29_: _dafny.Seq
                                            out30_: bool
                                            out31_: _dafny.Seq
                                            out29_, out30_, out31_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_38_g3_ = out29_
                                            d_39_i3_ = out30_
                                            d_40_c3_ = out31_
                                            generated = d_38_g3_
                                            insideConstrainedOut = d_39_i3_
                                            currentConstrainedOut = d_40_c3_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            d_3_spanTokensUsed_ = 0
                        elif True:
                            d_41_constrainedPrompt_: _dafny.Seq
                            d_41_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_42_next_: _dafny.Seq
                            out32_: _dafny.Seq
                            out32_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_41_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_42_next_ = out32_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_42_next_) == (eosToken):
                                d_43_gRolled_: _dafny.Seq
                                d_44_cRolled_: _dafny.Seq
                                out33_: _dafny.Seq
                                out34_: _dafny.Seq
                                out33_, out34_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_43_gRolled_ = out33_
                                d_44_cRolled_ = out34_
                                generated = d_43_gRolled_
                                currentConstrainedOut = d_44_cRolled_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_45_g2_: _dafny.Seq
                                    d_46_i2_: bool
                                    d_47_c2_: _dafny.Seq
                                    out35_: _dafny.Seq
                                    out36_: bool
                                    out37_: _dafny.Seq
                                    out35_, out36_, out37_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_45_g2_ = out35_
                                    d_46_i2_ = out36_
                                    d_47_c2_ = out37_
                                    generated = d_45_g2_
                                    insideConstrainedOut = d_46_i2_
                                    currentConstrainedOut = d_47_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_48_g2_: _dafny.Seq
                                d_49_i2_: bool
                                d_50_c2_: _dafny.Seq
                                out38_: _dafny.Seq
                                out39_: bool
                                out40_: _dafny.Seq
                                out38_, out39_, out40_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_42_next_)
                                d_48_g2_ = out38_
                                d_49_i2_ = out39_
                                d_50_c2_ = out40_
                                generated = d_48_g2_
                                insideConstrainedOut = d_49_i2_
                                currentConstrainedOut = d_50_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_51_g3_: _dafny.Seq
                                    d_52_i3_: bool
                                    d_53_c3_: _dafny.Seq
                                    out41_: _dafny.Seq
                                    out42_: bool
                                    out43_: _dafny.Seq
                                    out41_, out42_, out43_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_51_g3_ = out41_
                                    d_52_i3_ = out42_
                                    d_53_c3_ = out43_
                                    generated = d_51_g3_
                                    insideConstrainedOut = d_52_i3_
                                    currentConstrainedOut = d_53_c3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_54_gRolled_: _dafny.Seq
            d_55_cRolled_: _dafny.Seq
            out44_: _dafny.Seq
            out45_: _dafny.Seq
            out44_, out45_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_54_gRolled_ = out44_
            d_55_cRolled_ = out45_
            generated = d_54_gRolled_
            currentConstrainedOut = d_55_cRolled_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_56_g2_: _dafny.Seq
                d_57_i2_: bool
                d_58_c2_: _dafny.Seq
                out46_: _dafny.Seq
                out47_: bool
                out48_: _dafny.Seq
                out46_, out47_, out48_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_56_g2_ = out46_
                d_57_i2_ = out47_
                d_58_c2_ = out48_
                generated = d_56_g2_
                insideConstrainedOut = d_57_i2_
                currentConstrainedOut = d_58_c2_
                d_1_steps_ = (d_1_steps_) + (1)
            elif ((d_1_steps_) + (2)) <= (maxSteps):
                d_59_constrainedPrompt_: _dafny.Seq
                d_59_constrainedPrompt_ = (prompt) + (generated)
                d_60_next_: _dafny.Seq
                out49_: _dafny.Seq
                out49_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_59_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_60_next_ = out49_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_60_next_) != (eosToken):
                    d_61_g2_: _dafny.Seq
                    d_62_i2_: bool
                    d_63_c2_: _dafny.Seq
                    out50_: _dafny.Seq
                    out51_: bool
                    out52_: _dafny.Seq
                    out50_, out51_, out52_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_60_next_)
                    d_61_g2_ = out50_
                    d_62_i2_ = out51_
                    d_63_c2_ = out52_
                    generated = d_61_g2_
                    insideConstrainedOut = d_62_i2_
                    currentConstrainedOut = d_63_c2_
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                        d_64_g3_: _dafny.Seq
                        d_65_i3_: bool
                        d_66_c3_: _dafny.Seq
                        out53_: _dafny.Seq
                        out54_: bool
                        out55_: _dafny.Seq
                        out53_, out54_, out55_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_64_g3_ = out53_
                        d_65_i3_ = out54_
                        d_66_c3_ = out55_
                        generated = d_64_g3_
                        insideConstrainedOut = d_65_i3_
                        currentConstrainedOut = d_66_c3_
                        d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

