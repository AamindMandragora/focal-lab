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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final numeric answer inside << >> delimiters. Example: <<n1 * p1>>. Use only one final << >> span for the answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 25
        d_3_spanTokensUsed_: int
        d_3_spanTokensUsed_ = 0
        d_4_spanMaxTokens_: int
        d_4_spanMaxTokens_ = 16
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
                        if (d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_):
                            d_24_gRolled_: _dafny.Seq
                            d_25_cRolled_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: _dafny.Seq
                            out16_, out17_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_24_gRolled_ = out16_
                            d_25_cRolled_ = out17_
                            generated = d_24_gRolled_
                            currentConstrainedOut = d_25_cRolled_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_26_g2_: _dafny.Seq
                                d_27_i2_: bool
                                d_28_c2_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_26_g2_ = out18_
                                d_27_i2_ = out19_
                                d_28_c2_ = out20_
                                generated = d_26_g2_
                                insideConstrainedOut = d_27_i2_
                                currentConstrainedOut = d_28_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif (d_1_steps_) < (maxSteps):
                                d_29_constrainedPrompt_: _dafny.Seq
                                d_29_constrainedPrompt_ = (prompt) + (generated)
                                d_30_next_: _dafny.Seq
                                out21_: _dafny.Seq
                                out21_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_30_next_ = out21_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_30_next_) == (eosToken):
                                    pass
                                elif True:
                                    d_31_g2_: _dafny.Seq
                                    d_32_i2_: bool
                                    d_33_c2_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out23_: bool
                                    out24_: _dafny.Seq
                                    out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                                    d_31_g2_ = out22_
                                    d_32_i2_ = out23_
                                    d_33_c2_ = out24_
                                    generated = d_31_g2_
                                    insideConstrainedOut = d_32_i2_
                                    currentConstrainedOut = d_33_c2_
                                    d_3_spanTokensUsed_ = 1
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_34_g3_: _dafny.Seq
                                        d_35_i3_: bool
                                        d_36_c3_: _dafny.Seq
                                        out25_: _dafny.Seq
                                        out26_: bool
                                        out27_: _dafny.Seq
                                        out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_34_g3_ = out25_
                                        d_35_i3_ = out26_
                                        d_36_c3_ = out27_
                                        generated = d_34_g3_
                                        insideConstrainedOut = d_35_i3_
                                        currentConstrainedOut = d_36_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                        elif True:
                            d_37_constrainedPrompt_: _dafny.Seq
                            d_37_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_38_next_: _dafny.Seq
                            d_39_wasConstrained_: bool
                            out28_: _dafny.Seq
                            out29_: bool
                            out28_, out29_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_37_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_38_next_ = out28_
                            d_39_wasConstrained_ = out29_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_38_next_) == (eosToken):
                                d_40_gRolled_: _dafny.Seq
                                d_41_cRolled_: _dafny.Seq
                                out30_: _dafny.Seq
                                out31_: _dafny.Seq
                                out30_, out31_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_40_gRolled_ = out30_
                                d_41_cRolled_ = out31_
                                generated = d_40_gRolled_
                                currentConstrainedOut = d_41_cRolled_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_42_g2_: _dafny.Seq
                                    d_43_i2_: bool
                                    d_44_c2_: _dafny.Seq
                                    out32_: _dafny.Seq
                                    out33_: bool
                                    out34_: _dafny.Seq
                                    out32_, out33_, out34_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_42_g2_ = out32_
                                    d_43_i2_ = out33_
                                    d_44_c2_ = out34_
                                    generated = d_42_g2_
                                    insideConstrainedOut = d_43_i2_
                                    currentConstrainedOut = d_44_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_45_g2_: _dafny.Seq
                                d_46_i2_: bool
                                d_47_c2_: _dafny.Seq
                                out35_: _dafny.Seq
                                out36_: bool
                                out37_: _dafny.Seq
                                out35_, out36_, out37_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_38_next_)
                                d_45_g2_ = out35_
                                d_46_i2_ = out36_
                                d_47_c2_ = out37_
                                generated = d_45_g2_
                                insideConstrainedOut = d_46_i2_
                                currentConstrainedOut = d_47_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
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
                                    d_3_spanTokensUsed_ = 0
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_51_gRolled_: _dafny.Seq
            d_52_cRolled_: _dafny.Seq
            out41_: _dafny.Seq
            out42_: _dafny.Seq
            out41_, out42_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_51_gRolled_ = out41_
            d_52_cRolled_ = out42_
            generated = d_51_gRolled_
            currentConstrainedOut = d_52_cRolled_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_53_g2_: _dafny.Seq
                d_54_i2_: bool
                d_55_c2_: _dafny.Seq
                out43_: _dafny.Seq
                out44_: bool
                out45_: _dafny.Seq
                out43_, out44_, out45_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_53_g2_ = out43_
                d_54_i2_ = out44_
                d_55_c2_ = out45_
                generated = d_53_g2_
                insideConstrainedOut = d_54_i2_
                currentConstrainedOut = d_55_c2_
                d_1_steps_ = (d_1_steps_) + (1)
            elif ((d_1_steps_) + (2)) <= (maxSteps):
                d_56_constrainedPrompt_: _dafny.Seq
                d_56_constrainedPrompt_ = (prompt) + (generated)
                d_57_next_: _dafny.Seq
                out46_: _dafny.Seq
                out46_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_56_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_57_next_ = out46_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_57_next_) != (eosToken):
                    d_58_g2_: _dafny.Seq
                    d_59_i2_: bool
                    d_60_c2_: _dafny.Seq
                    out47_: _dafny.Seq
                    out48_: bool
                    out49_: _dafny.Seq
                    out47_, out48_, out49_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_57_next_)
                    d_58_g2_ = out47_
                    d_59_i2_ = out48_
                    d_60_c2_ = out49_
                    generated = d_58_g2_
                    insideConstrainedOut = d_59_i2_
                    currentConstrainedOut = d_60_c2_
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                        d_61_g3_: _dafny.Seq
                        d_62_i3_: bool
                        d_63_c3_: _dafny.Seq
                        out50_: _dafny.Seq
                        out51_: bool
                        out52_: _dafny.Seq
                        out50_, out51_, out52_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_61_g3_ = out50_
                        d_62_i3_ = out51_
                        d_63_c3_ = out52_
                        generated = d_61_g3_
                        insideConstrainedOut = d_62_i3_
                        currentConstrainedOut = d_63_c3_
                        d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

