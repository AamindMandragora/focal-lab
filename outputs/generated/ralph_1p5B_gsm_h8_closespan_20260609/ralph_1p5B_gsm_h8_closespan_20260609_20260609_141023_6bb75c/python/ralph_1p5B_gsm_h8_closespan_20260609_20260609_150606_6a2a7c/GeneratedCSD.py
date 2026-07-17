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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write each calculation and the final numerical answer inside << >> delimiters. Example: <<n1 + n2 * n3>>. Final answer: <<42>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 10
        d_3_spanTokensUsed_: int
        d_3_spanTokensUsed_ = 0
        d_4_spanMaxTokens_: int
        d_4_spanMaxTokens_ = 12
        d_5_hasSeenOpenSpan_: bool
        d_5_hasSeenOpenSpan_ = insideConstrained
        d_6_totalFreeSteps_: int
        d_6_totalFreeSteps_ = 0
        d_7_maxFreeSteps_: int
        d_7_maxFreeSteps_ = 60
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_remaining_: int
                        d_8_remaining_ = (maxSteps) - (d_1_steps_)
                        d_9_shouldForceOpen_: bool
                        d_9_shouldForceOpen_ = (((d_6_totalFreeSteps_) >= (d_7_maxFreeSteps_)) and ((d_8_remaining_) > (2))) or ((((d_8_remaining_) <= (65)) and (not(d_5_hasSeenOpenSpan_))) and ((d_8_remaining_) > (2)))
                        if d_9_shouldForceOpen_:
                            d_10_g2_: _dafny.Seq
                            d_11_i2_: bool
                            d_12_c2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_g2_ = out0_
                            d_11_i2_ = out1_
                            d_12_c2_ = out2_
                            generated = d_10_g2_
                            insideConstrainedOut = d_11_i2_
                            currentConstrainedOut = d_12_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                            d_5_hasSeenOpenSpan_ = True
                        elif True:
                            d_13_chunkBudget_: int
                            if (d_8_remaining_) < (d_2_freeChunkSize_):
                                d_13_chunkBudget_ = d_8_remaining_
                            elif True:
                                d_13_chunkBudget_ = d_2_freeChunkSize_
                            if (d_13_chunkBudget_) == (0):
                                raise _dafny.Break("0")
                            d_14_chunkGenerated_: _dafny.Seq
                            d_15_stoppedOnOpenSpan_: bool
                            d_16_stoppedOnEos_: bool
                            d_17_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_13_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_14_chunkGenerated_ = out3_
                            d_15_stoppedOnOpenSpan_ = out4_
                            d_16_stoppedOnEos_ = out5_
                            d_17_stepsUsed_ = out6_
                            generated = d_14_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_17_stepsUsed_)
                            d_6_totalFreeSteps_ = (d_6_totalFreeSteps_) + (d_17_stepsUsed_)
                            if d_16_stoppedOnEos_:
                                if (not(d_5_hasSeenOpenSpan_)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                                    d_18_g2_: _dafny.Seq
                                    d_19_i2_: bool
                                    d_20_c2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_18_g2_ = out7_
                                    d_19_i2_ = out8_
                                    d_20_c2_ = out9_
                                    generated = d_18_g2_
                                    insideConstrainedOut = d_19_i2_
                                    currentConstrainedOut = d_20_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                                    d_5_hasSeenOpenSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif d_15_stoppedOnOpenSpan_:
                                d_21_g2_: _dafny.Seq
                                d_22_i2_: bool
                                d_23_c2_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_21_g2_ = out10_
                                d_22_i2_ = out11_
                                d_23_c2_ = out12_
                                generated = d_21_g2_
                                insideConstrainedOut = d_22_i2_
                                currentConstrainedOut = d_23_c2_
                                d_3_spanTokensUsed_ = 0
                                d_5_hasSeenOpenSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_24_g2_: _dafny.Seq
                        d_25_i2_: bool
                        d_26_c2_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_24_g2_ = out13_
                        d_25_i2_ = out14_
                        d_26_c2_ = out15_
                        generated = d_24_g2_
                        insideConstrainedOut = d_25_i2_
                        currentConstrainedOut = d_26_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokensUsed_ = 0
                    elif True:
                        d_27_isDeadEnd_: bool
                        out16_: bool
                        out16_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_27_isDeadEnd_ = out16_
                        if (d_27_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (d_4_spanMaxTokens_)):
                            d_28_gRolled_: _dafny.Seq
                            d_29_cRolled_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: _dafny.Seq
                            out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_28_gRolled_ = out17_
                            d_29_cRolled_ = out18_
                            generated = d_28_gRolled_
                            currentConstrainedOut = d_29_cRolled_
                            d_3_spanTokensUsed_ = 0
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                if (d_1_steps_) < (maxSteps):
                                    d_30_g2_: _dafny.Seq
                                    d_31_i2_: bool
                                    d_32_c2_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_30_g2_ = out19_
                                    d_31_i2_ = out20_
                                    d_32_c2_ = out21_
                                    generated = d_30_g2_
                                    insideConstrainedOut = d_31_i2_
                                    currentConstrainedOut = d_32_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                if (d_1_steps_) < (maxSteps):
                                    d_33_constrainedPrompt_: _dafny.Seq
                                    d_33_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_34_next_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out22_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_33_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_34_next_ = out22_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_34_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_35_g2_: _dafny.Seq
                                        d_36_i2_: bool
                                        d_37_c2_: _dafny.Seq
                                        out23_: _dafny.Seq
                                        out24_: bool
                                        out25_: _dafny.Seq
                                        out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_34_next_)
                                        d_35_g2_ = out23_
                                        d_36_i2_ = out24_
                                        d_37_c2_ = out25_
                                        generated = d_35_g2_
                                        insideConstrainedOut = d_36_i2_
                                        currentConstrainedOut = d_37_c2_
                                        d_3_spanTokensUsed_ = 1
                                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                            d_38_g3_: _dafny.Seq
                                            d_39_i3_: bool
                                            d_40_c3_: _dafny.Seq
                                            out26_: _dafny.Seq
                                            out27_: bool
                                            out28_: _dafny.Seq
                                            out26_, out27_, out28_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_38_g3_ = out26_
                                            d_39_i3_ = out27_
                                            d_40_c3_ = out28_
                                            generated = d_38_g3_
                                            insideConstrainedOut = d_39_i3_
                                            currentConstrainedOut = d_40_c3_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            d_3_spanTokensUsed_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                        elif True:
                            d_41_constrainedPrompt_: _dafny.Seq
                            d_41_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_42_next_: _dafny.Seq
                            d_43_wasConstrained_: bool
                            out29_: _dafny.Seq
                            out30_: bool
                            out29_, out30_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_41_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_42_next_ = out29_
                            d_43_wasConstrained_ = out30_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_42_next_) == (eosToken):
                                d_44_gRolled_: _dafny.Seq
                                d_45_cRolled_: _dafny.Seq
                                out31_: _dafny.Seq
                                out32_: _dafny.Seq
                                out31_, out32_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_44_gRolled_ = out31_
                                d_45_cRolled_ = out32_
                                generated = d_44_gRolled_
                                currentConstrainedOut = d_45_cRolled_
                                d_3_spanTokensUsed_ = 0
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    if (d_1_steps_) < (maxSteps):
                                        d_46_g2_: _dafny.Seq
                                        d_47_i2_: bool
                                        d_48_c2_: _dafny.Seq
                                        out33_: _dafny.Seq
                                        out34_: bool
                                        out35_: _dafny.Seq
                                        out33_, out34_, out35_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_46_g2_ = out33_
                                        d_47_i2_ = out34_
                                        d_48_c2_ = out35_
                                        generated = d_46_g2_
                                        insideConstrainedOut = d_47_i2_
                                        currentConstrainedOut = d_48_c2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    if (d_1_steps_) < (maxSteps):
                                        d_49_cPrompt2_: _dafny.Seq
                                        d_49_cPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                        d_50_next2_: _dafny.Seq
                                        out36_: _dafny.Seq
                                        out36_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_49_cPrompt2_, currentConstrainedOut, eosToken)
                                        d_50_next2_ = out36_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if (d_50_next2_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            d_51_g2_: _dafny.Seq
                                            d_52_i2_: bool
                                            d_53_c2_: _dafny.Seq
                                            out37_: _dafny.Seq
                                            out38_: bool
                                            out39_: _dafny.Seq
                                            out37_, out38_, out39_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_50_next2_)
                                            d_51_g2_ = out37_
                                            d_52_i2_ = out38_
                                            d_53_c2_ = out39_
                                            generated = d_51_g2_
                                            insideConstrainedOut = d_52_i2_
                                            currentConstrainedOut = d_53_c2_
                                            d_3_spanTokensUsed_ = 1
                                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                                d_54_g3_: _dafny.Seq
                                                d_55_i3_: bool
                                                d_56_c3_: _dafny.Seq
                                                out40_: _dafny.Seq
                                                out41_: bool
                                                out42_: _dafny.Seq
                                                out40_, out41_, out42_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_54_g3_ = out40_
                                                d_55_i3_ = out41_
                                                d_56_c3_ = out42_
                                                generated = d_54_g3_
                                                insideConstrainedOut = d_55_i3_
                                                currentConstrainedOut = d_56_c3_
                                                d_1_steps_ = (d_1_steps_) + (1)
                                                d_3_spanTokensUsed_ = 0
                                    elif True:
                                        raise _dafny.Break("0")
                            elif True:
                                d_57_g2_: _dafny.Seq
                                d_58_i2_: bool
                                d_59_c2_: _dafny.Seq
                                out43_: _dafny.Seq
                                out44_: bool
                                out45_: _dafny.Seq
                                out43_, out44_, out45_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_42_next_)
                                d_57_g2_ = out43_
                                d_58_i2_ = out44_
                                d_59_c2_ = out45_
                                generated = d_57_g2_
                                insideConstrainedOut = d_58_i2_
                                currentConstrainedOut = d_59_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_60_gRolled_: _dafny.Seq
            d_61_cRolled_: _dafny.Seq
            out46_: _dafny.Seq
            out47_: _dafny.Seq
            out46_, out47_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_60_gRolled_ = out46_
            d_61_cRolled_ = out47_
            generated = d_60_gRolled_
            currentConstrainedOut = d_61_cRolled_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_62_constrainedPrompt_: _dafny.Seq
                d_62_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_63_next_: _dafny.Seq
                out48_: _dafny.Seq
                out48_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_62_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_63_next_ = out48_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_63_next_) != (eosToken):
                    d_64_g2_: _dafny.Seq
                    d_65_i2_: bool
                    d_66_c2_: _dafny.Seq
                    out49_: _dafny.Seq
                    out50_: bool
                    out51_: _dafny.Seq
                    out49_, out50_, out51_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_63_next_)
                    d_64_g2_ = out49_
                    d_65_i2_ = out50_
                    d_66_c2_ = out51_
                    generated = d_64_g2_
                    insideConstrainedOut = d_65_i2_
                    currentConstrainedOut = d_66_c2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_67_g2_: _dafny.Seq
                d_68_i2_: bool
                d_69_c2_: _dafny.Seq
                out52_: _dafny.Seq
                out53_: bool
                out54_: _dafny.Seq
                out52_, out53_, out54_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_67_g2_ = out52_
                d_68_i2_ = out53_
                d_69_c2_ = out54_
                generated = d_67_g2_
                insideConstrainedOut = d_68_i2_
                currentConstrainedOut = d_69_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

