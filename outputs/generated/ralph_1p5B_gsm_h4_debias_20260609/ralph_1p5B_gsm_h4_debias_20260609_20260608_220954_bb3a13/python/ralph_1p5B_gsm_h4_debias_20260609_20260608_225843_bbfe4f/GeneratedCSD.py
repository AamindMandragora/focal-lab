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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final numeric answer as a single expression inside << >> delimiters. Example: <<42>> or <<n1 + n2>>. Do not open << until you have the final answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 30
        d_3_spanTokensUsed_: int
        d_3_spanTokensUsed_ = 0
        d_4_spanMaxTokens_: int
        d_4_spanMaxTokens_ = 8
        d_5_hasSeenOpenSpan_: bool
        d_5_hasSeenOpenSpan_ = insideConstrained
        d_6_spanClosed_: bool
        d_6_spanClosed_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_6_spanClosed_:
                            raise _dafny.Break("0")
                        d_7_remaining_: int
                        d_7_remaining_ = (maxSteps) - (d_1_steps_)
                        if (((d_7_remaining_) <= (20)) and (not(d_5_hasSeenOpenSpan_))) and ((d_7_remaining_) > (2)):
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
                        d_6_spanClosed_ = True
                        raise _dafny.Break("0")
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
                            out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_26_gRolled_ = out17_
                            d_27_cRolled_ = out18_
                            generated = d_26_gRolled_
                            currentConstrainedOut = d_27_cRolled_
                            d_3_spanTokensUsed_ = 0
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_28_g2_: _dafny.Seq
                                d_29_i2_: bool
                                d_30_c2_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_28_g2_ = out19_
                                d_29_i2_ = out20_
                                d_30_c2_ = out21_
                                generated = d_28_g2_
                                insideConstrainedOut = d_29_i2_
                                currentConstrainedOut = d_30_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_spanTokensUsed_ = 0
                                d_6_spanClosed_ = True
                                raise _dafny.Break("0")
                            elif (d_1_steps_) < (maxSteps):
                                d_31_constrainedPrompt_: _dafny.Seq
                                d_31_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_32_next_: _dafny.Seq
                                out22_: _dafny.Seq
                                out22_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_31_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_32_next_ = out22_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_32_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_33_g2_: _dafny.Seq
                                    d_34_i2_: bool
                                    d_35_c2_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next_)
                                    d_33_g2_ = out23_
                                    d_34_i2_ = out24_
                                    d_35_c2_ = out25_
                                    generated = d_33_g2_
                                    insideConstrainedOut = d_34_i2_
                                    currentConstrainedOut = d_35_c2_
                                    d_3_spanTokensUsed_ = 1
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_36_g3_: _dafny.Seq
                                        d_37_i3_: bool
                                        d_38_c3_: _dafny.Seq
                                        out26_: _dafny.Seq
                                        out27_: bool
                                        out28_: _dafny.Seq
                                        out26_, out27_, out28_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_36_g3_ = out26_
                                        d_37_i3_ = out27_
                                        d_38_c3_ = out28_
                                        generated = d_36_g3_
                                        insideConstrainedOut = d_37_i3_
                                        currentConstrainedOut = d_38_c3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_3_spanTokensUsed_ = 0
                                        d_6_spanClosed_ = True
                                        raise _dafny.Break("0")
                        elif True:
                            d_39_constrainedPrompt_: _dafny.Seq
                            d_39_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_40_next_: _dafny.Seq
                            out29_: _dafny.Seq
                            out29_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_39_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_40_next_ = out29_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_40_next_) == (eosToken):
                                d_41_gRolled_: _dafny.Seq
                                d_42_cRolled_: _dafny.Seq
                                out30_: _dafny.Seq
                                out31_: _dafny.Seq
                                out30_, out31_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_41_gRolled_ = out30_
                                d_42_cRolled_ = out31_
                                generated = d_41_gRolled_
                                currentConstrainedOut = d_42_cRolled_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_43_g2_: _dafny.Seq
                                    d_44_i2_: bool
                                    d_45_c2_: _dafny.Seq
                                    out32_: _dafny.Seq
                                    out33_: bool
                                    out34_: _dafny.Seq
                                    out32_, out33_, out34_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_43_g2_ = out32_
                                    d_44_i2_ = out33_
                                    d_45_c2_ = out34_
                                    generated = d_43_g2_
                                    insideConstrainedOut = d_44_i2_
                                    currentConstrainedOut = d_45_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_6_spanClosed_ = True
                                    raise _dafny.Break("0")
                            elif True:
                                d_46_g2_: _dafny.Seq
                                d_47_i2_: bool
                                d_48_c2_: _dafny.Seq
                                out35_: _dafny.Seq
                                out36_: bool
                                out37_: _dafny.Seq
                                out35_, out36_, out37_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_40_next_)
                                d_46_g2_ = out35_
                                d_47_i2_ = out36_
                                d_48_c2_ = out37_
                                generated = d_46_g2_
                                insideConstrainedOut = d_47_i2_
                                currentConstrainedOut = d_48_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_49_gRolled_: _dafny.Seq
            d_50_cRolled_: _dafny.Seq
            out38_: _dafny.Seq
            out39_: _dafny.Seq
            out38_, out39_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_49_gRolled_ = out38_
            d_50_cRolled_ = out39_
            generated = d_49_gRolled_
            currentConstrainedOut = d_50_cRolled_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_51_constrainedPrompt_: _dafny.Seq
                d_51_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_52_next_: _dafny.Seq
                out40_: _dafny.Seq
                out40_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_51_constrainedPrompt_, currentConstrainedOut, eosToken)
                d_52_next_ = out40_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_52_next_) != (eosToken):
                    d_53_g2_: _dafny.Seq
                    d_54_i2_: bool
                    d_55_c2_: _dafny.Seq
                    out41_: _dafny.Seq
                    out42_: bool
                    out43_: _dafny.Seq
                    out41_, out42_, out43_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_52_next_)
                    d_53_g2_ = out41_
                    d_54_i2_ = out42_
                    d_55_c2_ = out43_
                    generated = d_53_g2_
                    insideConstrainedOut = d_54_i2_
                    currentConstrainedOut = d_55_c2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_56_g2_: _dafny.Seq
                d_57_i2_: bool
                d_58_c2_: _dafny.Seq
                out44_: _dafny.Seq
                out45_: bool
                out46_: _dafny.Seq
                out44_, out45_, out46_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_56_g2_ = out44_
                d_57_i2_ = out45_
                d_58_c2_ = out46_
                generated = d_56_g2_
                insideConstrainedOut = d_57_i2_
                currentConstrainedOut = d_58_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

