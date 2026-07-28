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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final answer inside << >> delimiters. Example: <<42>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_hasSeenOpenSpan_: bool
        d_2_hasSeenOpenSpan_ = insideConstrained
        d_3_spanTokensUsed_: int
        d_3_spanTokensUsed_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_4_remaining_) == (0):
                            raise _dafny.Break("0")
                        if ((d_4_remaining_) <= (50)) and (not(d_2_hasSeenOpenSpan_)):
                            d_5_g2_: _dafny.Seq
                            d_6_i2_: bool
                            d_7_c2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_g2_ = out0_
                            d_6_i2_ = out1_
                            d_7_c2_ = out2_
                            generated = d_5_g2_
                            insideConstrainedOut = d_6_i2_
                            currentConstrainedOut = d_7_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokensUsed_ = 0
                            d_2_hasSeenOpenSpan_ = True
                        elif True:
                            d_8_chunkBudget_: int
                            if (d_4_remaining_) < (30):
                                d_8_chunkBudget_ = d_4_remaining_
                            elif True:
                                d_8_chunkBudget_ = 30
                            d_9_chunkGenerated_: _dafny.Seq
                            d_10_stoppedOnOpenSpan_: bool
                            d_11_stoppedOnEos_: bool
                            d_12_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_chunkGenerated_ = out3_
                            d_10_stoppedOnOpenSpan_ = out4_
                            d_11_stoppedOnEos_ = out5_
                            d_12_stepsUsed_ = out6_
                            generated = d_9_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                            if d_11_stoppedOnEos_:
                                if (not(d_2_hasSeenOpenSpan_)) and (((d_1_steps_) + (2)) <= (maxSteps)):
                                    d_13_g2_: _dafny.Seq
                                    d_14_i2_: bool
                                    d_15_c2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_13_g2_ = out7_
                                    d_14_i2_ = out8_
                                    d_15_c2_ = out9_
                                    generated = d_13_g2_
                                    insideConstrainedOut = d_14_i2_
                                    currentConstrainedOut = d_15_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                                    d_2_hasSeenOpenSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif d_10_stoppedOnOpenSpan_:
                                d_16_g2_: _dafny.Seq
                                d_17_i2_: bool
                                d_18_c2_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_16_g2_ = out10_
                                d_17_i2_ = out11_
                                d_18_c2_ = out12_
                                generated = d_16_g2_
                                insideConstrainedOut = d_17_i2_
                                currentConstrainedOut = d_18_c2_
                                d_3_spanTokensUsed_ = 0
                                d_2_hasSeenOpenSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        if (d_1_steps_) >= (maxSteps):
                            raise _dafny.Break("0")
                        d_19_g2_: _dafny.Seq
                        d_20_i2_: bool
                        d_21_c2_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_19_g2_ = out13_
                        d_20_i2_ = out14_
                        d_21_c2_ = out15_
                        generated = d_19_g2_
                        insideConstrainedOut = d_20_i2_
                        currentConstrainedOut = d_21_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokensUsed_ = 0
                    elif True:
                        d_22_isDeadEnd_: bool
                        out16_: bool
                        out16_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_22_isDeadEnd_ = out16_
                        if (d_22_isDeadEnd_) or ((d_3_spanTokensUsed_) >= (15)):
                            d_23_gR_: _dafny.Seq
                            d_24_cR_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: _dafny.Seq
                            out17_, out18_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_23_gR_ = out17_
                            d_24_cR_ = out18_
                            generated = d_23_gR_
                            currentConstrainedOut = d_24_cR_
                            d_3_spanTokensUsed_ = 0
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_25_gR2_: _dafny.Seq
                                d_26_cR2_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: _dafny.Seq
                                out19_, out20_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_25_gR2_ = out19_
                                d_26_cR2_ = out20_
                                generated = d_25_gR2_
                                currentConstrainedOut = d_26_cR2_
                            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                d_27_gR3_: _dafny.Seq
                                d_28_cR3_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: _dafny.Seq
                                out21_, out22_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_27_gR3_ = out21_
                                d_28_cR3_ = out22_
                                generated = d_27_gR3_
                                currentConstrainedOut = d_28_cR3_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_29_g2_: _dafny.Seq
                                d_30_i2_: bool
                                d_31_c2_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: bool
                                out25_: _dafny.Seq
                                out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_29_g2_ = out23_
                                d_30_i2_ = out24_
                                d_31_c2_ = out25_
                                generated = d_29_g2_
                                insideConstrainedOut = d_30_i2_
                                currentConstrainedOut = d_31_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_spanTokensUsed_ = 0
                            elif (d_1_steps_) < (maxSteps):
                                d_32_next_: _dafny.Seq
                                out26_: _dafny.Seq
                                out26_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_32_next_ = out26_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_32_next_) != (eosToken):
                                    d_33_g2_: _dafny.Seq
                                    d_34_i2_: bool
                                    d_35_c2_: _dafny.Seq
                                    out27_: _dafny.Seq
                                    out28_: bool
                                    out29_: _dafny.Seq
                                    out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next_)
                                    d_33_g2_ = out27_
                                    d_34_i2_ = out28_
                                    d_35_c2_ = out29_
                                    generated = d_33_g2_
                                    insideConstrainedOut = d_34_i2_
                                    currentConstrainedOut = d_35_c2_
                                    d_3_spanTokensUsed_ = 1
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_36_next_: _dafny.Seq
                            d_37_wasConstrained_: bool
                            out30_: _dafny.Seq
                            out31_: bool
                            out30_, out31_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_36_next_ = out30_
                            d_37_wasConstrained_ = out31_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_36_next_) == (eosToken):
                                d_38_gR_: _dafny.Seq
                                d_39_cR_: _dafny.Seq
                                out32_: _dafny.Seq
                                out33_: _dafny.Seq
                                out32_, out33_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_38_gR_ = out32_
                                d_39_cR_ = out33_
                                generated = d_38_gR_
                                currentConstrainedOut = d_39_cR_
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                                    d_40_gR2_: _dafny.Seq
                                    d_41_cR2_: _dafny.Seq
                                    out34_: _dafny.Seq
                                    out35_: _dafny.Seq
                                    out34_, out35_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_40_gR2_ = out34_
                                    d_41_cR2_ = out35_
                                    generated = d_40_gR2_
                                    currentConstrainedOut = d_41_cR2_
                                d_3_spanTokensUsed_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_42_g2_: _dafny.Seq
                                    d_43_i2_: bool
                                    d_44_c2_: _dafny.Seq
                                    out36_: _dafny.Seq
                                    out37_: bool
                                    out38_: _dafny.Seq
                                    out36_, out37_, out38_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_42_g2_ = out36_
                                    d_43_i2_ = out37_
                                    d_44_c2_ = out38_
                                    generated = d_42_g2_
                                    insideConstrainedOut = d_43_i2_
                                    currentConstrainedOut = d_44_c2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokensUsed_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_45_g2_: _dafny.Seq
                                d_46_i2_: bool
                                d_47_c2_: _dafny.Seq
                                out39_: _dafny.Seq
                                out40_: bool
                                out41_: _dafny.Seq
                                out39_, out40_, out41_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_36_next_)
                                d_45_g2_ = out39_
                                d_46_i2_ = out40_
                                d_47_c2_ = out41_
                                generated = d_45_g2_
                                insideConstrainedOut = d_46_i2_
                                currentConstrainedOut = d_47_c2_
                                d_3_spanTokensUsed_ = (d_3_spanTokensUsed_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_48_gR_: _dafny.Seq
            d_49_cR_: _dafny.Seq
            out42_: _dafny.Seq
            out43_: _dafny.Seq
            out42_, out43_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
            d_48_gR_ = out42_
            d_49_cR_ = out43_
            generated = d_48_gR_
            currentConstrainedOut = d_49_cR_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_50_gR2_: _dafny.Seq
                d_51_cR2_: _dafny.Seq
                out44_: _dafny.Seq
                out45_: _dafny.Seq
                out44_, out45_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_50_gR2_ = out44_
                d_51_cR2_ = out45_
                generated = d_50_gR2_
                currentConstrainedOut = d_51_cR2_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((len(currentConstrainedOut)) > (0)):
                d_52_gR3_: _dafny.Seq
                d_53_cR3_: _dafny.Seq
                out46_: _dafny.Seq
                out47_: _dafny.Seq
                out46_, out47_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_52_gR3_ = out46_
                d_53_cR3_ = out47_
                generated = d_52_gR3_
                currentConstrainedOut = d_53_cR3_
            if (not((parser).IsCompletePrefix(currentConstrainedOut))) and (((d_1_steps_) + (1)) < (maxSteps)):
                d_54_next_: _dafny.Seq
                out48_: _dafny.Seq
                out48_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                d_54_next_ = out48_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_54_next_) != (eosToken):
                    d_55_g2_: _dafny.Seq
                    d_56_i2_: bool
                    d_57_c2_: _dafny.Seq
                    out49_: _dafny.Seq
                    out50_: bool
                    out51_: _dafny.Seq
                    out49_, out50_, out51_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_54_next_)
                    d_55_g2_ = out49_
                    d_56_i2_ = out50_
                    d_57_c2_ = out51_
                    generated = d_55_g2_
                    insideConstrainedOut = d_56_i2_
                    currentConstrainedOut = d_57_c2_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_58_g2_: _dafny.Seq
                d_59_i2_: bool
                d_60_c2_: _dafny.Seq
                out52_: _dafny.Seq
                out53_: bool
                out54_: _dafny.Seq
                out52_, out53_, out54_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_58_g2_ = out52_
                d_59_i2_ = out53_
                d_60_c2_ = out54_
                generated = d_58_g2_
                insideConstrainedOut = d_59_i2_
                currentConstrainedOut = d_60_c2_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

