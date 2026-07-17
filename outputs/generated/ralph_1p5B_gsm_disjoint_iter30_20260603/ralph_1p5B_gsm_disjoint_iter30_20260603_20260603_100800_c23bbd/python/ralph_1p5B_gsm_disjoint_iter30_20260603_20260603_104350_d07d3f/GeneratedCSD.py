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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write each arithmetic expression inside << >> delimiters. End your answer with the final numeric expression inside << >> after ####.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 25
        d_3_spanCount_: int
        d_3_spanCount_ = 0
        d_4_maxSpans_: int
        d_4_maxSpans_ = 15
        d_5_spanTokensUsed_: int
        d_5_spanTokensUsed_ = 0
        d_6_spanMaxTokens_: int
        d_6_spanMaxTokens_ = 30
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_chunkBudget_: int
                        d_7_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_7_chunkBudget_) > (d_2_freeChunkSize_):
                            d_7_chunkBudget_ = d_2_freeChunkSize_
                        if (d_7_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_8_chunkGenerated_: _dafny.Seq
                        d_9_stoppedOnOpenSpan_: bool
                        d_10_stoppedOnEos_: bool
                        d_11_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_8_chunkGenerated_ = out0_
                        d_9_stoppedOnOpenSpan_ = out1_
                        d_10_stoppedOnEos_ = out2_
                        d_11_stepsUsed_ = out3_
                        generated = d_8_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                        if d_10_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_9_stoppedOnOpenSpan_:
                            d_12_g2_: _dafny.Seq
                            d_13_i2_: bool
                            d_14_c2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_12_g2_ = out4_
                            d_13_i2_ = out5_
                            d_14_c2_ = out6_
                            generated = d_12_g2_
                            insideConstrainedOut = d_13_i2_
                            currentConstrainedOut = d_14_c2_
                            d_3_spanCount_ = (d_3_spanCount_) + (1)
                            d_5_spanTokensUsed_ = 0
                        elif True:
                            if ((d_3_spanCount_) < (d_4_maxSpans_)) and ((d_1_steps_) < (maxSteps)):
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
                                d_3_spanCount_ = (d_3_spanCount_) + (1)
                                d_5_spanTokensUsed_ = 0
                            elif True:
                                if (d_1_steps_) < (maxSteps):
                                    d_18_next_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                    d_18_next_ = out10_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_18_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_18_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_19_g2_: _dafny.Seq
                        d_20_i2_: bool
                        d_21_c2_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_19_g2_ = out11_
                        d_20_i2_ = out12_
                        d_21_c2_ = out13_
                        generated = d_19_g2_
                        insideConstrainedOut = d_20_i2_
                        currentConstrainedOut = d_21_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_5_spanTokensUsed_ = 0
                    elif (d_5_spanTokensUsed_) >= (d_6_spanMaxTokens_):
                        d_22_gRolled_: _dafny.Seq
                        d_23_cRolled_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: _dafny.Seq
                        out14_, out15_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_22_gRolled_ = out14_
                        d_23_cRolled_ = out15_
                        generated = d_22_gRolled_
                        currentConstrainedOut = d_23_cRolled_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_24_g2_: _dafny.Seq
                            d_25_i2_: bool
                            d_26_c2_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_24_g2_ = out16_
                            d_25_i2_ = out17_
                            d_26_c2_ = out18_
                            generated = d_24_g2_
                            insideConstrainedOut = d_25_i2_
                            currentConstrainedOut = d_26_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_5_spanTokensUsed_ = 0
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_27_constrainedPrompt_: _dafny.Seq
                                d_27_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_28_next_: _dafny.Seq
                                d_29_wasConstrained_: bool
                                out19_: _dafny.Seq
                                out20_: bool
                                out19_, out20_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_28_next_ = out19_
                                d_29_wasConstrained_ = out20_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_5_spanTokensUsed_ = 0
                                if (d_28_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_30_g2_: _dafny.Seq
                                    d_31_i2_: bool
                                    d_32_c2_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                                    d_30_g2_ = out21_
                                    d_31_i2_ = out22_
                                    d_32_c2_ = out23_
                                    generated = d_30_g2_
                                    insideConstrainedOut = d_31_i2_
                                    currentConstrainedOut = d_32_c2_
                    elif True:
                        d_33_constrainedPrompt_: _dafny.Seq
                        d_33_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_34_next_: _dafny.Seq
                        d_35_wasConstrained_: bool
                        out24_: _dafny.Seq
                        out25_: bool
                        out24_, out25_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_33_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_34_next_ = out24_
                        d_35_wasConstrained_ = out25_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_34_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_36_g2_: _dafny.Seq
                            d_37_i2_: bool
                            d_38_c2_: _dafny.Seq
                            out26_: _dafny.Seq
                            out27_: bool
                            out28_: _dafny.Seq
                            out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_34_next_)
                            d_36_g2_ = out26_
                            d_37_i2_ = out27_
                            d_38_c2_ = out28_
                            generated = d_36_g2_
                            insideConstrainedOut = d_37_i2_
                            currentConstrainedOut = d_38_c2_
                            d_5_spanTokensUsed_ = (d_5_spanTokensUsed_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

