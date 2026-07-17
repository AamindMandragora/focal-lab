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
        d_2_reasoningBudget_: int
        d_2_reasoningBudget_ = 120
        d_3_spanMaxTokens_: int
        d_3_spanMaxTokens_ = 20
        d_4_spanTokensUsed_: int
        d_4_spanTokensUsed_ = 0
        d_5_forcedOpen_: bool
        d_5_forcedOpen_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_1_steps_) < (d_2_reasoningBudget_):
                            d_6_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_next_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_7_g2_: _dafny.Seq
                                    d_8_i2_: bool
                                    d_9_c2_: _dafny.Seq
                                    out1_: _dafny.Seq
                                    out2_: bool
                                    out3_: _dafny.Seq
                                    out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_7_g2_ = out1_
                                    d_8_i2_ = out2_
                                    d_9_c2_ = out3_
                                    generated = d_7_g2_
                                    insideConstrainedOut = d_8_i2_
                                    currentConstrainedOut = d_9_c2_
                                    d_4_spanTokensUsed_ = 0
                        elif (not(d_5_forcedOpen_)) and (((d_1_steps_) + (2)) <= (maxSteps)):
                            d_10_g2_: _dafny.Seq
                            d_11_i2_: bool
                            d_12_c2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_g2_ = out4_
                            d_11_i2_ = out5_
                            d_12_c2_ = out6_
                            generated = d_10_g2_
                            insideConstrainedOut = d_11_i2_
                            currentConstrainedOut = d_12_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_5_forcedOpen_ = True
                            d_4_spanTokensUsed_ = 0
                        elif True:
                            d_13_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_14_g2_: _dafny.Seq
                            d_15_i2_: bool
                            d_16_c2_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_14_g2_ = out8_
                            d_15_i2_ = out9_
                            d_16_c2_ = out10_
                            generated = d_14_g2_
                            insideConstrainedOut = d_15_i2_
                            currentConstrainedOut = d_16_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_4_spanTokensUsed_ = 0
                        elif (d_4_spanTokensUsed_) >= (d_3_spanMaxTokens_):
                            d_17_gR_: _dafny.Seq
                            d_18_cR_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_17_gR_ = out11_
                            d_18_cR_ = out12_
                            generated = d_17_gR_
                            currentConstrainedOut = d_18_cR_
                            d_4_spanTokensUsed_ = 0
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                if (d_1_steps_) < (maxSteps):
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
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_22_constrainedPrompt_: _dafny.Seq
                                d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_23_next_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_23_next_ = out16_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_23_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_24_g2_: _dafny.Seq
                                    d_25_i2_: bool
                                    d_26_c2_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                    d_24_g2_ = out17_
                                    d_25_i2_ = out18_
                                    d_26_c2_ = out19_
                                    generated = d_24_g2_
                                    insideConstrainedOut = d_25_i2_
                                    currentConstrainedOut = d_26_c2_
                                    d_4_spanTokensUsed_ = (d_4_spanTokensUsed_) + (1)
                        elif True:
                            d_27_constrainedPrompt_: _dafny.Seq
                            d_27_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_28_next_: _dafny.Seq
                            out20_: _dafny.Seq
                            out20_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_28_next_ = out20_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_28_next_) == (eosToken):
                                d_29_gR_: _dafny.Seq
                                d_30_cR_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: _dafny.Seq
                                out21_, out22_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_29_gR_ = out21_
                                d_30_cR_ = out22_
                                generated = d_29_gR_
                                currentConstrainedOut = d_30_cR_
                                d_4_spanTokensUsed_ = 0
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    if (d_1_steps_) < (maxSteps):
                                        d_31_g2_: _dafny.Seq
                                        d_32_i2_: bool
                                        d_33_c2_: _dafny.Seq
                                        out23_: _dafny.Seq
                                        out24_: bool
                                        out25_: _dafny.Seq
                                        out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_31_g2_ = out23_
                                        d_32_i2_ = out24_
                                        d_33_c2_ = out25_
                                        generated = d_31_g2_
                                        insideConstrainedOut = d_32_i2_
                                        currentConstrainedOut = d_33_c2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    d_34_gR2_: _dafny.Seq
                                    d_35_cR2_: _dafny.Seq
                                    out26_: _dafny.Seq
                                    out27_: _dafny.Seq
                                    out26_, out27_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_34_gR2_ = out26_
                                    d_35_cR2_ = out27_
                                    generated = d_34_gR2_
                                    currentConstrainedOut = d_35_cR2_
                                    if (parser).IsCompletePrefix(currentConstrainedOut):
                                        if (d_1_steps_) < (maxSteps):
                                            d_36_g3_: _dafny.Seq
                                            d_37_i3_: bool
                                            d_38_c3_: _dafny.Seq
                                            out28_: _dafny.Seq
                                            out29_: bool
                                            out30_: _dafny.Seq
                                            out28_, out29_, out30_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_36_g3_ = out28_
                                            d_37_i3_ = out29_
                                            d_38_c3_ = out30_
                                            generated = d_36_g3_
                                            insideConstrainedOut = d_37_i3_
                                            currentConstrainedOut = d_38_c3_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                        elif True:
                                            raise _dafny.Break("0")
                                    elif True:
                                        raise _dafny.Break("0")
                            elif True:
                                d_39_g2_: _dafny.Seq
                                d_40_i2_: bool
                                d_41_c2_: _dafny.Seq
                                out31_: _dafny.Seq
                                out32_: bool
                                out33_: _dafny.Seq
                                out31_, out32_, out33_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                                d_39_g2_ = out31_
                                d_40_i2_ = out32_
                                d_41_c2_ = out33_
                                generated = d_39_g2_
                                insideConstrainedOut = d_40_i2_
                                currentConstrainedOut = d_41_c2_
                                d_4_spanTokensUsed_ = (d_4_spanTokensUsed_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

