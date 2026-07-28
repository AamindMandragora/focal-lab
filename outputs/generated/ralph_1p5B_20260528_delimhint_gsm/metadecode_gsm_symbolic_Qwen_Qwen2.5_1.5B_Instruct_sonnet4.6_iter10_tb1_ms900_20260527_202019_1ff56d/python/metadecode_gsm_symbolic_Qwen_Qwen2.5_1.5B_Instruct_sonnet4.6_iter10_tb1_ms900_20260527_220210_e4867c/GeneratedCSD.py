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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap every arithmetic expression and the final numerical answer inside << >> delimiters. Example: <<3 * 4 = 12>>. Keep each expression concise: use only numbers and operators +, -, *, /, =, (, ).")))
        d_1_mathDigitGroup_: _dafny.Seq
        d_1_mathDigitGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))])
        d_2_mathOpGroup_: _dafny.Seq
        d_2_mathOpGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))])
        d_3_mathGroups_: _dafny.Seq
        d_3_mathGroups_ = _dafny.SeqWithoutIsStrInference([d_1_mathDigitGroup_, d_2_mathOpGroup_])
        d_4_steps_: int
        d_4_steps_ = 0
        d_5_spanSteps_: int
        d_5_spanSteps_ = 0
        d_6_maxSpanSteps_: int
        d_6_maxSpanSteps_ = 40
        with _dafny.label("0"):
            while (d_4_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_chunkBudget_: int
                        if ((maxSteps) - (d_4_steps_)) < (80):
                            d_7_chunkBudget_ = (maxSteps) - (d_4_steps_)
                        elif True:
                            d_7_chunkBudget_ = 80
                        d_8_g_: _dafny.Seq
                        d_9_stoppedOnOpen_: bool
                        d_10_stoppedOnEos_: bool
                        d_11_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_8_g_ = out0_
                        d_9_stoppedOnOpen_ = out1_
                        d_10_stoppedOnEos_ = out2_
                        d_11_stepsUsed_ = out3_
                        generated = d_8_g_
                        d_4_steps_ = (d_4_steps_) + (d_11_stepsUsed_)
                        if d_10_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_9_stoppedOnOpen_:
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
                            d_5_spanSteps_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_g_: _dafny.Seq
                        d_16_i_: bool
                        d_17_c_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_g_ = out7_
                        d_16_i_ = out8_
                        d_17_c_ = out9_
                        generated = d_15_g_
                        insideConstrainedOut = d_16_i_
                        currentConstrainedOut = d_17_c_
                        d_4_steps_ = (d_4_steps_) + (1)
                        d_5_spanSteps_ = 0
                    elif (d_5_spanSteps_) >= (d_6_maxSpanSteps_):
                        d_18_rolledG_: _dafny.Seq
                        d_19_rolledC_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_18_rolledG_ = out10_
                        d_19_rolledC_ = out11_
                        generated = d_18_rolledG_
                        currentConstrainedOut = d_19_rolledC_
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_20_g_: _dafny.Seq
                            d_21_i_: bool
                            d_22_c_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_20_g_ = out12_
                            d_21_i_ = out13_
                            d_22_c_ = out14_
                            generated = d_20_g_
                            insideConstrainedOut = d_21_i_
                            currentConstrainedOut = d_22_c_
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_4_steps_ = (d_4_steps_) + (1)
                        d_5_spanSteps_ = 0
                    elif True:
                        d_23_constrainedPrompt_: _dafny.Seq
                        d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_24_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, d_3_mathGroups_, _dafny.BigRational('4e0'), 12, eosToken)
                        d_24_next_ = out15_
                        d_4_steps_ = (d_4_steps_) + (1)
                        d_5_spanSteps_ = (d_5_spanSteps_) + (1)
                        if (d_24_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_25_g_: _dafny.Seq
                            d_26_i_: bool
                            d_27_c_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                            d_25_g_ = out16_
                            d_26_i_ = out17_
                            d_27_c_ = out18_
                            generated = d_25_g_
                            insideConstrainedOut = d_26_i_
                            currentConstrainedOut = d_27_c_
                    pass
            pass
        cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

