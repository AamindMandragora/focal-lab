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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap every arithmetic expression and the final numerical answer inside << >> delimiters, for example <<3 * 4 = 12>>. Keep each expression concise.")))
        d_1_mathDigitGroup_: _dafny.Seq
        d_1_mathDigitGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))])
        d_2_mathOpGroup_: _dafny.Seq
        d_2_mathOpGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))])
        d_3_mathNumGroup_: _dafny.Seq
        d_3_mathNumGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "10")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "100")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1000")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0.5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "12")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "60"))])
        d_4_mathGroups_: _dafny.Seq
        d_4_mathGroups_ = _dafny.SeqWithoutIsStrInference([d_1_mathDigitGroup_, d_2_mathOpGroup_, d_3_mathNumGroup_])
        d_5_steps_: int
        d_5_steps_ = 0
        d_6_spanSteps_: int
        d_6_spanSteps_ = 0
        d_7_maxSpanSteps_: int
        d_7_maxSpanSteps_ = 35
        with _dafny.label("0"):
            while (d_5_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_8_next_ = out0_
                        d_5_steps_ = (d_5_steps_) + (1)
                        if (d_8_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                            if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_6_spanSteps_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_9_g_: _dafny.Seq
                        d_10_i_: bool
                        d_11_c_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_9_g_ = out1_
                        d_10_i_ = out2_
                        d_11_c_ = out3_
                        generated = d_9_g_
                        insideConstrainedOut = d_10_i_
                        currentConstrainedOut = d_11_c_
                        d_5_steps_ = (d_5_steps_) + (1)
                    elif (d_6_spanSteps_) >= (d_7_maxSpanSteps_):
                        d_12_rolledG_: _dafny.Seq
                        d_13_rolledC_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: _dafny.Seq
                        out4_, out5_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_12_rolledG_ = out4_
                        d_13_rolledC_ = out5_
                        generated = d_12_rolledG_
                        currentConstrainedOut = d_13_rolledC_
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_14_g_: _dafny.Seq
                            d_15_i_: bool
                            d_16_c_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_14_g_ = out6_
                            d_15_i_ = out7_
                            d_16_c_ = out8_
                            generated = d_14_g_
                            insideConstrainedOut = d_15_i_
                            currentConstrainedOut = d_16_c_
                            d_5_steps_ = (d_5_steps_) + (1)
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_5_steps_ = (d_5_steps_) + (1)
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_penaltyTokens_: _dafny.Seq
                        d_18_penaltyTokens_ = currentConstrainedOut
                        d_19_next_: _dafny.Seq
                        out9_: _dafny.Seq
                        out9_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, d_4_mathGroups_, _dafny.BigRational('4e0'), d_18_penaltyTokens_, _dafny.BigRational('3e0'), 12, eosToken)
                        d_19_next_ = out9_
                        d_5_steps_ = (d_5_steps_) + (1)
                        d_6_spanSteps_ = (d_6_spanSteps_) + (1)
                        if (d_19_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_20_g_: _dafny.Seq
                            d_21_i_: bool
                            d_22_c_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_20_g_ = out10_
                            d_21_i_ = out11_
                            d_22_c_ = out12_
                            generated = d_20_g_
                            insideConstrainedOut = d_21_i_
                            currentConstrainedOut = d_22_c_
                    pass
            pass
        cost = d_5_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

