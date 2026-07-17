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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, showing each calculation. Wrap every arithmetic expression and the final numerical answer in << >> delimiters.")))
        d_1_mathDigits_: _dafny.Seq
        d_1_mathDigits_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))])
        d_2_mathOps_: _dafny.Seq
        d_2_mathOps_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "."))])
        d_3_mathMisc_: _dafny.Seq
        d_3_mathMisc_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "100")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "10")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1000")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0.5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0.25"))])
        d_4_mathGroups_: _dafny.Seq
        d_4_mathGroups_ = _dafny.SeqWithoutIsStrInference([d_1_mathDigits_, d_2_mathOps_, d_3_mathMisc_])
        d_5_steps_: int
        d_5_steps_ = 0
        d_6_chunkSize_: int
        d_6_chunkSize_ = 60
        with _dafny.label("0"):
            while (d_5_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_remaining_: int
                        d_7_remaining_ = (maxSteps) - (d_5_steps_)
                        d_8_budget_: int
                        if (d_7_remaining_) > (20):
                            if ((d_7_remaining_) - (20)) < (d_6_chunkSize_):
                                d_8_budget_ = (d_7_remaining_) - (20)
                            elif True:
                                d_8_budget_ = d_6_chunkSize_
                        elif True:
                            d_8_budget_ = 1
                        if (d_8_budget_) == (0):
                            raise _dafny.Break("0")
                        d_9_generatedOut_: _dafny.Seq
                        d_10_stoppedOnOpenSpan_: bool
                        d_11_stoppedOnEos_: bool
                        d_12_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_budget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_9_generatedOut_ = out0_
                        d_10_stoppedOnOpenSpan_ = out1_
                        d_11_stoppedOnEos_ = out2_
                        d_12_stepsUsed_ = out3_
                        generated = d_9_generatedOut_
                        d_5_steps_ = (d_5_steps_) + (d_12_stepsUsed_)
                        if d_10_stoppedOnOpenSpan_:
                            d_13_g_: _dafny.Seq
                            d_14_i_: bool
                            d_15_c_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_13_g_ = out4_
                            d_14_i_ = out5_
                            d_15_c_ = out6_
                            generated = d_13_g_
                            insideConstrainedOut = d_14_i_
                            currentConstrainedOut = d_15_c_
                        elif d_11_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if (d_5_steps_) < (maxSteps):
                                d_16_g_: _dafny.Seq
                                d_17_i_: bool
                                d_18_c_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_16_g_ = out7_
                                d_17_i_ = out8_
                                d_18_c_ = out9_
                                generated = d_16_g_
                                insideConstrainedOut = d_17_i_
                                currentConstrainedOut = d_18_c_
                                d_5_steps_ = (d_5_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_19_g_: _dafny.Seq
                        d_20_i_: bool
                        d_21_c_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_19_g_ = out10_
                        d_20_i_ = out11_
                        d_21_c_ = out12_
                        generated = d_19_g_
                        insideConstrainedOut = d_20_i_
                        currentConstrainedOut = d_21_c_
                        d_5_steps_ = (d_5_steps_) + (1)
                    elif True:
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_23_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, d_4_mathGroups_, _dafny.BigRational('4e0'), 12, eosToken)
                        d_23_next_ = out13_
                        d_5_steps_ = (d_5_steps_) + (1)
                        if (d_23_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_24_g_: _dafny.Seq
                            d_25_i_: bool
                            d_26_c_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                            d_24_g_ = out14_
                            d_25_i_ = out15_
                            d_26_c_ = out16_
                            generated = d_24_g_
                            insideConstrainedOut = d_25_i_
                            currentConstrainedOut = d_26_c_
                    pass
            pass
        cost = d_5_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

