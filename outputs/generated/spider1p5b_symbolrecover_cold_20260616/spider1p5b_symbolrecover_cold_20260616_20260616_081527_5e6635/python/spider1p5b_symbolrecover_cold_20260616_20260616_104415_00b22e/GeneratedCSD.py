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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer the question using ONLY the tables and columns in the provided database schema. Output exactly: SQL: <<your SQL query here>> with no other text. Use only single-quoted strings, no semicolons, no backticks.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_prefixBudget_: int = int(0)
        if (8) <= (maxSteps):
            d_2_prefixBudget_ = 8
        elif True:
            d_2_prefixBudget_ = maxSteps
        while ((d_1_steps_) < (d_2_prefixBudget_)) and (not(insideConstrainedOut)):
            d_3_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_3_next_ = out0_
            d_1_steps_ = (d_1_steps_) + (1)
            if (d_3_next_) == (eosToken):
                d_1_steps_ = d_2_prefixBudget_
            elif (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                d_4_g_: _dafny.Seq
                d_5_i_: bool
                d_6_c_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_4_g_ = out1_
                d_5_i_ = out2_
                d_6_c_ = out3_
                generated = d_4_g_
                insideConstrainedOut = d_5_i_
                currentConstrainedOut = d_6_c_
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_7_g_: _dafny.Seq
            d_8_i_: bool
            d_9_c_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_7_g_ = out4_
            d_8_i_ = out5_
            d_9_c_ = out6_
            generated = d_7_g_
            insideConstrainedOut = d_8_i_
            currentConstrainedOut = d_9_c_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_10_g_: _dafny.Seq
                    d_11_i_: bool
                    d_12_c_: _dafny.Seq
                    d_13_closed_: bool
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out10_: bool
                    out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_10_g_ = out7_
                    d_11_i_ = out8_
                    d_12_c_ = out9_
                    d_13_closed_ = out10_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_13_closed_:
                        generated = d_10_g_
                        insideConstrainedOut = d_11_i_
                        currentConstrainedOut = d_12_c_
                    elif (d_1_steps_) < (maxSteps):
                        d_14_next_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, prompt, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_14_next_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_15_g2_: _dafny.Seq
                            d_16_i2_: bool
                            d_17_c2_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            d_15_g2_ = out12_
                            d_16_i2_ = out13_
                            d_17_c2_ = out14_
                            generated = d_15_g2_
                            insideConstrainedOut = d_16_i2_
                            currentConstrainedOut = d_17_c2_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_18_remainingBudget_: int
            d_18_remainingBudget_ = (maxSteps) - (d_1_steps_)
            d_19_g_: _dafny.Seq
            d_20_i_: bool
            d_21_c_: _dafny.Seq
            out15_: _dafny.Seq
            out16_: bool
            out17_: _dafny.Seq
            out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_remainingBudget_)
            d_19_g_ = out15_
            d_20_i_ = out16_
            d_21_c_ = out17_
            generated = d_19_g_
            insideConstrainedOut = d_20_i_
            currentConstrainedOut = d_21_c_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        if ((cost) == (0)) and ((maxSteps) > (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

