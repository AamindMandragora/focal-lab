import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import module_ as module_
import _dafny as _dafny
import System_ as System_
import VerifiedDecoderAgent as VerifiedDecoderAgent

# Module: ReferenceCarsCSD

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
        d_1_resScore_: _dafny.BigRational
        out0_: _dafny.BigRational
        out0_ = (d_0_helpers_).PrefixResemblesPromptExamples(lm, generatedPrefix)
        d_1_resScore_ = out0_
        d_2_g_: _dafny.Seq
        d_2_g_ = generatedPrefix
        d_3_inside_: bool
        d_3_inside_ = insideConstrained
        d_4_cur_: _dafny.Seq
        d_4_cur_ = currentConstrained
        d_5_rejectedTokens_: _dafny.Seq
        d_5_rejectedTokens_ = _dafny.SeqWithoutIsStrInference([])
        d_6_spanEntryLen_: int
        if d_3_inside_:
            d_6_spanEntryLen_ = (len(d_2_g_)) - (len(d_4_cur_))
        elif True:
            d_6_spanEntryLen_ = 0
        if (maxSteps) == (0):
            generated = d_2_g_
            insideConstrainedOut = d_3_inside_
            if d_3_inside_:
                currentConstrainedOut = d_4_cur_
            elif True:
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            cost = d_0_helpers_.cost
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        with _dafny.label("0"):
            while (d_0_helpers_.cost) < (maxSteps):
                with _dafny.c_label("0"):
                    if (d_3_inside_) and ((parser).IsCompletePrefix(d_4_cur_)):
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, d_2_g_, d_4_cur_)
                        d_2_g_ = out1_
                        d_3_inside_ = out2_
                        d_4_cur_ = out3_
                    elif not(d_3_inside_):
                        d_7_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, d_2_g_)
                        d_7_next_ = out4_
                        d_2_g_ = (d_2_g_) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                        if (d_7_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_3_inside_ = True
                            d_4_cur_ = _dafny.SeqWithoutIsStrInference([])
                            d_6_spanEntryLen_ = len(d_2_g_)
                            d_5_rejectedTokens_ = _dafny.SeqWithoutIsStrInference([])
                    elif (len(d_4_cur_)) == (0):
                        d_8_next_: _dafny.Seq
                        out5_: _dafny.Seq
                        out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, d_4_cur_, eosToken)
                        d_8_next_ = out5_
                        if (d_8_next_) == (eosToken):
                            raise _dafny.Break("0")
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, d_2_g_, d_4_cur_, d_8_next_)
                        d_2_g_ = out6_
                        d_3_inside_ = out7_
                        d_4_cur_ = out8_
                    elif (len(d_5_rejectedTokens_)) == (0):
                        d_9_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_10_isValid_: bool = False
                        out9_: _dafny.Seq
                        out10_: bool
                        out9_, out10_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, prompt, d_4_cur_, _dafny.BigRational('0e0'), eosToken)
                        d_9_next_ = out9_
                        d_10_isValid_ = out10_
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("0")
                        if d_10_isValid_:
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, d_2_g_, d_4_cur_, d_9_next_)
                            d_2_g_ = out11_
                            d_3_inside_ = out12_
                            d_4_cur_ = out13_
                        elif True:
                            d_5_rejectedTokens_ = (d_5_rejectedTokens_) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                            d_2_g_ = _dafny.SeqWithoutIsStrInference((d_2_g_)[:d_6_spanEntryLen_:])
                            d_4_cur_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_11_next_: _dafny.Seq
                        out14_: _dafny.Seq
                        out14_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, prompt, d_4_cur_, d_5_rejectedTokens_, _dafny.BigRational('1e8'), eosToken)
                        d_11_next_ = out14_
                        if (d_11_next_) == (eosToken):
                            raise _dafny.Break("0")
                        out15_: _dafny.Seq
                        out16_: bool
                        out17_: _dafny.Seq
                        out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, d_2_g_, d_4_cur_, d_11_next_)
                        d_2_g_ = out15_
                        d_3_inside_ = out16_
                        d_4_cur_ = out17_
                    pass
            pass
        generated = d_2_g_
        insideConstrainedOut = d_3_inside_
        if d_3_inside_:
            currentConstrainedOut = d_4_cur_
        elif True:
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_0_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost

