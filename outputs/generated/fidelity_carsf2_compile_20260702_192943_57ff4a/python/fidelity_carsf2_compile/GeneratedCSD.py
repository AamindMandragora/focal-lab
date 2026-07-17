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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        d_1_g_: _dafny.Seq
        d_1_g_ = generatedPrefix
        d_2_inside_: bool
        d_2_inside_ = insideConstrained
        d_3_cur_: _dafny.Seq
        d_3_cur_ = currentConstrained
        d_4_rejectedTokens_: _dafny.Seq
        d_4_rejectedTokens_ = _dafny.SeqWithoutIsStrInference([])
        d_5_spanEntryLen_: int
        if d_2_inside_:
            d_5_spanEntryLen_ = (len(d_1_g_)) - (len(d_3_cur_))
        elif True:
            d_5_spanEntryLen_ = 0
        d_6_closeRequested_: bool
        d_6_closeRequested_ = False
        d_7_stopAfterClose_: bool
        d_7_stopAfterClose_ = False
        if (maxSteps) == (0):
            generated = d_1_g_
            insideConstrainedOut = d_2_inside_
            if d_2_inside_:
                currentConstrainedOut = d_3_cur_
            elif True:
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            cost = d_0_helpers_.cost
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        with _dafny.label("0"):
            while (d_0_helpers_.cost) < (maxSteps):
                with _dafny.c_label("0"):
                    if (d_2_inside_) and (d_6_closeRequested_):
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, d_1_g_, d_3_cur_)
                        d_1_g_ = out0_
                        d_2_inside_ = out1_
                        d_3_cur_ = out2_
                        d_6_closeRequested_ = False
                        if d_7_stopAfterClose_:
                            raise _dafny.Break("0")
                    elif not(d_2_inside_):
                        d_8_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, d_1_g_)
                        d_8_next_ = out3_
                        d_1_g_ = (d_1_g_) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                        if (d_8_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_2_inside_ = True
                            d_3_cur_ = _dafny.SeqWithoutIsStrInference([])
                            d_5_spanEntryLen_ = len(d_1_g_)
                            d_4_rejectedTokens_ = _dafny.SeqWithoutIsStrInference([])
                    elif (len(d_3_cur_)) == (0):
                        d_9_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, d_3_cur_, eosToken)
                        d_9_next_ = out4_
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("0")
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, d_1_g_, d_3_cur_, d_9_next_)
                        d_1_g_ = out5_
                        d_2_inside_ = out6_
                        d_3_cur_ = out7_
                    elif (len(d_4_rejectedTokens_)) == (0):
                        d_10_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_11_isValid_: bool = False
                        out8_: _dafny.Seq
                        out9_: bool
                        out8_, out9_ = (d_0_helpers_).SoftConstrainedStep(lm, parser, prompt, d_3_cur_, _dafny.BigRational('0e0'), eosToken)
                        d_10_next_ = out8_
                        d_11_isValid_ = out9_
                        if (d_10_next_) == (eosToken):
                            if (parser).IsCompletePrefix(d_3_cur_):
                                d_6_closeRequested_ = True
                                d_7_stopAfterClose_ = True
                            elif True:
                                d_4_rejectedTokens_ = (d_4_rejectedTokens_) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                d_1_g_ = _dafny.SeqWithoutIsStrInference((d_1_g_)[:d_5_spanEntryLen_:])
                                d_3_cur_ = _dafny.SeqWithoutIsStrInference([])
                        elif d_11_isValid_:
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, d_1_g_, d_3_cur_, d_10_next_)
                            d_1_g_ = out10_
                            d_2_inside_ = out11_
                            d_3_cur_ = out12_
                        elif ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) and ((parser).IsCompletePrefix(d_3_cur_)):
                            d_6_closeRequested_ = True
                        elif True:
                            d_4_rejectedTokens_ = (d_4_rejectedTokens_) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                            d_1_g_ = _dafny.SeqWithoutIsStrInference((d_1_g_)[:d_5_spanEntryLen_:])
                            d_3_cur_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_12_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, prompt, d_3_cur_, d_4_rejectedTokens_, _dafny.BigRational('1e8'), eosToken)
                        d_12_next_ = out13_
                        if (d_12_next_) == (eosToken):
                            if (parser).IsCompletePrefix(d_3_cur_):
                                d_6_closeRequested_ = True
                                d_7_stopAfterClose_ = True
                            elif True:
                                d_4_rejectedTokens_ = (d_4_rejectedTokens_) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                d_1_g_ = _dafny.SeqWithoutIsStrInference((d_1_g_)[:d_5_spanEntryLen_:])
                                d_3_cur_ = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, d_1_g_, d_3_cur_, d_12_next_)
                            d_1_g_ = out14_
                            d_2_inside_ = out15_
                            d_3_cur_ = out16_
                    pass
            pass
        generated = d_1_g_
        insideConstrainedOut = d_2_inside_
        if d_2_inside_:
            currentConstrainedOut = d_3_cur_
        elif True:
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_0_helpers_.cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost

