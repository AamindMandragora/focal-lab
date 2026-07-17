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
        d_1_g_: _dafny.Seq
        d_1_g_ = generatedPrefix
        d_2_inside_: bool
        d_2_inside_ = insideConstrained
        d_3_cur_: _dafny.Seq
        d_3_cur_ = currentConstrained
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
                    if (d_2_inside_) and ((parser).IsCompletePrefix(d_3_cur_)):
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, d_1_g_, d_3_cur_)
                        d_1_g_ = out0_
                        d_2_inside_ = out1_
                        d_3_cur_ = out2_
                    elif not(d_2_inside_):
                        d_4_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, d_1_g_)
                        d_4_next_ = out3_
                        d_1_g_ = (d_1_g_) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_2_inside_ = True
                            d_3_cur_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_5_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, prompt, d_3_cur_, _dafny.SeqWithoutIsStrInference([]), _dafny.BigRational('0e0'), eosToken)
                        d_5_next_ = out4_
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, d_1_g_, d_3_cur_, d_5_next_)
                        d_1_g_ = out5_
                        d_2_inside_ = out6_
                        d_3_cur_ = out7_
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

