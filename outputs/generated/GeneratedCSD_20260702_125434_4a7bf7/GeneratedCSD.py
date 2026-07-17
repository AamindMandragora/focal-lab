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
        with _dafny.label("0"):
            while (d_0_helpers_.cost) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(d_2_inside_):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, d_1_g_)
                        d_4_next_ = out0_
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        d_1_g_ = (d_1_g_) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        if VerifiedDecoderAgent.default__.Contains(d_4_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_2_inside_ = True
                            d_3_cur_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_5_constrainedPrompt_: _dafny.Seq
                        d_5_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((d_1_g_)[:(len(d_1_g_)) - (len(d_3_cur_)):]))
                        d_6_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_5_constrainedPrompt_, d_3_cur_, eosToken)
                        d_6_next_ = out1_
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        out2_: _dafny.Seq
                        out3_: bool
                        out4_: _dafny.Seq
                        out2_, out3_, out4_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, d_1_g_, d_3_cur_, d_6_next_)
                        d_1_g_ = out2_
                        d_2_inside_ = out3_
                        d_3_cur_ = out4_
                        if VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                            d_2_inside_ = False
                            d_3_cur_ = _dafny.SeqWithoutIsStrInference([])
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

