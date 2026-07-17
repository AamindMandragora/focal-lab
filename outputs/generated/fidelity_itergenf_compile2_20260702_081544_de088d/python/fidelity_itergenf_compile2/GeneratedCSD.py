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
        d_4_spanEntryLen_: int
        if d_2_inside_:
            d_4_spanEntryLen_ = (len(d_1_g_)) - (len(d_3_cur_))
        elif True:
            d_4_spanEntryLen_ = 0
        d_5_rejectedTokens_: _dafny.Seq
        d_5_rejectedTokens_ = _dafny.SeqWithoutIsStrInference([])
        d_6_rollbackCount_: int
        d_6_rollbackCount_ = 0
        d_7_BACKWARDS__LIMIT_: int
        d_7_BACKWARDS__LIMIT_ = 10
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
                        d_4_spanEntryLen_ = 0
                        d_5_rejectedTokens_ = _dafny.SeqWithoutIsStrInference([])
                        d_6_rollbackCount_ = 0
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
                            d_4_spanEntryLen_ = len(d_1_g_)
                            d_5_rejectedTokens_ = _dafny.SeqWithoutIsStrInference([])
                            d_6_rollbackCount_ = 0
                    elif (parser).IsDeadPrefix(d_3_cur_):
                        if (d_6_rollbackCount_) >= (d_7_BACKWARDS__LIMIT_):
                            d_1_g_ = _dafny.SeqWithoutIsStrInference((d_1_g_)[:d_4_spanEntryLen_:])
                            d_2_inside_ = False
                            d_3_cur_ = _dafny.SeqWithoutIsStrInference([])
                            d_4_spanEntryLen_ = 0
                            d_5_rejectedTokens_ = _dafny.SeqWithoutIsStrInference([])
                            d_6_rollbackCount_ = 0
                        elif True:
                            if (len(d_3_cur_)) > (0):
                                d_5_rejectedTokens_ = (d_5_rejectedTokens_) + (_dafny.SeqWithoutIsStrInference([(d_3_cur_)[0]]))
                            d_1_g_ = _dafny.SeqWithoutIsStrInference((d_1_g_)[:d_4_spanEntryLen_:])
                            d_3_cur_ = _dafny.SeqWithoutIsStrInference([])
                            d_6_rollbackCount_ = (d_6_rollbackCount_) + (1)
                    elif True:
                        d_9_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (len(d_5_rejectedTokens_)) == (0):
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, d_3_cur_, eosToken)
                            d_9_next_ = out4_
                        elif True:
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, prompt, d_3_cur_, d_5_rejectedTokens_, _dafny.BigRational('1e8'), eosToken)
                            d_9_next_ = out5_
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("0")
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, d_1_g_, d_3_cur_, d_9_next_)
                        d_1_g_ = out6_
                        d_2_inside_ = out7_
                        d_3_cur_ = out8_
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

