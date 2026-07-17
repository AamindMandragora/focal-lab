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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write the simplest correct SQL query. Do not use table aliases. Do not use AS keyword. Do not use INNER JOIN, use just JOIN. Use exact column names from schema. Answer directly without extra complexity."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if insideConstrainedOut:
            if (maxSteps) > (0):
                d_2_closeBudget_: int
                d_2_closeBudget_ = maxSteps
                d_3_cg_: _dafny.Seq
                d_4_ci_: bool
                d_5_cc_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_2_closeBudget_)
                d_3_cg_ = out0_
                d_4_ci_ = out1_
                d_5_cc_ = out2_
                generated = d_3_cg_
                insideConstrainedOut = d_4_ci_
                currentConstrainedOut = d_5_cc_
                cost = maxSteps
        elif True:
            if (maxSteps) > (0):
                d_6_steps_: int
                d_6_steps_ = 0
                d_7_currentC_: _dafny.Seq
                d_7_currentC_ = _dafny.SeqWithoutIsStrInference([])
                d_8_penaltyTokens_: _dafny.Seq
                d_8_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OUTER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "RIGHT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "alias"))])
                with _dafny.label("1_0_0"):
                    while (d_6_steps_) < (maxSteps):
                        with _dafny.c_label("1_0_0"):
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, prompt, d_7_currentC_, validTokenGroups, _dafny.BigRational('2e0'), d_8_penaltyTokens_, _dafny.BigRational('3e0'), 12, eosToken)
                            d_9_next_ = out3_
                            d_6_steps_ = (d_6_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("1_0_0")
                            elif True:
                                d_10_valid_: bool
                                out4_: bool
                                out4_ = (d_0_helpers_).IsTokenValidNext(parser, d_7_currentC_, d_9_next_)
                                d_10_valid_ = out4_
                                if d_10_valid_:
                                    d_7_currentC_ = (d_7_currentC_) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                            pass
                    pass
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                cost = d_6_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

