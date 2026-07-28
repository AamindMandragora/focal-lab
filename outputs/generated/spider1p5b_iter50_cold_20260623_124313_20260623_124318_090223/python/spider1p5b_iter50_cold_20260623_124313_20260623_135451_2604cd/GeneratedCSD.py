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
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            if (maxSteps) == (0):
                pass
            elif insideConstrained:
                d_1_closeBudget_: int
                d_1_closeBudget_ = maxSteps
                d_2_cg_: _dafny.Seq
                d_3_ci_: bool
                d_4_cc_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_1_closeBudget_)
                d_2_cg_ = out0_
                d_3_ci_ = out1_
                d_4_cc_ = out2_
                generated = d_2_cg_
                insideConstrainedOut = d_3_ci_
                currentConstrainedOut = d_4_cc_
                cost = maxSteps
            elif True:
                d_5_penaltyTokens_: _dafny.Seq
                d_5_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " RIGHT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "RIGHT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " OUTER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OUTER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " CROSS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CROSS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " NATURAL")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NATURAL")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " T3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " d.")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "d.")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " t.")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t.")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " s.")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "s.")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " e.")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "e.")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " alias")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "alias"))])
                d_6_steps_: int
                d_6_steps_ = 0
                d_7_accum_: _dafny.Seq
                d_7_accum_ = _dafny.SeqWithoutIsStrInference([])
                d_8_currentAcc_: _dafny.Seq
                d_8_currentAcc_ = _dafny.SeqWithoutIsStrInference([])
                with _dafny.label("1_1_0"):
                    while (d_6_steps_) < (maxSteps):
                        with _dafny.c_label("1_1_0"):
                            (d_0_helpers_).SafePenalizeTokenLogits(lm, d_5_penaltyTokens_, _dafny.BigRational('5e0'))
                            (d_0_helpers_).BoostValidGroups(lm, parser, d_8_currentAcc_, validTokenGroups, _dafny.BigRational('2e0'))
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, prompt, d_8_currentAcc_, d_5_penaltyTokens_, _dafny.BigRational('5e0'), eosToken)
                            d_9_next_ = out3_
                            d_6_steps_ = (d_6_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("1_1_0")
                            elif True:
                                d_10_valid_: bool
                                out4_: bool
                                out4_ = (d_0_helpers_).IsTokenValidNext(parser, d_8_currentAcc_, d_9_next_)
                                d_10_valid_ = out4_
                                if d_10_valid_:
                                    d_8_currentAcc_ = (d_8_currentAcc_) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                    d_7_accum_ = (d_7_accum_) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                            pass
                    pass
                generated = (generatedPrefix) + (d_7_accum_)
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                cost = d_6_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

