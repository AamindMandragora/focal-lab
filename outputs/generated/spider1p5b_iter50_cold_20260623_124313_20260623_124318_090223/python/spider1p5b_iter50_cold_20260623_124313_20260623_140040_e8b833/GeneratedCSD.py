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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a simple, correct SQL query. Rules: (1) Never use aliases - no AS keyword, no 't1.', 't2.' prefixes. (2) For counting: always use COUNT(*) not COUNT(DISTINCT col). (3) For 'both X and Y belong to category': use INTERSECT with two SELECT statements. (4) For grouping with count: SELECT col, COUNT(*) FROM table GROUP BY col. (5) Use simple JOINs with ON clause when joining tables. (6) Use exact column and table names from the schema. (7) Keep queries minimal - avoid unnecessary subqueries. (8) For finding common items: prefer INTERSECT over complex WHERE IN subqueries.")))
        if (insideConstrainedOut) and ((maxSteps) > (0)):
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
        elif (not(insideConstrainedOut)) and ((maxSteps) > (0)):
            d_5_penaltyTokens_: _dafny.Seq
            d_5_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " inner")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "inner")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " left")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "left")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " RIGHT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "RIGHT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " right")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "right")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " OUTER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OUTER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " outer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "outer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " CROSS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CROSS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " cross")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "cross")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " NATURAL")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NATURAL")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " natural")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "natural")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " distinct")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " alias")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "alias"))])
            d_6_steps_: int
            d_6_steps_ = 0
            d_7_accum_: _dafny.Seq
            d_7_accum_ = _dafny.SeqWithoutIsStrInference([])
            d_8_currentAcc_: _dafny.Seq
            d_8_currentAcc_ = _dafny.SeqWithoutIsStrInference([])
            with _dafny.label("1_0_0"):
                while (d_6_steps_) < (maxSteps):
                    with _dafny.c_label("1_0_0"):
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = prompt
                        d_10_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_9_constrainedPrompt_, d_8_currentAcc_, d_5_penaltyTokens_, _dafny.BigRational('4e0'), eosToken)
                        d_10_next_ = out3_
                        d_6_steps_ = (d_6_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            raise _dafny.Break("1_0_0")
                        elif True:
                            d_11_valid_: bool
                            out4_: bool
                            out4_ = (d_0_helpers_).IsTokenValidNext(parser, d_8_currentAcc_, d_10_next_)
                            d_11_valid_ = out4_
                            if d_11_valid_:
                                d_8_currentAcc_ = (d_8_currentAcc_) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                d_7_accum_ = (d_7_accum_) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                        pass
                pass
            generated = (generatedPrefix) + (d_7_accum_)
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            cost = d_6_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

