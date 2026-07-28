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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write a correct SQL query. Key rules: (1) For 'both X and Y' conditions use INTERSECT: SELECT col FROM t WHERE cond1 INTERSECT SELECT col FROM t WHERE cond2. (2) For 'either X or Y' use UNION: SELECT col FROM t WHERE cond1 UNION SELECT col FROM t WHERE cond2. (3) Never use AS aliases - use full table.column notation. (4) For counting by group: SELECT col, COUNT(*) FROM table GROUP BY col. (5) Use exact column and table names from the schema. (6) Keep queries simple and direct. Do not use subqueries when a JOIN works.")))
        if (insideConstrainedOut) and ((maxSteps) > (0)):
            d_1_cg_: _dafny.Seq
            d_2_ci_: bool
            d_3_cc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, maxSteps)
            d_1_cg_ = out0_
            d_2_ci_ = out1_
            d_3_cc_ = out2_
            generated = d_1_cg_
            insideConstrainedOut = d_2_ci_
            currentConstrainedOut = d_3_cc_
            cost = maxSteps
        elif (not(insideConstrainedOut)) and ((maxSteps) > (0)):
            d_4_penaltyTokens_: _dafny.Seq
            d_4_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " RIGHT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "RIGHT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " OUTER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OUTER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " alias")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "alias")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " CROSS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CROSS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " NATURAL")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NATURAL")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " WITH")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WITH"))])
            d_5_steps_: int
            d_5_steps_ = 0
            d_6_accum_: _dafny.Seq
            d_6_accum_ = _dafny.SeqWithoutIsStrInference([])
            d_7_currentAcc_: _dafny.Seq
            d_7_currentAcc_ = _dafny.SeqWithoutIsStrInference([])
            with _dafny.label("1_0_0"):
                while (d_5_steps_) < (maxSteps):
                    with _dafny.c_label("1_0_0"):
                        d_8_constrainedPrompt_: _dafny.Seq
                        d_8_constrainedPrompt_ = prompt
                        d_9_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_8_constrainedPrompt_, d_7_currentAcc_, validTokenGroups, _dafny.BigRational('2e0'), d_4_penaltyTokens_, _dafny.BigRational('4e0'), 12, eosToken)
                        d_9_next_ = out3_
                        d_5_steps_ = (d_5_steps_) + (1)
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("1_0_0")
                        elif True:
                            d_10_valid_: bool
                            out4_: bool
                            out4_ = (d_0_helpers_).IsTokenValidNext(parser, d_7_currentAcc_, d_9_next_)
                            d_10_valid_ = out4_
                            if d_10_valid_:
                                d_7_currentAcc_ = (d_7_currentAcc_) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                d_6_accum_ = (d_6_accum_) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                        pass
                pass
            generated = (generatedPrefix) + (d_6_accum_)
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            cost = d_5_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

