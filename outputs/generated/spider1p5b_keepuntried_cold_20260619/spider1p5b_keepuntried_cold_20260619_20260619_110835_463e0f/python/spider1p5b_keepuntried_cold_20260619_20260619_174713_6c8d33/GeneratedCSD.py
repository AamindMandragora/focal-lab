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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single SQL query. Use SELECT, FROM, JOIN, WHERE, GROUP BY, ORDER BY, LIMIT as needed. Use only tables and columns from the schema. Prefer simple JOINs over nested subqueries. Output only the SQL inside the delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            generated = out0_
            insideConstrainedOut = out1_
            currentConstrainedOut = out2_
            d_1_steps_ = (d_1_steps_) + (1)
        d_2_spanTokens_: int
        d_2_spanTokens_ = 0
        d_3_maxSpanTokens_: int
        d_3_maxSpanTokens_ = 180
        with _dafny.label("0"):
            while (((d_1_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_4_cg_: _dafny.Seq
                    d_5_ci_: bool
                    d_6_cc_: _dafny.Seq
                    d_7_closed_: bool
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out6_: bool
                    out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_4_cg_ = out3_
                    d_5_ci_ = out4_
                    d_6_cc_ = out5_
                    d_7_closed_ = out6_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_7_closed_:
                        generated = d_4_cg_
                        insideConstrainedOut = d_5_ci_
                        currentConstrainedOut = d_6_cc_
                        raise _dafny.Break("0")
                    if (d_1_steps_) >= (maxSteps):
                        raise _dafny.Break("0")
                    if (d_2_spanTokens_) >= (d_3_maxSpanTokens_):
                        raise _dafny.Break("0")
                    d_8_penaltyLen_: int
                    if (len(currentConstrainedOut)) >= (3):
                        d_8_penaltyLen_ = 3
                    elif True:
                        d_8_penaltyLen_ = len(currentConstrainedOut)
                    if (d_8_penaltyLen_) > (0):
                        d_9_recentTokens_: _dafny.Seq
                        d_9_recentTokens_ = _dafny.SeqWithoutIsStrInference((currentConstrainedOut)[(len(currentConstrainedOut)) - (d_8_penaltyLen_)::])
                        (d_0_helpers_).SafePenalizeTokenLogits(lm, d_9_recentTokens_, _dafny.BigRational('2e0'))
                    d_10_constrainedPrompt_: _dafny.Seq
                    d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_11_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                    d_11_next_ = out7_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_11_next_) == (eosToken):
                        raise _dafny.Break("0")
                    d_12_ag_: _dafny.Seq
                    d_13_ai_: bool
                    d_14_ac_: _dafny.Seq
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                    d_12_ag_ = out8_
                    d_13_ai_ = out9_
                    d_14_ac_ = out10_
                    generated = d_12_ag_
                    insideConstrainedOut = d_13_ai_
                    currentConstrainedOut = d_14_ac_
                    d_2_spanTokens_ = (d_2_spanTokens_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

