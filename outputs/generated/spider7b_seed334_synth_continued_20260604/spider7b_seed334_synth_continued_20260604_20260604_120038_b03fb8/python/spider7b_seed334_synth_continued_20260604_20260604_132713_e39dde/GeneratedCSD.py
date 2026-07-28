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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQL query answering the question. Format: <<QUERY>> where QUERY is a valid SQL statement. Use only table/column names from the schema. Prefer the simplest correct query. Use WHERE for single-table filters. Use JOIN only when columns from multiple tables are needed. For N smallest/largest items use ORDER BY col ASC/DESC LIMIT N. For 'both' conditions use INTERSECT. For exclusion use NOT IN or EXCEPT. For grouped aggregates use GROUP BY with HAVING. Use COUNT(DISTINCT col) separately for each column when counting distinct values of multiple columns. Do not use ORDER BY or LIMIT after INTERSECT/UNION/EXCEPT."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        cost = d_2_steps_
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            d_4_g_k_: _dafny.Seq
                            d_5_ins_k_: bool
                            d_6_cur_k_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_4_g_k_ = out1_
                            d_5_ins_k_ = out2_
                            d_6_cur_k_ = out3_
                            generated = d_4_g_k_
                            insideConstrainedOut = d_5_ins_k_
                            currentConstrainedOut = d_6_cur_k_
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_7_g_k_: _dafny.Seq
                            d_8_ins_k_: bool
                            d_9_cur_k_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_g_k_ = out4_
                            d_8_ins_k_ = out5_
                            d_9_cur_k_ = out6_
                            generated = d_7_g_k_
                            insideConstrainedOut = d_8_ins_k_
                            currentConstrainedOut = d_9_cur_k_
                            d_2_steps_ = (d_2_steps_) + (1)
                            cost = d_2_steps_
                        elif (parser).IsDeadPrefix(currentConstrainedOut):
                            d_10_g_k_: _dafny.Seq
                            d_11_cur_k_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: _dafny.Seq
                            out7_, out8_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_10_g_k_ = out7_
                            d_11_cur_k_ = out8_
                            generated = d_10_g_k_
                            currentConstrainedOut = d_11_cur_k_
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_12_next_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, prompt, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_12_next_ = out9_
                            d_2_steps_ = (d_2_steps_) + (1)
                            cost = d_2_steps_
                            if (d_12_next_) == (eosToken):
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_13_valid_: bool
                                out10_: bool
                                out10_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_12_next_)
                                d_13_valid_ = out10_
                                if d_13_valid_:
                                    d_14_g_k_: _dafny.Seq
                                    d_15_ins_k_: bool
                                    d_16_cur_k_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                    d_14_g_k_ = out11_
                                    d_15_ins_k_ = out12_
                                    d_16_cur_k_ = out13_
                                    generated = d_14_g_k_
                                    insideConstrainedOut = d_15_ins_k_
                                    currentConstrainedOut = d_16_cur_k_
                    pass
            pass
        return generated, insideConstrainedOut, currentConstrainedOut, cost

