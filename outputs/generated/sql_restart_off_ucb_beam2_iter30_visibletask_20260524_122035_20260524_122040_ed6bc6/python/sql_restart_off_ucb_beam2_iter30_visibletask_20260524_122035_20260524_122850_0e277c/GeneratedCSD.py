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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one line in this format and nothing else: SQL: <<query>>\n\nFormatting requirements for the query inside << >>:\n- Use lowercase SQL keywords: select, from, where, group by, order by, having, join, on, and, or, not, in, like, between, as, distinct, count, avg, sum, min, max, asc, desc, limit, intersect, union, except.\n- Put a space before and after every parenthesis: write 'count ( * )' not 'count(*)', and 'where x in ( select y from t )' not 'where x in (select y from t)'.\n- Put a space around comparison and arithmetic operators: '=', '<', '>', '<=', '>=', '!=', '+', '-'.\n- Use fully-qualified column names like table.column. Do NOT use aliases (no 'AS T1', no 'T1.col'). Repeat the table name instead.\n- Use only the tables and columns from the provided schema; do not invent joins or columns.\n- End immediately after >>. No trailing prose, no explanations, no code fences, no semicolons.\n\nExamples of correct outputs:\nSQL: <<select count ( * ) from continents>>\nSQL: <<select avg ( age ) from student where stuid not in ( select stuid from has_pet )>>\nSQL: <<select airlines.airline from airlines join flights on airlines.uid = flights.airline where flights.sourceairport = 'APG' intersect select airlines.airline from airlines join flights on airlines.uid = flights.airline where flights.sourceairport = 'CVO'>>\nSQL: <<select t_head.name from t_head where t_head.age > 56>>")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkBudget_: int
                        if (d_3_remaining_) > (8):
                            d_4_chunkBudget_ = 8
                        elif True:
                            d_4_chunkBudget_ = d_3_remaining_
                        d_5_chunkedG_: _dafny.Seq
                        d_6_stoppedOpen_: bool
                        d_7_stoppedEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkedG_ = out0_
                        d_6_stoppedOpen_ = out1_
                        d_7_stoppedEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif (d_8_stepsUsed_) == (0):
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_9_closedGenerated_: _dafny.Seq
                        d_10_closedInside_: bool
                        d_11_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_9_closedGenerated_ = out4_
                        d_10_closedInside_ = out5_
                        d_11_closedCurrent_ = out6_
                        generated = d_9_closedGenerated_
                        insideConstrainedOut = d_10_closedInside_
                        currentConstrainedOut = d_11_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_validCount_: int
                        out7_: int
                        out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_13_validCount_ = out7_
                        d_14_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (d_13_validCount_) <= (d_2_narrowThreshold_):
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), d_2_narrowThreshold_, eosToken)
                            d_14_next_ = out8_
                        elif True:
                            d_15_softNext_: _dafny.Seq
                            d_16_usedFallback_: bool
                            out9_: _dafny.Seq
                            out10_: bool
                            out9_, out10_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('6e0'), eosToken)
                            d_15_softNext_ = out9_
                            d_16_usedFallback_ = out10_
                            d_14_next_ = d_15_softNext_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_17_appendedGenerated_: _dafny.Seq
                            d_18_appendedInside_: bool
                            d_19_appendedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            d_17_appendedGenerated_ = out11_
                            d_18_appendedInside_ = out12_
                            d_19_appendedCurrent_ = out13_
                            generated = d_17_appendedGenerated_
                            insideConstrainedOut = d_18_appendedInside_
                            currentConstrainedOut = d_19_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

