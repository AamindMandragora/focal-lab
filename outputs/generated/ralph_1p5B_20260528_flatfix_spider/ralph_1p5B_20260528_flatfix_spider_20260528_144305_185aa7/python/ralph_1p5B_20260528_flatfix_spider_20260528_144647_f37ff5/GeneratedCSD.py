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
        (d_0_helpers_).AppendTaskGuidance(lm, (((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer with exactly one line: SQL: <<YOUR SQL QUERY>>. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use only the tables and columns from the schema. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "No markdown, no explanation, no semicolon. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "The SQL must be inside << >> delimiters."))))
        d_1_seenFrom_: bool
        d_1_seenFrom_ = False
        d_2_seenWhere_: bool
        d_2_seenWhere_ = False
        d_3_steps_: int
        d_3_steps_ = 0
        with _dafny.label("0"):
            while (d_3_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_3_steps_)
                        d_5_chunkBudget_: int
                        if (d_4_remaining_) > (40):
                            d_5_chunkBudget_ = 40
                        elif True:
                            d_5_chunkBudget_ = d_4_remaining_
                        d_6_generatedOut_: _dafny.Seq
                        d_7_stoppedOnOpenSpan_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_generatedOut_ = out0_
                        d_7_stoppedOnOpenSpan_ = out1_
                        d_8_stoppedOnEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        d_3_steps_ = (d_3_steps_) + (d_9_stepsUsed_)
                        generated = d_6_generatedOut_
                        if d_8_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOnOpenSpan_:
                            d_10_g2_: _dafny.Seq
                            d_11_i2_: bool
                            d_12_c2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_10_g2_ = out4_
                            d_11_i2_ = out5_
                            d_12_c2_ = out6_
                            generated = d_10_g2_
                            insideConstrainedOut = d_11_i2_
                            currentConstrainedOut = d_12_c2_
                            d_1_seenFrom_ = False
                            d_2_seenWhere_ = False
                        elif (d_3_steps_) < (maxSteps):
                            d_13_g2_: _dafny.Seq
                            d_14_i2_: bool
                            d_15_c2_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_13_g2_ = out7_
                            d_14_i2_ = out8_
                            d_15_c2_ = out9_
                            generated = d_13_g2_
                            insideConstrainedOut = d_14_i2_
                            currentConstrainedOut = d_15_c2_
                            d_3_steps_ = (d_3_steps_) + (1)
                            d_1_seenFrom_ = False
                            d_2_seenWhere_ = False
                        elif True:
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out10_
                        d_17_closedInside_ = out11_
                        d_18_closedCurrent_ = out12_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_3_steps_ = (d_3_steps_) + (1)
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if not(d_1_seenFrom_):
                            d_21_sqlGroups_: _dafny.Seq
                            d_21_sqlGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "count")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "avg")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MAX")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LEFT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on"))])])
                            d_22_allGroups_: _dafny.Seq
                            d_22_allGroups_ = (d_21_sqlGroups_) + (validTokenGroups)
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, d_22_allGroups_, _dafny.BigRational('5e0'), eosToken)
                            d_20_next_ = out13_
                        elif True:
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_20_next_ = out14_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_23_appendedGenerated_: _dafny.Seq
                            d_24_appendedInside_: bool
                            d_25_appendedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_23_appendedGenerated_ = out15_
                            d_24_appendedInside_ = out16_
                            d_25_appendedCurrent_ = out17_
                            generated = d_23_appendedGenerated_
                            insideConstrainedOut = d_24_appendedInside_
                            currentConstrainedOut = d_25_appendedCurrent_
                            if ((d_20_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))) or ((d_20_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from")))):
                                d_1_seenFrom_ = True
                            if ((d_20_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")))) or ((d_20_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where")))):
                                d_2_seenWhere_ = True
                    pass
            pass
        cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

