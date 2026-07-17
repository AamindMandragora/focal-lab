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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query in lowercase. Format must be exactly: SQL: <<your sql query>>. Use only the schema tables and columns listed. Write simple queries - avoid unnecessary JOINs. No explanation. No markdown. Example: SQL: <<select name from table where condition>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_seenSql_: bool
        d_2_seenSql_ = False
        d_3_seenColon_: bool
        d_3_seenColon_ = False
        d_4_unconstrainedCount_: int
        d_4_unconstrainedCount_ = 0
        d_5_maxUnconstrained_: int
        d_5_maxUnconstrained_ = 15
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_seenSql_) and (d_3_seenColon_):
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out0_
                            d_7_openedInside_ = out1_
                            d_8_openedCurrent_ = out2_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_seenSql_ = False
                            d_3_seenColon_ = False
                        elif (d_4_unconstrainedCount_) >= (d_5_maxUnconstrained_):
                            d_9_openedGenerated_: _dafny.Seq
                            d_10_openedInside_: bool
                            d_11_openedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_9_openedGenerated_ = out3_
                            d_10_openedInside_ = out4_
                            d_11_openedCurrent_ = out5_
                            generated = d_9_openedGenerated_
                            insideConstrainedOut = d_10_openedInside_
                            currentConstrainedOut = d_11_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_seenSql_ = False
                            d_3_seenColon_ = False
                        elif True:
                            d_12_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_4_unconstrainedCount_ = (d_4_unconstrainedCount_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                if (d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_seenSql_ = False
                                    d_3_seenColon_ = False
                                elif ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sql")))):
                                    d_2_seenSql_ = True
                                    d_3_seenColon_ = False
                                elif ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sql:")))):
                                    d_2_seenSql_ = True
                                    d_3_seenColon_ = True
                                elif (((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ": "))))) and (d_2_seenSql_):
                                    d_3_seenColon_ = True
                                elif (((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\t")))):
                                    pass
                                elif True:
                                    d_2_seenSql_ = False
                                    d_3_seenColon_ = False
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_closedGenerated_: _dafny.Seq
                        d_14_closedInside_: bool
                        d_15_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_closedGenerated_ = out7_
                        d_14_closedInside_ = out8_
                        d_15_closedCurrent_ = out9_
                        generated = d_13_closedGenerated_
                        insideConstrainedOut = d_14_closedInside_
                        currentConstrainedOut = d_15_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_17_queryLen_: int
                        d_17_queryLen_ = len(currentConstrainedOut)
                        d_18_next_: _dafny.Seq
                        d_18_next_ = eosToken
                        if (d_17_queryLen_) == (0):
                            d_19_startGroups_: _dafny.Seq
                            d_19_startGroups_ = (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "with")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WITH"))])])) + (validTokenGroups)
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, d_19_startGroups_, _dafny.BigRational('1e1'), eosToken)
                            d_18_next_ = out10_
                        elif (d_17_queryLen_) > (60):
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_18_next_ = out11_
                        elif True:
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), 20, eosToken)
                            d_18_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_18_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_20_appendedGenerated_: _dafny.Seq
                            d_21_appendedInside_: bool
                            d_22_appendedCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_20_appendedGenerated_ = out13_
                            d_21_appendedInside_ = out14_
                            d_22_appendedCurrent_ = out15_
                            generated = d_20_appendedGenerated_
                            insideConstrainedOut = d_21_appendedInside_
                            currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

