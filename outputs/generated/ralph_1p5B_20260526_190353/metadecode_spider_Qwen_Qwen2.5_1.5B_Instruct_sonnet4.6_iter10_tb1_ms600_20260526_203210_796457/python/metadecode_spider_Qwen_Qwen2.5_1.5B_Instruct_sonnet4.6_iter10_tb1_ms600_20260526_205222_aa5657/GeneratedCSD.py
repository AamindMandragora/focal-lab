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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query using only the schema tables and columns. Format must be exactly: SQL: <<YOUR SQL QUERY HERE>>. No explanation. No markdown. No semicolons inside the query.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_seenSql_: bool
        d_2_seenSql_ = False
        d_3_seenColon_: bool
        d_3_seenColon_ = False
        d_4_unconstrainedSteps_: int
        d_4_unconstrainedSteps_ = 0
        d_5_maxUnconstrainedSteps_: int
        d_5_maxUnconstrainedSteps_ = 15
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_seenSql_) and (d_3_seenColon_)) or ((d_4_unconstrainedSteps_) >= (d_5_maxUnconstrainedSteps_)):
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
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_4_unconstrainedSteps_ = (d_4_unconstrainedSteps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_seenSql_ = False
                                    d_3_seenColon_ = False
                                elif ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL")))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sql")))):
                                    d_2_seenSql_ = True
                                    d_3_seenColon_ = False
                                elif ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sql:")))):
                                    d_2_seenSql_ = True
                                    d_3_seenColon_ = True
                                elif (((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ": "))))) and (d_2_seenSql_):
                                    d_3_seenColon_ = True
                                elif ((((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ")))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\t"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ")))):
                                    pass
                                elif True:
                                    d_2_seenSql_ = False
                                    d_3_seenColon_ = False
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_closedGenerated_: _dafny.Seq
                        d_11_closedInside_: bool
                        d_12_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_closedGenerated_ = out4_
                        d_11_closedInside_ = out5_
                        d_12_closedCurrent_ = out6_
                        generated = d_10_closedGenerated_
                        insideConstrainedOut = d_11_closedInside_
                        currentConstrainedOut = d_12_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_14_queryLen_: int
                        d_14_queryLen_ = len(currentConstrainedOut)
                        d_15_next_: _dafny.Seq
                        d_15_next_ = eosToken
                        if (d_14_queryLen_) == (0):
                            d_16_startGroups_: _dafny.Seq
                            d_16_startGroups_ = (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WITH")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "with"))])])) + (validTokenGroups)
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, d_16_startGroups_, _dafny.BigRational('8e0'), eosToken)
                            d_15_next_ = out7_
                        elif (d_14_queryLen_) > (60):
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                            d_15_next_ = out8_
                        elif True:
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), 50, eosToken)
                            d_15_next_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_17_appendedGenerated_: _dafny.Seq
                            d_18_appendedInside_: bool
                            d_19_appendedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                            d_17_appendedGenerated_ = out10_
                            d_18_appendedInside_ = out11_
                            d_19_appendedCurrent_ = out12_
                            generated = d_17_appendedGenerated_
                            insideConstrainedOut = d_18_appendedInside_
                            currentConstrainedOut = d_19_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

