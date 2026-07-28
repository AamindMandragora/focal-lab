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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SQLite query using only the provided schema context. Output exactly: SQL: <<YOUR QUERY>>. No explanation, no Markdown, no extra text. Use exact table and column names from the schema. Prefer the simplest correct query, include joins only when required, and let the constrained decoder close the delimiters."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer schema identifiers from the contextual token groups whenever they are parser-valid.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_headerStage_: int
        d_2_headerStage_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 16
        d_4_sqlKeywordGroups_: _dafny.Seq
        d_4_sqlKeywordGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "IN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EXISTS"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "COUNT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SUM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AVG")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "MAX"))])])
        d_5_steps_: int
        d_5_steps_ = 0
        with _dafny.label("0"):
            while (d_5_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_headerStage_) == (0):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:"))]))
                            d_2_headerStage_ = 1
                            d_5_steps_ = (d_5_steps_) + (1)
                        elif (d_2_headerStage_) == (1):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
                            d_2_headerStage_ = 2
                            d_5_steps_ = (d_5_steps_) + (1)
                        elif True:
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
                            d_5_steps_ = (d_5_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_9_closedGenerated_: _dafny.Seq
                        d_10_closedInside_: bool
                        d_11_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_9_closedGenerated_ = out3_
                        d_10_closedInside_ = out4_
                        d_11_closedCurrent_ = out5_
                        generated = d_9_closedGenerated_
                        insideConstrainedOut = d_10_closedInside_
                        currentConstrainedOut = d_11_closedCurrent_
                        d_5_steps_ = (d_5_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_groups_: _dafny.Seq
                        d_13_groups_ = (d_4_sqlKeywordGroups_) + (validTokenGroups)
                        d_14_next_: _dafny.Seq
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, d_13_groups_, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                        d_14_next_ = out6_
                        d_5_steps_ = (d_5_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            pass
                        elif True:
                            d_15_valid_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_14_next_)
                            d_15_valid_ = out7_
                            if d_15_valid_:
                                d_16_appendedGenerated_: _dafny.Seq
                                d_17_appendedInside_: bool
                                d_18_appendedCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                d_16_appendedGenerated_ = out8_
                                d_17_appendedInside_ = out9_
                                d_18_appendedCurrent_ = out10_
                                generated = d_16_appendedGenerated_
                                insideConstrainedOut = d_17_appendedInside_
                                currentConstrainedOut = d_18_appendedCurrent_
                    pass
            pass
        cost = d_5_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

